#!/usr/bin/env bash
# On-policy distillation | text | SGLang/vLLM rollout | MindSpeed-MM FSDP training | NPU

# ray stop --force
# ps -aux | grep "VLLM" | grep -v grep| awk '{print $2}' | xargs kill -9 | pkill -9 python

# set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# Mindspeed-MM fsdp config
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export OMP_NUM_THREADS=1

# ---- NPU env ----------------------------------------------------------------
# 8 dies expose 16 logical NPUs on this machine. Use the first 4 dies by default:
# die0=(0,1), die1=(2,3), die2=(4,5), die3=(6,7).
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
VISIBLE_DEVICE_COUNT=$(awk -F',' '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")
# original: VLLM_ATTENTION_BACKEND=ASCEND
export VLLM_ATTENTION_BACKEND=ASCEND
# original: VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_ENABLE_NZ=0
# Keep verl's vLLM-Ascend A2 MC2 workaround enabled. The extended MoE weight
# loader patch is opt-in because standalone vLLM-Ascend Qwen3.5 MoE can run
# normally on some stacks, and the co-location issue is more likely lifecycle-
# related than a universal fused-MoE loader issue.
export VERL_NPU_ENABLE_VLLM_ASCEND_MOE_WEIGHT_LOADER_PATCH=${VERL_NPU_ENABLE_VLLM_ASCEND_MOE_WEIGHT_LOADER_PATCH:-0}
export VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2=${VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2:-1}
STUDENT_MODEL=/home/s00525112/model/Qwen3.5-2B
TEACHER_MODEL=/home/s00525112/model/Qwen3.5-35B-A3B
DCP_MODEL_PATH="/home/s00525112/model/Qwen3.5-2B-dcp"
sp_size=1

# export ASCEND_LAUNCH_BLOCKING=1

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${VISIBLE_DEVICE_COUNT}}
COLOCATE_WITH_ACTOR_ROLLOUT=${COLOCATE_WITH_ACTOR_ROLLOUT:-True}
# vLLM/NPU currently cannot full-unload teacher weights with sleep(level=2).
# teacher_sleep_level=0 keeps the teacher engine resident and skips vLLM sleep/wake;
# use 1 to release cache memory only if your vLLM-Ascend stack is stable with it.
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
VLLM_TEACHER_SLEEP_LEVEL=${VLLM_TEACHER_SLEEP_LEVEL:-0}

case "${COLOCATE_WITH_ACTOR_ROLLOUT}" in
    True|true|1|yes|YES|on|ON)
        TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-${NGPUS_PER_NODE}}
        if [ "${TEACHER_WORLD_SIZE}" != "${NGPUS_PER_NODE}" ]; then
            echo "When COLOCATE_WITH_ACTOR_ROLLOUT=True, TEACHER_WORLD_SIZE must equal NGPUS_PER_NODE." >&2
            exit 1
        fi
        if [ "${ROLLOUT_BACKEND}" = "vllm" ]; then
            case "${VLLM_TEACHER_SLEEP_LEVEL}" in
                0|1) ;;
                *)
                    echo "NPU vLLM teacher co-location requires VLLM_TEACHER_SLEEP_LEVEL=0 or 1." >&2
                    exit 1
                    ;;
            esac
        fi
        ;;
    *)
        TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-8}
        ;;
esac

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# GRAD_ACC = PPO_MINI_BATCH_SIZE / ( n_gpus x PPO_MICRO_BATCH_SIZE_PER_GPU )
train_batch_size=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE:-1}
GRAD_ACCU_STEPS=$((PPO_MINI_BATCH_SIZE / NGPUS_PER_NODE / PPO_MICRO_BATCH_SIZE_PER_GPU))
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}
DATA_LOAD_SEED=${DATA_LOAD_SEED:-42}


actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
# Start conservatively for 8-card co-location. vLLM's utilization mostly caps
# KV/cache allocation; model weights and student training peaks still need room.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.25}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
teacher_tp=${TEACHER_TP:-${NGPUS_PER_NODE}}
# Keep vLLM-Ascend MoE expert parallel disabled by default for Qwen3.5-35B-A3B.
# The EP path can hit fused MoE grouped-matmul shape mismatches on NPU; TP=8
# still uses all 8 visible logical NPUs for the teacher replica.
teacher_ep=${TEACHER_EP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.25}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_distill_gsm8k}
experiment_name=${EXPERIMENT_NAME:-qwen35_2b_from_qwen35_35b_mm_fsdp2}
start_time=$(date +%Y%m%d)_$(date +%H%M%S)

# profiling configuration
PROFILE_STEPS=${PROFILE_STEPS:-"[2]"}
PROFILE_RANKS_ALL=${PROFILE_RANKS_ALL:-False}
DISCRETE=${DISCRETE:-True}
if [ -z "${PROFILE_RANKS+x}" ]; then
    PROFILE_RANKS="["
    for ((rank = 0; rank < NGPUS_PER_NODE; rank++)); do
        if [ "$rank" -gt 0 ]; then
            PROFILE_RANKS+=","
        fi
        PROFILE_RANKS+="$rank"
    done
    PROFILE_RANKS+="]"
fi

# profiling NPU options
SAVE_PATH=${SAVE_PATH:-"./profile_data/${start_time}"}
LEVEL=${LEVEL:-"level0"}
CONTENTS=${CONTENTS:-"['npu','cpu','stack']"}
ANALYSIS=${ANALYSIS:-True}
# ---- end user-adjustable ----

train_data=/home/s00525112/data/gsm8k/train.parquet
test_data=/home/s00525112/data/gsm8k/test.parquet

train_files="['$train_data']"
val_files="['$test_data']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='right'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_init=0
    actor_rollout_ref.actor.optim.lr_warmup_steps=20
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.total_training_steps=400
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=${DATA_LOAD_SEED}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.calculate_log_probs=True
)

MINDSPEED_CONFIG=(
    actor_rollout_ref.actor.mindspeed.strategy=mindspeed_fsdp
    actor_rollout_ref.actor.mindspeed.model_name=qwen3.5-27b
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.load=${DCP_MODEL_PATH}
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.plugin='[mindspeed_mm/fsdp/models/qwen3_5, mindspeed_mm/fsdp/data/datasets/huggingface]'
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.micro_batch_size=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.gradient_accumulation_steps=${GRAD_ACCU_STEPS}
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.model_id=qwen3_5
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.use_triton_gdn=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.freeze='[model.visual]'
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.apply_modules="['model.visual', \
    'model.visual.blocks.{*}', 'model.language_model', 'model.language_model.embed_tokens', \
    'model.language_model.layers.{*}', 'lm_head']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.hook_modules="['model.language_model.layers.{*}']"
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.param_dtype=bf16 
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.reduce_dtype=fp32 # gradient allreduce 用 fp32 更稳 
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.output_dtype=bf16 
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.num_to_forward_prefetch=1
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.num_to_backward_prefetch=1
    +actor_rollout_ref.ref.mindspeed.fsdp_kwargs.parallel.fsdp_plan.num_to_forward_prefetch=1
    +actor_rollout_ref.ref.mindspeed.fsdp_kwargs.parallel.fsdp_plan.num_to_backward_prefetch=1
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute_plan.apply_modules="['model.language_model.layers.{*}']"
    actor_rollout_ref.actor.mindspeed.ulysses_sequence_parallel_size=$sp_size
    actor_rollout_ref.ref.mindspeed.ulysses_sequence_parallel_size=$sp_size
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.offload_policy=True
    actor_rollout_ref.ref.mindspeed.param_offload=True
    actor_rollout_ref.ref.mindspeed.optimizer_offload=False
    actor_rollout_ref.ref.mindspeed.offload_policy=True
    actor_rollout_ref.actor.optim.optimizer=adamw
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=400
)

PROFILER=(
    global_profiler.tool=npu
    global_profiler.steps=${PROFILE_STEPS}
    global_profiler.save_path=${SAVE_PATH}
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=${PROFILE_RANKS}
    actor_rollout_ref.actor.profiler.all_ranks=${PROFILE_RANKS_ALL}
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=${DISCRETE}
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=${CONTENTS}
    actor_rollout_ref.actor.profiler.tool_config.npu.level=${LEVEL}
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${ANALYSIS}
    actor_rollout_ref.ref.profiler.enable=True
    actor_rollout_ref.ref.profiler.ranks=${PROFILE_RANKS}
    actor_rollout_ref.ref.profiler.all_ranks=${PROFILE_RANKS_ALL}
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete=${DISCRETE}
    actor_rollout_ref.ref.profiler.tool_config.npu.contents=${CONTENTS}
    actor_rollout_ref.ref.profiler.tool_config.npu.level=${LEVEL}
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${ANALYSIS}
)

EXTRA=(
    distillation.enabled=True
    distillation.colocate_with_actor_rollout=${COLOCATE_WITH_ACTOR_ROLLOUT}
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.expert_parallel_size=${teacher_ep}
    distillation.teacher_models.teacher_model.inference.name=${ROLLOUT_BACKEND}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${max_num_tokens}
    distillation.teacher_models.teacher_model.inference.dtype=bfloat16
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.teacher_models.teacher_model.inference.enforce_eager=True
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=False
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False
    distillation.teacher_models.teacher_model.inference.max_num_seqs=16
)

if [ "${ROLLOUT_BACKEND}" = "sglang" ]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend
        actor_rollout_ref.rollout.enable_chunked_prefill=False
    )
    EXTRA+=(
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.attention_backend=ascend
        distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=False
    )
elif [ "${ROLLOUT_BACKEND}" = "vllm" ]; then
    EXTRA+=(
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.teacher_sleep_level=${VLLM_TEACHER_SLEEP_LEVEL}
    )
fi


########################### launch ###########################
mkdir -p logs
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${PROFILER[@]}" \
    "${EXTRA[@]}" \
    "${MINDSPEED_CONFIG[@]}" \
    "$@" 2>&1 | tee logs/qwen3_5-2b-35b-mm-fsdp-1kto1k-${start_time}.log
