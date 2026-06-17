# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
    MindSpeedOptimizerConfig,
    MindSpeedEngineConfig,
)

from verl.utils.model import print_model_size
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.model import convert_weight_keys

from ..base import EngineRegistry, BaseEngine

try:
    from ..megatron import MegatronEngineWithLMHead
except ImportError:
    MegatronEngineWithLMHead = BaseEngine

try:
    from ..fsdp import FSDPEngineWithLMHead
except ImportError:
    FSDPEngineWithLMHead = BaseEngine

from .utils import (
    MCORE_SUPPORT_LLM_MODELS,
    MCORE_SUPPORT_MM_MODELS,
    FSDP_SUPPORT_LLM_MODELS,
    FSDP_SUPPORT_MM_MODELS,
    apply_patch,
    gpt_model_provider,
    get_fsdp_trainer,
    move_buffers_to_device_recursive,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        if repatch is not None:
            repatch_config = dict(self.engine_config.get("override_transformer_config", {}))
            repatch_config.setdefault("use_flash_attn", True)
            if self.engine_config.context_parallel_size > 1:
                repatch_config["context_parallel_size"] = self.engine_config.context_parallel_size
            repatch(repatch_config)
        super()._init_device_mesh()


@EngineRegistry.register(model_type="language_model", backend="mindspeed_megatron", device="npu")
class MindSpeedMegatronEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MindSpeedEngineConfig,
        optimizer_config: MindSpeedOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        self.is_llm_model = engine_config.model_name in MCORE_SUPPORT_LLM_MODELS
        self.is_mm_model = engine_config.model_name in MCORE_SUPPORT_MM_MODELS
        if not self.is_llm_model and not self.is_mm_model:
            raise ValueError(f"{engine_config.model_name} is not supported for mindspeed_megatron backend now")
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        if self.is_llm_model:
            apply_patch(self.model_config, self.engine_config, self.optimizer_config)
        elif self.is_mm_model:
            raise ValueError(f"mm_model is not supported for mindspeed_megatron backend now")
        super()._init_device_mesh()

    def _build_megatron_module(self):
        if self.is_llm_model:
            is_value_model = (
                    "ForTokenClassification" in self.model_config.architectures[0]
                    or "ForSequenceClassification" in self.model_config.architectures[0]
            )

            self.is_value_model = is_value_model

            import torch.distributed
            from megatron.core.enums import ModelType
            from megatron.training.training import get_model

            # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
            if self.engine_config.forward_only:
                module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
            else:
                module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                raise ValueError(f"vanilla_bridge should be true now, but got {self.vanilla_bridge}")

            if torch.distributed.get_rank() == 0:
                print_model_size(module[0])

            if self.enable_routing_replay:
                from verl.utils.megatron.router_replay_patch import RouterReplay
                print(f"routing replay layers: {len(RouterReplay.router_instances)}")

            return module

        elif self.is_mm_model:
            raise ValueError(f"mm_model is not supported for mindspeed_megatron backend now")


@EngineRegistry.register(model_type="language_model", backend="mindspeed_fsdp", device="npu")
class MindSpeedFSDPEngineWithLMHead(FSDPEngineWithLMHead):
    def __init__(
            self,
            model_config: HFModelConfig,
            engine_config: MindSpeedEngineConfig,
            optimizer_config: MindSpeedOptimizerConfig,
            checkpoint_config: CheckpointConfig,
    ):
        self.is_llm_model = engine_config.model_name in FSDP_SUPPORT_LLM_MODELS
        self.is_mm_model = engine_config.model_name in FSDP_SUPPORT_MM_MODELS
        if not self.is_llm_model and not self.is_mm_model:
            raise ValueError(f"{engine_config.model_name} is not supported for mindspeed_fsdp backend now")
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _log_backend_batch_info(
        self,
        local_batch_size_per_dp: int,
        micro_batch_sizes_per_dp: list[int],
        gradient_accumulation_steps: int,
    ) -> None:
        from mindspeed_mm.fsdp.utils.batch_debug import log_verl_batch_info

        log_verl_batch_info(
            local_batch_size_per_dp=local_batch_size_per_dp,
            micro_batch_sizes_per_dp=micro_batch_sizes_per_dp,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def _patch_mindspeed_mm_padded_cp(self):
        if not self.is_mm_model:
            return
        if self.ulysses_sequence_parallel_size <= 1:
            return
        if self.model_config.get("use_remove_padding", False):
            return

        import importlib

        def generate_padded_cu_seqlen_params(position_ids, need_cpu_tensor=True):
            # MindSpeed-MM Qwen3.5 defaults to packed 1NTD/TND when CP cu_seqlens
            # are present. In verl padded mode, tensors are [B, S], so do not
            # expose actual-token boundaries and let MindSpeed-MM fall back to BNSD.
            seq_len = position_ids.shape[-1]

            return {
                "cu_seq_lens_q": None,
                "cu_seq_lens_k": None,
                "max_length_q": seq_len,
                "max_length_k": seq_len,
            }

        module_names = [
            "mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5",
            "mindspeed_mm.fsdp.models.qwen3_5_moe.modeling_qwen3_5_moe",
        ]
        patched_modules = []
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generate_ulysses_cu_seqlen_params"):
                module.generate_ulysses_cu_seqlen_params = generate_padded_cu_seqlen_params
                patched_modules.append(module_name)

        if patched_modules:
            logger.warning(
                "Patched MindSpeed-MM padded CP cu_seqlens for use_remove_padding=False, "
                "ulysses_sequence_parallel_size=%s, modules=%s",
                self.ulysses_sequence_parallel_size,
                patched_modules,
            )

    def _build_model_optimizer(self):
        if self.is_llm_model:
            raise ValueError(f"llm_model is not supported for mindspeed_fsdp backend now")
        elif self.is_mm_model:
            # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
            import torch
            torch.distributed.barrier()

            log_gpu_memory_usage("Before FSDP", logger=None)

            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
            self.trainer, self.mm_args = get_fsdp_trainer(self.model_config, self.engine_config, self.optimizer_config)
            self._patch_mindspeed_mm_padded_cp()
            module = self.trainer.model
            log_gpu_memory_usage("After FSDP", logger=None)

            if self.rank == 0:
                print_model_size(module)
            log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

            if not self.engine_config.forward_only:
                # Initialize optimizer with model parameters and config settings
                optimizer = self.trainer.optimizer
                # Create learning rate scheduler with warmup and decay settings
                lr_scheduler = self.trainer.lr_scheduler
            else:
                optimizer = None
                lr_scheduler = None

            self.module = module
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu,
            offload_fsdp_model_to_cpu,
            collect_lora_params,
            merged_lora_context,
            normalize_peft_param_name,
            replace_lora_wrapper,
        )

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
        else:
            if self._is_offload_param:
                params = self.module.state_dict()
            else:
                move_buffers_to_device_recursive(self.module, "cpu")
                params = self.module.state_dict()
                move_buffers_to_device_recursive(self.module, "npu")

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
        params = {k.replace("module.", ""): v for k, v in params.items()}

        for k in list(params.keys()):
            if "mlp.experts.gate_up_proj" in k or "mlp.experts.down_proj" in k:
                print(f"--> modify shape {k}")
                params[k] = params[k].transpose(1, 2).contiguous()

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            import torch
            from torch.distributed.tensor import DTensor
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu,
            offload_fsdp_model_to_cpu,
        )
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        state = {
            "model": self.module,
            "extra_state": {
                "iteration": global_step,
                "consumed_train_samples": 0,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
        }
        if not self.mm_args.training.no_save_optim:
            state["optimizer"] = self.optimizer
        if not self.mm_args.training.no_save_rng:
            state["extra_state"]["torch_rng_state"] = torch.get_rng_state()

        self.trainer.checkpointer.save(
            path=local_path,
            state=state,
            iteration=global_step,
            save_async=self.mm_args.training.save_async,
        )

        if max_ckpt_to_keep is not None:
            self._cleanup_old_checkpoints(local_path, max_ckpt_to_keep)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu,
            offload_fsdp_model_to_cpu,
            offload_fsdp_optimizer,
        )
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        state = {"model": self.module, "extra_state": {}}
        if not self.mm_args.training.no_load_optim:
            state["optimizer"] = self.optimizer

        release = self.trainer.checkpointer.load(
            path=local_path,
            state=state,
            load_rank0_and_broadcast=self.mm_args.training.load_rank0_and_broadcast,
            load_strict=self.mm_args.training.load_strict,
        )

        if not release and "extra_state" in state:
            if "lr_scheduler" in state["extra_state"]:
                self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            if not self.mm_args.training.no_load_rng and "torch_rng_state" in state["extra_state"]:
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def _cleanup_old_checkpoints(self, base_path, max_to_keep):
        iter_pattern = re.compile(r"iter_(\d+)")
        iter_dirs = []
        for name in os.listdir(base_path):
            match = iter_pattern.match(name)
            if match:
                iter_dirs.append((int(match.group(1)), os.path.join(base_path, name)))

        iter_dirs.sort(key=lambda x: x[0])
        while len(iter_dirs) > max_to_keep:
            _, old_dir = iter_dirs.pop(0)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
