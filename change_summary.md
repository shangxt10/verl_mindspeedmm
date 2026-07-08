# VERL对接MindSpeed-MM进行OPD适配修改

## 修改历史

- 分支：`cp_bug_fix`
- 基准提交：`e066848231f7ae3f32255e5e555ea239b9387890`

## 功能性修改汇总

功能性修改主要包括 MindSpeed 后端重构和 MindSpeed-MM FSDP 适配两个主要方向：

1. 将原有 MindSpeed 后端拆分为 `mindspeed_megatron` 和 `mindspeed_fsdp` 两类 engine strategy。
2. 新增 MindSpeed-MM FSDP engine，用于在 NPU 上训练 Qwen3.5 多模态模型。
3. 新增 MindSpeed optimizer 配置，并把 verl 的优化器字段转换为 MindSpeed-MM FSDP Trainer 实际使用的字段。
4. 修复/增强 packed、padding、3D position ids、VLM fused labels、checkpoint merge 等多模态训练路径。
5. 增加 Qwen3.5-27B / Qwen3.5-35B MindSpeed-MM FSDP GRPO 启动脚本。
6. 更新原 MindSpeed-LLM 示例脚本，迁移到新的 `mindspeed_megatron` / `mcore_kwargs` 配置结构。

## 代码修改点

### 1. MindSpeed engine strategy 重构（关键修改）

相关文件：

- `verl/workers/config/engine.py`
- `verl/workers/engine/mindspeed/transformer_impl.py`
- `verl/workers/engine/mindspeed/utils.py`
- `verl/workers/engine/__init__.py`
- `verl/workers/engine/mindspeed/__init__.py`
- `verl/trainer/config/engine/mindspeed.yaml`
- `verl/trainer/config/actor/mindspeed_actor.yaml`
- `verl/trainer/config/critic/mindspeed_critic.yaml`
- `verl/trainer/config/ref/mindspeed_ref.yaml`

修改点：

- `MindSpeedEngineConfig` 从只继承 `McoreEngineConfig` 改为同时继承 `McoreEngineConfig` 和 `FSDPEngineConfig`。
- `strategy` 从旧的 `mindspeed_llm` / `mindspeed_mm` 调整为 `mindspeed_megatron` / `mindspeed_fsdp`。
- 新增 `model_name` 字段，用于区分当前 MindSpeed 后端支持的模型类型。
- 原 `llm_kwargs` / `mm_kwargs` 拆分并重命名为：
  - `mcore_kwargs`：MindSpeed Megatron/MCore 后端配置。
  - `fsdp_kwargs`：MindSpeed-MM FSDP 后端配置。
- `mindspeed_actor.yaml`、`mindspeed_critic.yaml`、`mindspeed_ref.yaml` 默认优化器从 `megatron` 切换为 `mindspeed`。
- MindSpeed engine 导出类从 `MindSpeedLLMEngineWithLMHead` 调整为：
  - `MindSpeedMegatronEngineWithLMHead`
  - `MindSpeedFSDPEngineWithLMHead`

作用：

- 让 MindSpeed 后端可以同时承载 Megatron/MCore LLM 路径和 MindSpeed-MM FSDP 多模态路径。
- 避免旧配置中 `mindspeed_llm` / `mindspeed_mm` 语义混在一起，便于针对不同后端做模型支持校验和初始化。
- 让 actor、critic、ref 都可以统一使用新的 MindSpeed optimizer 和 engine 配置。

### 2. 新增 MindSpeed-MM FSDP engine（关键修改）

相关文件：

- `verl/workers/engine/mindspeed/transformer_impl.py`
- `verl/workers/engine/mindspeed/utils.py`

修改点：

- 新增 `MindSpeedFSDPEngineWithLMHead`，注册为：

```python
@EngineRegistry.register(model_type="language_model", backend="mindspeed_fsdp", device="npu")
```

- 支持模型白名单：
  - `FSDP_SUPPORT_MM_MODELS = ["qwen3.5-27b", "qwen3.5-35b"]`
  - 当前 `mindspeed_fsdp` 只支持多模态模型，不支持纯 LLM。
- 通过 `get_fsdp_trainer()` 调用 MindSpeed-MM 的 `Trainer` 初始化 FSDP 模型、优化器和 scheduler。
- 根据 `offload_policy` / `forward_only` 设置 MindSpeed-MM FSDP 的 CPU offload。
- 将 verl 侧 `model_config.path` 写入 MindSpeed-MM 的 `mm_args.model.model_name_or_path`。
- 支持 MindSpeed-MM checkpoint save/load：
  - `save_checkpoint()`
  - `load_checkpoint()`
  - `_cleanup_old_checkpoints()`
- 实现 `get_per_tensor_param()`，用于从 MindSpeed-MM FSDP 模型导出权重并同步给 rollout/ref 等组件。
- 对 MoE expert 权重做 shape 适配：`mlp.experts.gate_up_proj` 和 `mlp.experts.down_proj` 执行 transpose。

作用：

- 让 verl 可以用 `model_engine=mindspeed` + `actor_rollout_ref.actor.mindspeed.strategy=mindspeed_fsdp` 启动 MindSpeed-MM FSDP 多模态训练。
- 支持 Qwen3.5-27B、Qwen3.5-35B-A3B 等 MindSpeed-MM 模型在 NPU 上参与 GRPO 训练。
- 支持 FSDP 参数 offload、optimizer offload、checkpoint 保存/加载、权重同步到 rollout。

### 3. MindSpeed Megatron engine 重命名和模型校验

相关文件：

- `verl/workers/engine/mindspeed/transformer_impl.py`
- `verl/workers/engine/mindspeed/utils.py`

修改点：

- 将原 `MindSpeedLLMEngineWithLMHead` 重命名/调整为 `MindSpeedMegatronEngineWithLMHead`。
- 注册 backend 从 `mindspeed_llm` 改为 `mindspeed_megatron`。
- 新增支持模型列表：
  - `MCORE_SUPPORT_LLM_MODELS = ["qwen3-moe-30b", "qwen3-32b", "qwen3-8b"]`
  - `MCORE_SUPPORT_MM_MODELS = []`
- 在初始化时根据 `engine_config.model_name` 做模型支持校验。
- 对 MindSpeed Megatron 路径只允许 LLM 模型，多模态模型直接报错。
- `get_base_mcore_config_from_engine_config()` 统一读取 `engine_config.mcore_kwargs`。

作用：

- 明确 MindSpeed Megatron 路径只服务 Qwen3/Qwen3-MoE LLM。
- 为 MindSpeed-MM FSDP 多模态路径腾出独立 strategy，减少配置混淆。

### 4. 新增 MindSpeed optimizer 配置（关键修改）

相关文件：

- `verl/workers/config/optimizer.py`
- `verl/trainer/config/optim/mindspeed.yaml`
- `verl/workers/engine/mindspeed/utils.py`

修改点：

- 新增 `MindSpeedOptimizerConfig`，继承基础 `OptimizerConfig`。
- 新增字段：
  - `optimizer`
  - `lr_warmup_init`
  - `lr_warmup_ratio`
  - `lr_decay_steps`
  - `lr_decay_style`
  - `min_lr`
  - `weight_decay_incr_style`
  - `lr_wsd_decay_style`
  - `lr_wsd_decay_steps`
  - `use_checkpoint_opt_param_scheduler`
  - `override_optimizer_config`
- 新增默认配置文件 `verl/trainer/config/optim/mindspeed.yaml`。
- `get_base_mcore_config_from_optim_config()` 改为接收 `MindSpeedOptimizerConfig`。
- `get_fsdp_trainer()` 中把 verl optimizer 配置转换为 MindSpeed-MM FSDP Trainer 字段：
  - `mm_args.training.lr`
  - `mm_args.training.lr_decay_style`
  - `mm_args.training.train_iters`
  - `mm_args.training.lr_warmup_ratio`
  - `mm_args.training.weight_decay`
  - `mm_args.training.clip_grad`
  - `mm_args.training.optimizer`
- 新增 `_get_mindspeed_fsdp_total_steps()` 和 `_get_mindspeed_fsdp_warmup_ratio()`，用于把 `lr_warmup_steps` / `lr_warmup_steps_ratio` / `lr_warmup_ratio` 转成 MindSpeed-MM 所需配置。

作用：

- 修复 verl actor optim 配置没有正确传递到 MindSpeed-MM FSDP Trainer 的问题。
- 让 MindSpeed-MM 的训练步数、warmup、学习率、weight decay、梯度裁剪等行为和 verl 配置保持一致。

### 5. MindSpeed-MM FSDP padded CP / packed cu_seqlens 适配（关键修改）

相关文件：

- `verl/workers/engine/mindspeed/transformer_impl.py`

修改点：

- 新增 `_patch_mindspeed_mm_padded_cp()`：
  - 当 `use_remove_padding=False` 且 `ulysses_sequence_parallel_size > 1` 时，patch MindSpeed-MM Qwen3.5 / Qwen3.5-MoE 的 `generate_ulysses_cu_seqlen_params()`。
  - 对 padded tensor 返回 `cu_seq_lens_q=None` / `cu_seq_lens_k=None`，让 MindSpeed-MM 走 BNSD，而不是误判为 packed TND/1NTD。
- 新增 `_gather_mindspeed_mm_padded_cp_outputs()`：
  - 在 padded CP 场景下，如果 MindSpeed-MM 输出被 CP 切分，则对 `logits`、`log_probs`、`entropy` 做 gather。
  - 通过 `gather_forward_split_backward_with_cp()` 恢复到 full sequence 长度。
- `prepare_model_outputs()` 先 gather padded CP outputs，再走 FSDP 基类输出处理。
- 对 `use_remove_padding=True` 场景，把 packed 样本真实边界从 verl 传给 MindSpeed-MM，使 MM 端 `_explicit_cu_seqlen_params()` 能拿到正确 `cu_seqlens`。

作用：

- 修复 MindSpeed-MM 在 padded CP 下把 `[B, S]` 输入误当 packed TND/1NTD，导致 `not support input_layout TND with dim_num 4` 等错误。
- 修复 CP 切分后输出长度和 verl 后处理期望长度不一致的问题。
- 修复 `use_remove_padding=True` 且 micro batch size 大于 1 时，packed forward 被当成单条长序列处理的问题。

### 6. FSDP VLM / packed 输入增强

相关文件：

- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/utils/model.py`

修改点：

- 新增 `is_vision_language_model()`，用于判断 module/config 是否包含 `vision_config`。
- `FSDPEngineWithLMHead.prepare_model_inputs()` 中：
  - 在 remove-padding 路径下向模型输入额外传入 `cu_seqlens=input_ids.offsets().to(dtype=torch.int32)`。
  - 对 3D `position_ids` 不再硬编码 rope dim 为 4，而是读取 `position_ids.shape[1]`。
  - fused kernels + Ulysses SP + VLM 场景下，将已经 shift 且按 SP 切分的 labels 传给模型。
- `get_per_tensor_param()` 中只有开启 param offload 时才调用 `load_fsdp_model_to_gpu()`。

作用：

- 支持 VLM 模型在 packed/remove-padding 路径下拿到真实样本边界。
- 修复 VLM Ulysses SP 场景下 fused forward 重新 roll labels 造成 shard 边界错位的问题。
- 支持 rope dim 不固定为 4 的 3D position ids。
- 避免非 offload 场景下不必要地把 FSDP 模型强制 load 到 GPU/NPU。

### 7. 3D position ids 修复

相关文件：

- `verl/utils/tensordict_utils.py`

修改点：

- 重写 `maybe_fix_3d_position_ids()` 的修复逻辑。
- 原逻辑只是在 3D nested `position_ids` 上强制设置 `_ragged_idx = 2`。
- 新逻辑会检查 TensorDict consolidate / pickle 后的 nested tensor 是否出现 offsets/values 布局异常。
- 如果检测到 broken layout：
  - 优先从 nested `input_ids.offsets()` 获取目标序列边界。
  - 将 values reshape 回 `[batch, rope_dim, max_seq_len]`。
  - 按每个样本真实长度重新拼接为 jagged values。
  - 使用 `torch.nested.nested_tensor_from_jagged(..., jagged_dim=2)` 重建 position_ids。
- 如果无法安全修复，则保留 `_ragged_idx = 2` 的兼容路径。

作用：

- 修复 VLM 3D position ids 在 TensorDict 序列化/反序列化后 nested tensor 元数据异常导致的 indexing / unbind / split 错误。
- 对 batch size 小、各样本 seq_len 相同、mrope 等更容易触发 ragged 维度误判的场景更稳。

### 8. Agent loop 多模态 position ids 兼容

相关文件：

- `verl/experimental/agent_loop/agent_loop.py`

修改点：

- `_postprocess_multi_modal_inputs()` 允许 `output.multi_modal_data` 为空。
- 没有 image/video 输入时直接返回已有 multi-modal inputs。
- 计算 position ids 时区分：
  - 是否有 image token。
  - 是否有 video token。
  - 是否有对应 grid 信息。
- 纯文本数据但 processor 存在时，不再强行调用多模态 rope 逻辑，而是构造兼容 Qwen3-VL 的 4 维 position ids。
- 使用 `inspect.signature()` 判断 `processor.get_rope_index()` 是否支持 `mm_token_type_ids`，避免不同 transformers 版本签名不一致导致报错。
- 生成 `text_position_ids` 时保持 device 和 `input_ids.device` 一致。

作用：

- 修复 Qwen3-VL agent loop 在纯文本样本或缺少 image/video grid 时的 position ids 计算问题。
- 提升多模态/纯文本混合数据在 agent loop 中的兼容性。

### 9. Distillation / forward KL top-k 修复（关键修改）

相关文件：

- `verl/trainer/distillation/fsdp/losses.py`
- `verl/trainer/distillation/losses.py`

修改点：

- FSDP top-k KL 中对 `student_topk_log_probs` 和 `teacher_topk_log_probs` 重新执行 `F.log_softmax(..., dim=-1)`。
- 新增 `_resolve_topk_loss_strategy()`：
  - 当 actor strategy 为 `mindspeed` 时，根据 `config.engine.strategy` 或 `config.mindspeed.strategy` 解析到底层后端。
- `compute_topk_loss()` 支持新的 strategy：
  - FSDP 类：`fsdp`、`fsdp2`、`veomni`、`mindspeed_fsdp`
  - Megatron 类：`megatron`、`mindspeed_megatron`
- 不支持的 strategy 报错信息中同时给出原始 `config.strategy` 和 resolved strategy。
- `distillation_loss()` 在进入具体 loss 计算前，把 `dp_size`、`batch_num_tokens`、`global_batch_size`、`loss_scale_factor` 同步到 `config.global_batch_info` 和 `loss_config.global_batch_info`。

作用：

- 修复 `forward_kl_topk` 在 `model_engine=mindspeed` 下无法识别底层后端、报 `Unsupported strategy` 的问题。
- 让 top-k distillation 在 MindSpeed FSDP 和 MindSpeed Megatron 两条路径都能路由到正确 loss 实现。
- 修正 top-k log prob 归一化，避免 KL 计算使用未重新归一化的 top-k log prob。
- 让 policy-gradient distillation loss 能拿到完整 global batch 信息。

### 10. FSDP checkpoint merge key 修复

相关文件：

- `verl/model_merger/fsdp_model_merger.py`

修改点：

- 新增 `_remove_language_model_key_segments()`。
- merge 保存 HF 模型前，将 state dict key 中的 `language_model.` 片段删除。
- 如果删除后产生 key 冲突，直接抛出 `ValueError`。

作用：

- 修复某些 FSDP checkpoint 合并后 key 多出 `language_model.`，导致 HF 模型加载或保存结构不匹配的问题。
- 通过冲突检查避免静默覆盖权重。

### 11. NPU expandable segments 支持

相关文件：

- `verl/utils/device.py`

修改点：

- `set_expandable_segments()` 中新增 NPU 分支：

```python
torch.npu.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

作用：

- 让 NPU 环境也能响应 expandable segments 内存分配器设置。
- 和 CUDA 分支保持行为一致，用于改善特定训练场景下的内存碎片/分配策略。

### 12. MindSpeed 示例脚本迁移

相关文件：

- `examples/grpo_trainer/run_qwen3-32b_sglang_mindspeedllm_npu.sh`
- `examples/grpo_trainer/run_qwen3moe-30b_sglang_mindspeedllm_npu.sh`
- `tests/special_npu/run_qwen3_30b_grpo_mindspeedllm.sh`
- `tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh`

修改点：

- 原 `actor_rollout_ref.actor.mindspeed.llm_kwargs.*` 迁移到 `actor_rollout_ref.actor.mindspeed.mcore_kwargs.*`。
- 显式设置：
  - `actor_rollout_ref.actor.mindspeed.strategy=mindspeed_megatron`
  - `actor_rollout_ref.actor.mindspeed.model_name=...`
- Qwen3 / Qwen3-MoE 的 spec、seq_length、micro_batch_size、recompute、MoE 参数都迁移到 `mcore_kwargs` 下。

作用：

- 让旧 MindSpeed-LLM 示例脚本适配新的 MindSpeed engine strategy 和配置结构。
- 保持 Qwen3-8B、Qwen3-32B、Qwen3-MoE-30B 的 NPU GRPO/MindSpeed Megatron 启动能力。

### 13. 新增 Qwen3.5 MindSpeed-MM FSDP 示例脚本

相关文件：

- `examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp_mindspeedmm_npu.sh`
- `examples/grpo_trainer/run_qwen3_5_35b_vllm_fsdp_mindspeedmm_npu.sh`

修改点：

- 新增 Qwen3.5-27B + vLLM rollout + MindSpeed-MM FSDP actor 的 GRPO 脚本。
- 新增 Qwen3.5-35B-A3B + vLLM rollout + MindSpeed-MM FSDP actor 的 GRPO 脚本。
- 配置内容包括：
  - `model_engine=mindspeed`
  - `actor_rollout_ref.actor.mindspeed.strategy=mindspeed_fsdp`
  - `actor_rollout_ref.actor.mindspeed.model_name=qwen3.5-27b/qwen3.5-35b`
  - MindSpeed-MM plugin 路径。
  - DCP checkpoint load 路径。
  - FSDP apply modules / recompute modules / EP plan / hook modules。
  - visual 模块 freeze。
  - param / optimizer offload。
  - vLLM tensor parallel、chunked prefill、cudagraph capture 配置。

作用：

- 提供可直接参考的 Qwen3.5 多模态 NPU GRPO 训练启动方式。
- 覆盖 dense Qwen3.5-27B 和 MoE Qwen3.5-35B-A3B 两类 MindSpeed-MM 模型。

## 按文件归类的功能性修改清单

| 文件 | 功能性修改 |
| --- | --- |
| `verl/workers/config/engine.py` | MindSpeed config 支持 `mindspeed_megatron` / `mindspeed_fsdp`，新增 `model_name`、`mcore_kwargs`、`fsdp_kwargs`。 |
| `verl/workers/config/optimizer.py` | 新增 `MindSpeedOptimizerConfig`。 |
| `verl/trainer/config/optim/mindspeed.yaml` | 新增 MindSpeed optimizer 默认配置。 |
| `verl/trainer/config/engine/mindspeed.yaml` | 新增 MindSpeed FSDP 配置结构，默认 strategy 调整为 `mindspeed_fsdp`。 |
| `verl/trainer/config/actor/mindspeed_actor.yaml` | actor 默认使用 MindSpeed optimizer，并加载 fsdp engine 配置。 |
| `verl/trainer/config/critic/mindspeed_critic.yaml` | critic 默认使用 MindSpeed optimizer，并加载 fsdp engine 配置。 |
| `verl/trainer/config/ref/mindspeed_ref.yaml` | ref 支持新的 MindSpeed strategy/model/mcore/fsdp 配置字段。 |
| `verl/workers/engine/mindspeed/transformer_impl.py` | 新增 MindSpeed Megatron/FSDP engine，实现 MindSpeed-MM FSDP 初始化、权重导出、checkpoint、CP 输出 gather、padded CP patch。 |
| `verl/workers/engine/mindspeed/utils.py` | 新增模型支持列表、FSDP Trainer 构造、optimizer 字段转换、buffer device 迁移工具。 |
| `verl/workers/engine/__init__.py` | 导出新的 MindSpeed engine 类。 |
| `verl/workers/engine/mindspeed/__init__.py` | 导出新的 MindSpeed engine 类。 |
| `verl/workers/engine/fsdp/transformer_impl.py` | VLM 判断、packed `cu_seqlens` 传递、3D position ids rope dim 泛化、VLM fused labels 传递、offload 条件修正。 |
| `verl/utils/model.py` | 新增 `is_vision_language_model()`。 |
| `verl/utils/tensordict_utils.py` | 修复 3D nested `position_ids` 的 broken ragged metadata。 |
| `verl/utils/device.py` | NPU 支持 expandable segments allocator 设置。 |
| `verl/experimental/agent_loop/agent_loop.py` | Qwen3-VL agent loop 多模态/纯文本 position ids 兼容。 |
| `verl/trainer/distillation/fsdp/losses.py` | top-k KL log prob 重新归一化。 |
| `verl/trainer/distillation/losses.py` | MindSpeed strategy 解析、支持 `mindspeed_fsdp` / `mindspeed_megatron`、同步 global batch info。 |
| `verl/model_merger/fsdp_model_merger.py` | merge HF 模型前删除 `language_model.` key 片段。 |
| `examples/grpo_trainer/run_qwen3-32b_sglang_mindspeedllm_npu.sh` | 迁移到 `mindspeed_megatron` + `mcore_kwargs`。 |
| `examples/grpo_trainer/run_qwen3moe-30b_sglang_mindspeedllm_npu.sh` | 迁移到 `mindspeed_megatron` + `mcore_kwargs`。 |
| `tests/special_npu/run_qwen3_30b_grpo_mindspeedllm.sh` | 迁移到 `mindspeed_megatron` + `mcore_kwargs`。 |
| `tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh` | 迁移到 `mindspeed_megatron` + `mcore_kwargs`。 |
| `examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp_mindspeedmm_npu.sh` | 新增 Qwen3.5-27B MindSpeed-MM FSDP GRPO 示例。 |
| `examples/grpo_trainer/run_qwen3_5_35b_vllm_fsdp_mindspeedmm_npu.sh` | 新增 Qwen3.5-35B-A3B MindSpeed-MM FSDP GRPO 示例。 |
