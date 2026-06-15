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


import torch
import torch.nn.functional as F

from verl.utils.opd_debug import log_opd_tensor
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    sp_size = get_ulysses_sequence_parallel_world_size()
    sp_rank = get_ulysses_sequence_parallel_rank()
    if sp_size > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]
    local_seq_len = teacher_topk_ids.shape[1]
    shard_metadata = {
        "cp_size": sp_size,
        "cp_rank": sp_rank,
        "sequence_start": sp_rank * local_seq_len,
        "sequence_end": (sp_rank + 1) * local_seq_len,
    }

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    log_opd_tensor("teacher_topk_ids_local", teacher_topk_ids, **shard_metadata)
    log_opd_tensor("student_topk_logprobs_local", student_topk_log_probs, **shard_metadata)
    log_opd_tensor("teacher_topk_logprobs_local", teacher_topk_log_probs, **shard_metadata)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    log_opd_tensor("student_topk_mass_local", student_mass, **shard_metadata)
    log_opd_tensor("teacher_topk_mass_local", teacher_mass, **shard_metadata)
    loss_config: DistillationLossConfig = config.distillation_loss
    student_topk_log_probs = F.log_softmax(student_topk_log_probs, dim=-1)
    teacher_topk_log_probs = F.log_softmax(teacher_topk_log_probs, dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    log_opd_tensor("student_topk_logprobs_normalized_local", student_topk_log_probs, **shard_metadata)
    log_opd_tensor("teacher_topk_logprobs_normalized_local", teacher_topk_log_probs, **shard_metadata)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)
    log_opd_tensor("distillation_loss_per_token_local", distillation_losses, **shard_metadata)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}
