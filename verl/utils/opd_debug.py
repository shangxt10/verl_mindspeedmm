# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import socket
from typing import Any

import torch


_LOG_COUNTS: dict[str, int] = {}
_FILE_ERROR_REPORTED = False


def opd_debug_enabled() -> bool:
    return os.getenv("MINDSPEED_MM_OPD_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _emit_opd_debug(message: str, rank: int) -> None:
    """Emit to stdout and, when configured, a process-local rank log."""
    print(message, flush=True)

    debug_dir = os.getenv("MINDSPEED_MM_OPD_DEBUG_DIR", "").strip()
    if not debug_dir:
        return

    global _FILE_ERROR_REPORTED
    try:
        os.makedirs(debug_dir, exist_ok=True)
        hostname = socket.gethostname().replace(os.sep, "_")
        log_path = os.path.join(debug_dir, f"opd_{hostname}_rank_{rank}_pid_{os.getpid()}.log")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message)
            if not message.endswith("\n"):
                log_file.write("\n")
    except OSError as error:
        if not _FILE_ERROR_REPORTED:
            print(f"[MindSpeed-MM][OPD_DEBUG][FILE_ERROR] {error}", flush=True)
            _FILE_ERROR_REPORTED = True


def log_opd_tensor(name: str, tensor: torch.Tensor, **metadata: Any) -> None:
    """Print tensor diagnostics without materializing the full tensor by default."""
    if not opd_debug_enabled():
        return

    detached = tensor.detach()
    flat = detached.reshape(-1)
    max_elements = int(os.getenv("MINDSPEED_MM_OPD_DEBUG_MAX_ELEMENTS", "16"))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    call_index = _LOG_COUNTS.get(name, 0)
    _LOG_COUNTS[name] = call_index + 1

    if flat.numel() == 0:
        stats = "empty=True"
        sample = []
    else:
        stats_tensor = flat if flat.is_floating_point() else flat.float()
        stats = (
            f"numel={flat.numel()}, finite={torch.isfinite(stats_tensor).all().item()}, "
            f"min={stats_tensor.min().item():.8g}, max={stats_tensor.max().item():.8g}, "
            f"sum={stats_tensor.sum().item():.8g}, mean={stats_tensor.mean().item():.8g}, "
            f"norm={stats_tensor.norm().item():.8g}"
        )
        sample = flat[:max_elements].cpu().tolist()

    metadata_text = ", ".join(f"{key}={value}" for key, value in metadata.items())
    if metadata_text:
        metadata_text = f", {metadata_text}"
    _emit_opd_debug(
        f"[MindSpeed-MM][OPD_DEBUG][{name}] rank={rank}, call={call_index}, shape={tuple(detached.shape)}, "
        f"dtype={detached.dtype}, {stats}, sample={sample}{metadata_text}",
        rank,
    )

    full_names = {item.strip() for item in os.getenv("MINDSPEED_MM_OPD_DEBUG_FULL_NAMES", "").split(",")}
    log_full_tensor = os.getenv("MINDSPEED_MM_OPD_DEBUG_FULL", "0").lower() in {"1", "true", "yes", "on"}
    if log_full_tensor or name in full_names:
        _emit_opd_debug(f"[MindSpeed-MM][OPD_DEBUG][{name}][FULL] rank={rank}\n{detached.cpu()}", rank)
