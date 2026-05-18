# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang API compatibility helpers used by AIC collectors.

``legacy`` targets the sglang-v0.5.10 style used by the old runtime:
ForwardBatch is built directly from ScheduleBatch and model forwards require
an active model_executor.forward_context.

``current`` targets the newer sglang-v0.5.12 style used by AIC: ScheduleBatch
is converted to ModelWorkerBatch before constructing ForwardBatch, and the
forward-context module may not exist.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator


_BRANCH_ENV = "COLLECTOR_SGLANG_VERSION_BRANCH"


def sglang_version_branch() -> str:
    """Return ``legacy`` or ``current`` for SGLang API branching.

    ``auto`` detects the active API shape from ``ScheduleBatch``.  Explicit
    version aliases are accepted so command lines can say
    ``--sglang-version-branch v0.5.10`` or ``v0.5.12``.
    """

    raw = os.environ.get(_BRANCH_ENV, "auto").strip().lower().replace("_", "-")
    if raw in {"legacy", "old", "v0.5.10", "0.5.10", "sglang-v0.5.10"}:
        return "legacy"
    if raw in {"current", "new", "v0.5.12", "0.5.12", "sglang-v0.5.12"}:
        return "current"
    if raw not in {"", "auto"}:
        raise ValueError(
            f"Unsupported {_BRANCH_ENV}={raw!r}; expected auto, v0.5.10, or v0.5.12"
        )

    try:
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        return "current" if hasattr(ScheduleBatch, "get_model_worker_batch") else "legacy"
    except Exception:
        return "legacy"


def build_forward_batch(batch, model_runner):
    """Construct a ForwardBatch according to the selected SGLang API branch."""

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    branch = sglang_version_branch()
    if branch == "current":
        if not hasattr(batch, "get_model_worker_batch"):
            raise RuntimeError(
                "COLLECTOR_SGLANG_VERSION_BRANCH=v0.5.12 requires "
                "ScheduleBatch.get_model_worker_batch(), but the active SGLang "
                "runtime does not provide it. Use --sglang-version-branch v0.5.10 "
                "for sglang-v0.5.10."
            )
        batch = batch.get_model_worker_batch()
    return ForwardBatch.init_new(batch, model_runner)


@contextlib.contextmanager
def maybe_forward_context(model_runner) -> Iterator[None]:
    """Publish SGLang's forward context when the selected branch requires it."""

    if sglang_version_branch() != "legacy":
        yield
        return

    try:
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
    except Exception:
        yield
        return

    with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
        yield


def paged_mqa_seq_lens(seq_lens):
    """Return seq_lens shape expected by the selected paged_mqa implementation."""

    use_sglang_fallback = sglang_version_branch() == "legacy"
    try:
        import torch

        use_sglang_fallback = use_sglang_fallback or (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12
        )
    except Exception:
        pass

    if use_sglang_fallback and getattr(seq_lens, "dim", lambda: 1)() > 1:
        return seq_lens.squeeze(-1)
    return seq_lens
