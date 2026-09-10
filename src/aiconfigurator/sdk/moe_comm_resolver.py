# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data-driven MoE communication resolution for exact model configs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import (
    _architecture_to_model_family,
    _get_model_info,
    check_is_moe,
)
from aiconfigurator.sdk.models.blocks.moe import LARGE_EP_READY_FAMILIES, MoEBlockShape
from aiconfigurator.sdk.models.helpers import _apply_model_quant_defaults
from aiconfigurator.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

_DEEPEP_NODE1_FALLBACK_BACKENDS = frozenset(("deepep_ht", "deepep_ll", "trtllm_deepep_ht", "trtllm_deepep_ll"))
MOE_COMM_MODE_CHOICES = frozenset(("auto", "fused", "a2a"))

LargeEpCoverage = Mapping[str, Mapping[str, Set[int]]]


def a2a_covers_parallel(
    pairs: set[tuple[int, int]],
    *,
    framework: str,
    comm_backend: str,
    moe_ep_size: int,
    expected_nodes: int,
    gpus_per_node: int,
) -> bool:
    """Whether A2A data can serve a target EP/node scale.

    Prefer an exact scale. Otherwise, vLLM/TRT-LLM DeepEP HT/LL may use the
    physical full-node ``(ep=gpus_per_node, node_num=1)`` row. SGLang keeps
    its legacy normalized ``(ep=8, node_num=1)`` coordinate from PR #1314.
    The query engine marks every substitution as estimated.
    """
    if (moe_ep_size, expected_nodes) in pairs:
        return True
    if comm_backend not in _DEEPEP_NODE1_FALLBACK_BACKENDS or expected_nodes <= 1:
        return False
    if framework == "sglang":
        # Preserve PR #1314's legacy adapter coordinate. Those tables are
        # normalized to EP8/node1 independently of the queried system width.
        return (8, 1) in pairs
    return framework in {"vllm", "trtllm"} and (gpus_per_node, 1) in pairs


def moe_compute_coverage(
    database: Any,
    *,
    backend_name: str,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    quant_mode: common.MoEQuantMode,
    phase: str,
) -> set[int]:
    """Return compute coverage for the large-EP graph without comm fallback.

    Prefer the unified WideEP compute table. vLLM/TRT-LLM may additionally
    use their regular expert-kernel table; communication remains exclusively
    modeled by ``MoEAllToAll``.
    """
    compute_probe = getattr(database, "moe_expert_compute_coverage", None)
    covered = (
        set(compute_probe(hidden_size, inter_size, topk, num_experts, quant_mode, phase))
        if compute_probe is not None
        else set()
    )
    if backend_name in {"vllm", "trtllm"}:
        legacy_probe = getattr(database, "legacy_moe_compute_coverage", None)
        if legacy_probe is not None:
            covered.update(legacy_probe(hidden_size, inter_size, topk, num_experts, quant_mode))
    return covered


def select_moe_comm_backend(
    coverage: LargeEpCoverage,
    *,
    backend_name: str,
    attention_dp_size: int,
    moe_ep_size: int,
) -> dict[str, str]:
    """Select one communication backend for every covered phase at one EP."""
    if backend_name == "trtllm" and attention_dp_size <= 1:
        return {}

    resolved: dict[str, str] = {}
    for phase, per_backend in coverage.items():
        for comm_backend, eps in per_backend.items():
            if moe_ep_size in eps:
                resolved[phase] = comm_backend
                break
    return resolved


def resolve_model_config_moe_comm(
    model_config: ModelConfig,
    *,
    model_path: str,
    backend_name: str,
    database: Any,
    required_phases: tuple[str, ...],
    fmha_quant_mode_explicit: bool = False,
    kvcache_quant_mode_explicit: bool = False,
    coverage_snapshot: LargeEpCoverage | None = None,
    moe_comm_mode: str = "auto",
) -> dict[str, str] | None:
    """Resolve large-EP communication data for table-backed database modes.

    Analytical mode is intentionally table-free.  Its Rust MoE, dispatch, and
    collective operators already price the configured EP width with formulas,
    so it must keep the legacy fused ``MoEDispatch + MoE`` graph instead of
    being promoted to the table-backed ``MoeAllToAll + MoeExpertCompute``
    graph.  In particular, an analytical cross-node EP must not require a
    ``moe_a2a_perf.parquet`` file just to build the model config.

    ``moe_comm_mode`` controls the route: ``"auto"`` uses coverage and family
    capability, ``"fused"`` forces the legacy fused graph, and ``"a2a"``
    explicitly requires the generic large-EP graph. An explicit ``"a2a"``
    request does not silently fall back to fused when its tables are missing.
    """
    if not check_is_moe(model_path):
        return None
    if moe_comm_mode not in MOE_COMM_MODE_CHOICES:
        raise ValueError(
            f"moe_comm_mode must be one of {sorted(MOE_COMM_MODE_CHOICES)}, got {moe_comm_mode!r}"
        )
    # Keep the concrete config self-describing even when this resolver is
    # called by a legacy direct-estimate path instead of Task.
    model_config.moe_comm_mode = moe_comm_mode

    if moe_comm_mode == "fused":
        return None

    # ``database`` is a configured PerfDatabase view in every production call
    # site.  Keep the fallback attributes for lightweight test doubles and
    # older SDK callers that expose only ``database_mode``.
    database_mode = None
    if database is not None:
        get_mode = getattr(database, "get_default_database_mode", None)
        database_mode = get_mode() if callable(get_mode) else getattr(database, "database_mode", None)
    is_analytical = (
        database_mode == common.DatabaseMode.ANALYTICAL
        or getattr(database_mode, "name", None) == common.DatabaseMode.ANALYTICAL.name
        or (isinstance(database_mode, str) and database_mode.upper() == common.DatabaseMode.ANALYTICAL.name)
    )
    if is_analytical and moe_comm_mode != "a2a":
        return None

    info = _get_model_info(model_path)
    family = _architecture_to_model_family(info["architecture"])
    moe_tp_size = int(model_config.moe_tp_size or 1)
    moe_ep_size = int(model_config.moe_ep_size or 1)
    if moe_ep_size <= 1:
        return None
    # The generic large-EP graph is opt-in at the model-family level. The gate
    # is evaluated only for an actual EP>1 request; EP1 does not need a
    # communication graph regardless of the selected policy.
    if family not in LARGE_EP_READY_FAMILIES:
        if moe_comm_mode == "a2a":
            raise ValueError(
                f"moe_comm_mode='a2a' is not supported for model family {family!r}; "
                "use moe_comm_mode='fused' or collect and wire a model-specific large-EP graph."
            )
        return None

    gpus_per_node = 0
    if database is not None and hasattr(database, "system_spec"):
        gpus_per_node = int(database.system_spec.get("node", {}).get("num_gpus_per_node", 0) or 0)
    if not gpus_per_node:
        gpus_per_node = int(model_config.num_gpus_per_node or 0)
    if not gpus_per_node:
        raise ValueError("Cannot resolve MoE communication: num_gpus_per_node is missing from the system spec.")
    model_config.num_gpus_per_node = gpus_per_node

    cross_node = moe_ep_size > gpus_per_node
    if not cross_node and coverage_snapshot is None and moe_comm_mode != "a2a":
        # Exact CLI estimates historically keep intra-node EP on the fused
        # path. Task supplies its cached coverage snapshot and can still
        # resolve an explicitly covered intra-node tuple.
        return None
    if moe_tp_size != 1:
        if cross_node:
            raise ValueError(
                "Cross-node EP requires pure expert parallelism "
                f"(moe_tp_size=1); got moe_tp_size={moe_tp_size}, moe_ep_size={moe_ep_size}."
            )
        return None

    _apply_model_quant_defaults(
        model_config,
        info.get("raw_config", {}),
        info["architecture"],
        backend_name,
        worker_name=model_path,
    )
    shape = MoEBlockShape.from_model_info(info)
    sm_version = (
        database.system_spec.get("gpu", {}).get("sm_version")
        if database is not None and hasattr(database, "system_spec")
        else None
    )
    sm_version = int(sm_version) if sm_version is not None else None
    expected_nodes = nodes_for(moe_ep_size, gpus_per_node)

    resolved: dict[str, str]
    if coverage_snapshot is not None:
        resolved = select_moe_comm_backend(
            coverage_snapshot,
            backend_name=backend_name,
            attention_dp_size=int(model_config.attention_dp_size or 1),
            moe_ep_size=moe_ep_size,
        )
    else:
        coverage: dict[str, dict[str, set[int]]] = {}
        resolved = {}
    if coverage_snapshot is None and family in LARGE_EP_READY_FAMILIES and database is not None:
        a2a_probe = getattr(database, "moe_a2a_coverage", None)
        for phase in dict.fromkeys(required_phases):
            a2a = (
                a2a_probe(
                    shape.hidden_size,
                    shape.topk,
                    shape.num_experts,
                    model_config.moe_quant_mode,
                    phase,
                )
                if a2a_probe is not None
                else {}
            )
            compute_eps = moe_compute_coverage(
                database,
                backend_name=backend_name,
                hidden_size=shape.hidden_size,
                inter_size=shape.moe_inter_size,
                topk=shape.topk,
                num_experts=shape.num_experts,
                quant_mode=model_config.moe_quant_mode,
                phase=phase,
            )
            for comm_backend, backend_spec in MOE_A2A_BACKENDS.items():
                if backend_name not in backend_spec.frameworks or phase not in backend_spec.inference_phases:
                    continue
                if backend_name == "trtllm" and int(model_config.attention_dp_size or 1) <= 1:
                    # Keep parity with Task._resolve_moe_comm_backend: TRT-LLM
                    # large EP requires attention DP. Do not return early;
                    # cross-node EP must reach the missing-coverage error.
                    continue
                if (
                    a2a_covers_parallel(
                        a2a.get(comm_backend, set()),
                        framework=backend_name,
                        comm_backend=comm_backend,
                        moe_ep_size=moe_ep_size,
                        expected_nodes=expected_nodes,
                        gpus_per_node=gpus_per_node,
                    )
                    and moe_ep_size in compute_eps
                    and backend_spec.feasible(
                        topk=shape.topk,
                        num_experts=shape.num_experts,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                        sm_version=sm_version,
                    )
                ):
                    coverage.setdefault(phase, {}).setdefault(comm_backend, set()).add(moe_ep_size)
                    break
        resolved = select_moe_comm_backend(
            coverage,
            backend_name=backend_name,
            attention_dp_size=int(model_config.attention_dp_size or 1),
            moe_ep_size=moe_ep_size,
        )

    missing = set(required_phases) - set(resolved)
    if missing:
        if cross_node or moe_comm_mode == "a2a":
            system_name = getattr(database, "system", "") if database is not None else ""
            version = getattr(database, "version", "") if database is not None else ""
            reason = (
                "Cross-node EP requires DeepEP A2A data"
                if cross_node
                else "moe_comm_mode='a2a' requires compatible A2A data"
            )
            raise PerfDataNotAvailableError(
                f"{reason}, but no compatible exact or supported node-1 coverage was found for "
                f"model={model_path!r}, system={system_name!r}, backend={backend_name!r}, version={version!r}, "
                f"moe_ep={moe_ep_size}, gpus_per_node={gpus_per_node}, "
                f"phase(s)={','.join(sorted(missing))}."
            )
        return None

    model_config.moe_comm_backend = resolved
    if database is not None:
        model_config.system = getattr(database, "system", None)
    if backend_name == "sglang" and info["architecture"] == "DeepseekV3ForCausalLM":
        # The SGLang WideEP MLA collectors label their rows fp8_block/fp8.
        # Preserve explicit user modes, but restore the PR #1314 defaults for
        # inferred DeepSeek-V3/R1 configs once this tuple selects DeepEP.
        if not fmha_quant_mode_explicit:
            model_config.fmha_quant_mode = common.FMHAQuantMode.fp8_block
        if not kvcache_quant_mode_explicit:
            model_config.kvcache_quant_mode = common.KVCacheQuantMode.fp8
    return resolved
