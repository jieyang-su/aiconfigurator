# SPDX-License-Identifier: Apache-2.0

"""Versioned rank-local MoE workload replay for SGLang DeepEP.

The SGLang ``per_token`` expert-distribution recorder writes one file per EP
rank.  Each file contains the router's physical top-k choices and, for normal
DeepEP dispatch, the post-dispatch rank-local ``recv_topk_ids``.  This module
merges those files into one portable bundle and provides strict workload
selection for the standalone WideEP MoE compute collector.

The bundle intentionally stores workload tensors, not hidden states or model
weights.  Benchmark inputs can therefore be generated deterministically while
preserving the real token-to-expert incidence and rank boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)
MANIFEST_FILENAME = "manifest.csv"


@dataclass(frozen=True)
class RankLocalWorkload:
    rank: int
    local_topk_ids: torch.Tensor
    num_recv_tokens_per_expert: tuple[int, ...]
    masked_m: torch.Tensor
    expected_m: int
    dispatch_layout: dict[str, Any] | None = None

    @property
    def num_recv_tokens(self) -> int:
        return int(self.local_topk_ids.shape[0])


def _phase_from_forward_mode(value: Any) -> str | None:
    mode = str(value).lower()
    # Some SGLang recorder versions serialize ForwardMode(IntEnum) through
    # str(), yielding the numeric enum value rather than its symbolic name.
    if mode == "1":
        return "context"
    if mode == "2":
        return "generation"
    if "decode" in mode:
        return "generation"
    if any(name in mode for name in ("extend", "prefill", "context")):
        return "context"
    return None


def _load_rank_records(recorder_dir: Path) -> tuple[dict[int, list[dict]], torch.Tensor]:
    records_by_rank: dict[int, list[dict]] = {}
    physical_to_logical_map = None
    for path in sorted(recorder_dir.glob("expert_distribution_recorder_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "records" not in payload:
            continue
        current_map = payload.get("last_physical_to_logical_map")
        if current_map is None:
            raise ValueError(f"Missing last_physical_to_logical_map in {path}")
        current_map = current_map.to(torch.int64).cpu()
        if physical_to_logical_map is None:
            physical_to_logical_map = current_map
        elif not torch.equal(physical_to_logical_map, current_map):
            raise ValueError("physical_to_logical_map changed across recorder rank files")
        for record in payload["records"]:
            if not isinstance(record, dict):
                continue
            rank = int(record["rank"])
            records_by_rank.setdefault(rank, []).append(record)
    if not records_by_rank:
        raise ValueError(
            f"No per_token recorder files found in {recorder_dir}; "
            "set expert_distribution_recorder_mode=per_token"
        )
    assert physical_to_logical_map is not None
    return records_by_rank, physical_to_logical_map


def _normal_workload(
    record: dict,
    *,
    rank: int,
    layer_index: int,
    num_local_physical_experts: int,
    allow_router_fallback: bool = False,
    allow_missing_dispatch_fallback: bool = False,
    model_layer_id: int | None = None,
) -> RankLocalWorkload:
    accepted_layer_ids = {layer_index}
    if model_layer_id is not None:
        accepted_layer_ids.add(model_layer_id)
    matches = [
        item
        for item in record.get("misc_objects", ())
        if isinstance(item, dict)
        and int(item.get("layer_id", -1)) in accepted_layer_ids
    ]
    if not matches:
        # Some SGLang versions number DeepEP dispatch records by compressed
        # MoE-layer ordinal (0..N-1), while the recorder's router tensors and
        # physical map retain original model-layer IDs. A one-MoE-layer
        # calibration model is unambiguous, so accept its sole dispatch item.
        dispatch_items = [
            item
            for item in record.get("misc_objects", ())
            if isinstance(item, dict) and "local_physical_count_of_layer" in item
        ]
        if len(dispatch_items) == 1:
            matches = dispatch_items
    if not matches and allow_missing_dispatch_fallback:
        return _router_rank_local_workload(
            record,
            rank=rank,
            layer_index=layer_index,
            num_local_physical_experts=num_local_physical_experts,
            model_layer_id=model_layer_id,
            fallback_reason="missing_normal_dispatch_misc",
        )
    if not matches and allow_router_fallback:
        topk_ids = record["topk_ids_of_layer"][layer_index].to(torch.int32).cpu()
        valid = topk_ids[topk_ids >= 0]
        if valid.numel() and int(valid.max().item()) >= num_local_physical_experts:
            raise ValueError("EP=1 router record contains a non-local expert ID")
        counts = torch.bincount(
            valid.to(torch.int64),
            minlength=num_local_physical_experts,
        ).to(torch.int32)
        count_tuple = tuple(int(value) for value in counts.tolist())
        return RankLocalWorkload(
            rank=rank,
            local_topk_ids=topk_ids.contiguous(),
            num_recv_tokens_per_expert=count_tuple,
            masked_m=counts,
            expected_m=int(
                math.ceil(int(counts.sum().item()) / max(1, len(counts)))
            ),
        )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one normal-dispatch record for rank={rank}, layer={layer_index}; "
            f"got {len(matches)}"
        )
    item = matches[0]
    recv_topk_ids = item.get("recv_topk_ids")
    if recv_topk_ids is None:
        raise ValueError(f"Missing recv_topk_ids for rank={rank}, layer={layer_index}")
    recv_topk_ids = recv_topk_ids.to(torch.int32).cpu().contiguous()
    valid = recv_topk_ids[recv_topk_ids >= 0]
    if valid.numel() and int(valid.max().item()) >= num_local_physical_experts:
        rank_offset = rank * num_local_physical_experts
        recv_topk_ids = torch.where(
            recv_topk_ids >= 0,
            recv_topk_ids - rank_offset,
            recv_topk_ids,
        )
        valid = recv_topk_ids[recv_topk_ids >= 0]
    if valid.numel() and (
        int(valid.min().item()) < 0
        or int(valid.max().item()) >= num_local_physical_experts
    ):
        return _router_rank_local_workload(
            record,
            rank=rank,
            layer_index=layer_index,
            num_local_physical_experts=num_local_physical_experts,
            model_layer_id=model_layer_id,
        )
    dispatch_counts = tuple(
        int(value) for value in item["local_physical_count_of_layer"]
    )
    if len(dispatch_counts) != num_local_physical_experts:
        raise ValueError(
            f"Rank {rank} has {len(dispatch_counts)} local counts, expected {num_local_physical_experts}"
        )
    # ``local_physical_count_of_layer`` is the DeepEP normal dispatch capacity
    # in some SGLang builds, padded to the kernel block size.  For standalone
    # MoE replay we need the real rank-local assignment count that matches
    # ``recv_topk_ids``; otherwise sparse prefill cases can feed impossible
    # per-expert counts such as 256 assignments for only 192 received tokens.
    recv_topk_ids = _deduplicate_local_topk_ids(recv_topk_ids)
    counts_tensor = torch.bincount(
        recv_topk_ids[recv_topk_ids >= 0].to(torch.int64),
        minlength=num_local_physical_experts,
    ).to(torch.int32)
    counts = tuple(int(value) for value in counts_tensor.tolist())
    masked_m = torch.tensor(counts, dtype=torch.int32)
    layout_items = [
        value
        for value in record.get("misc_objects", ())
        if isinstance(value, dict)
        and value.get("aic_record_type") == "dispatch_layout"
        and value.get("phase") == "context"
        and int(value.get("layer_id", -1)) in accepted_layer_ids
    ]
    return RankLocalWorkload(
        rank=rank,
        local_topk_ids=recv_topk_ids,
        num_recv_tokens_per_expert=counts,
        masked_m=masked_m,
        expected_m=int(math.ceil(sum(counts) / max(1, len(counts)))),
        dispatch_layout=layout_items[-1] if layout_items else None,
    )


def _router_layer_topk_ids(
    record: dict,
    *,
    layer_index: int,
    model_layer_id: int | None = None,
) -> torch.Tensor:
    topk_ids_by_layer = record["topk_ids_of_layer"]
    for candidate in (layer_index, model_layer_id):
        if candidate is None:
            continue
        try:
            return topk_ids_by_layer[int(candidate)].to(torch.int32).cpu()
        except (IndexError, KeyError, TypeError):
            continue
    if len(topk_ids_by_layer) == 1:
        return topk_ids_by_layer[0].to(torch.int32).cpu()
    raise ValueError(
        f"No router topk_ids for layer={layer_index}, model_layer_id={model_layer_id}"
    )


def _deduplicate_local_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    """Keep at most one slot per local expert within each token row."""

    topk_ids = topk_ids.to(torch.int32).cpu().contiguous()
    if topk_ids.numel() == 0:
        return topk_ids
    rows = topk_ids.clone()
    for row_index in range(int(rows.shape[0])):
        seen = set()
        for col_index in range(int(rows.shape[1])):
            value = int(rows[row_index, col_index].item())
            if value < 0:
                continue
            if value in seen:
                rows[row_index, col_index] = -1
            else:
                seen.add(value)
    return rows.contiguous()


def _router_rank_local_workload(
    record: dict,
    *,
    rank: int,
    layer_index: int,
    num_local_physical_experts: int,
    model_layer_id: int | None = None,
    fallback_reason: str | None = None,
) -> RankLocalWorkload:
    topk_ids = _router_layer_topk_ids(
        record,
        layer_index=layer_index,
        model_layer_id=model_layer_id,
    )
    rank_begin = rank * num_local_physical_experts
    rank_end = rank_begin + num_local_physical_experts
    local_mask = (topk_ids >= rank_begin) & (topk_ids < rank_end)
    local_ids = torch.where(local_mask, topk_ids - rank_begin, -1).to(torch.int32)
    token_mask = local_mask.any(dim=1)
    local_topk_ids = local_ids[token_mask].contiguous()
    counts = torch.bincount(
        local_topk_ids[local_topk_ids >= 0].to(torch.int64),
        minlength=num_local_physical_experts,
    ).to(torch.int32)
    count_tuple = tuple(int(value) for value in counts.tolist())
    return RankLocalWorkload(
        rank=rank,
        local_topk_ids=local_topk_ids,
        num_recv_tokens_per_expert=count_tuple,
        masked_m=counts,
        expected_m=int(math.ceil(int(counts.sum().item()) / max(1, len(counts)))),
        dispatch_layout=(
            {
                "aic_record_type": "router_rank_local_fallback",
                "reason": fallback_reason,
            }
            if fallback_reason
            else None
        ),
    )


def _low_latency_workload(
    source_records: Iterable[dict],
    *,
    rank: int,
    layer_index: int,
    num_local_physical_experts: int,
) -> RankLocalWorkload:
    source_records = list(source_records)
    # Each EP rank records routing for the decode tokens that originate on
    # that source rank. The target rank's compute workload is the all-to-all
    # union of assignments from every source rank, not just its same-rank
    # record.
    topk_ids = torch.cat(
        [
            record["topk_ids_of_layer"][layer_index].to(torch.int64).cpu()
            for record in source_records
        ],
        dim=0,
    )
    valid_global_ids = topk_ids[topk_ids >= 0]
    rank_begin = rank * num_local_physical_experts
    rank_end = rank_begin + num_local_physical_experts
    local_mask = (topk_ids >= rank_begin) & (topk_ids < rank_end)
    local_ids = torch.where(local_mask, topk_ids - rank_begin, -1).to(torch.int32)
    counts = torch.bincount(
        local_ids[local_ids >= 0].to(torch.int64),
        minlength=num_local_physical_experts,
    ).to(torch.int32)
    # The per-token record is global physical routing on every rank.  Keep only
    # tokens that actually select at least one expert owned by this rank.
    token_mask = local_mask.any(dim=1)
    local_topk_ids = local_ids[token_mask].contiguous()
    if valid_global_ids.numel() and int(valid_global_ids.max().item()) >= (
        num_local_physical_experts * max(1, rank + 1)
    ):
        # IDs belonging to later ranks are expected; this branch only documents
        # that the input is global rather than accidentally rank-local.
        pass
    count_tuple = tuple(int(value) for value in counts.tolist())
    active_experts = int((counts > 0).sum().item())
    target_record = source_records[rank]
    layout_items = [
        value
        for value in target_record.get("misc_objects", ())
        if isinstance(value, dict)
        and value.get("aic_record_type") == "dispatch_layout"
        and value.get("phase") == "generation"
        and int(value.get("layer_id", -1)) == layer_index
    ]
    if not layout_items:
        all_layout_items = [
            value
            for value in target_record.get("misc_objects", ())
            if isinstance(value, dict)
            and value.get("aic_record_type") == "dispatch_layout"
            and value.get("phase") == "generation"
        ]
        if len(all_layout_items) == 1:
            layout_items = all_layout_items
    return RankLocalWorkload(
        rank=rank,
        local_topk_ids=local_topk_ids,
        num_recv_tokens_per_expert=count_tuple,
        masked_m=counts,
        # DeepGEMM's low-latency kernel selector estimates M over experts
        # that actually receive tokens. Dividing by all local experts
        # underestimates expected_m for sparse decode batches and can select a
        # much faster, non-representative kernel template.
        expected_m=int(
            math.ceil(int(counts.sum().item()) / max(1, active_experts))
        ),
        dispatch_layout=layout_items[-1] if layout_items else None,
    )


def _serialize_workload(workload: RankLocalWorkload) -> dict[str, Any]:
    return {
        "rank": workload.rank,
        "local_topk_ids": workload.local_topk_ids,
        "num_recv_tokens_per_expert": list(workload.num_recv_tokens_per_expert),
        "masked_m": workload.masked_m,
        "expected_m": workload.expected_m,
        "dispatch_layout": workload.dispatch_layout,
    }


def _deserialize_workload(value: dict[str, Any]) -> RankLocalWorkload:
    local_topk_ids = _deduplicate_local_topk_ids(value["local_topk_ids"])
    stored_counts = tuple(int(item) for item in value["num_recv_tokens_per_expert"])
    if stored_counts:
        counts_tensor = torch.bincount(
            local_topk_ids[local_topk_ids >= 0].to(torch.int64),
            minlength=len(stored_counts),
        ).to(torch.int32)
        counts = tuple(int(item) for item in counts_tensor.tolist())
    else:
        counts = stored_counts
    return RankLocalWorkload(
        rank=int(value["rank"]),
        local_topk_ids=local_topk_ids,
        num_recv_tokens_per_expert=counts,
        masked_m=torch.tensor(counts, dtype=torch.int32),
        expected_m=int(value["expected_m"]),
        dispatch_layout=value.get("dispatch_layout"),
    )


def _validate_sample(
    workloads: Iterable[RankLocalWorkload],
    *,
    ep_size: int,
    num_local_physical_experts: int,
) -> tuple[RankLocalWorkload, ...]:
    workloads = tuple(sorted(workloads, key=lambda item: item.rank))
    if [item.rank for item in workloads] != list(range(ep_size)):
        raise ValueError(
            f"Incomplete EP sample: ranks={[item.rank for item in workloads]}, ep_size={ep_size}"
        )
    for workload in workloads:
        if len(workload.num_recv_tokens_per_expert) != num_local_physical_experts:
            raise ValueError("Invalid local expert count width")
        if tuple(int(v) for v in workload.masked_m.tolist()) != (
            workload.num_recv_tokens_per_expert
        ):
            raise ValueError("masked_m and num_recv_tokens_per_expert disagree")
        valid = workload.local_topk_ids[workload.local_topk_ids >= 0]
        if valid.numel() and int(valid.max().item()) >= num_local_physical_experts:
            raise ValueError("local_topk_ids contains a non-local expert ID")
    return workloads


def _count_summary(
    counts: Iterable[int | float],
    *,
    num_experts: int,
) -> dict[str, Any]:
    values = [float(value) for value in counts]
    if len(values) < num_experts:
        values.extend([0.0] * (num_experts - len(values)))
    elif len(values) > num_experts:
        values = values[:num_experts]
    total = float(sum(values))
    active = int(sum(value > 0 for value in values))
    max_assignments = float(max(values, default=0.0))
    mean = total / max(1, len(values))
    nonzero_mean = total / active if active else 0.0
    return {
        "num_experts": len(values),
        "total_assignments": total,
        "active_experts": active,
        "max_assignments": max_assignments,
        "mean_assignments": mean,
        "max_over_mean": max_assignments / mean if mean else "",
        "nonzero_mean_assignments": nonzero_mean if active else "",
        "max_over_nonzero_mean": (
            max_assignments / nonzero_mean if nonzero_mean else ""
        ),
        "expert_assignments_json": json.dumps(
            [int(value) if float(value).is_integer() else value for value in values],
            separators=(",", ":"),
        ),
    }


def _physical_counts_from_topk(
    record: dict,
    *,
    layer_index: int,
    model_layer_id: int,
    num_physical_experts: int,
) -> torch.Tensor:
    topk_ids = _router_layer_topk_ids(
        record,
        layer_index=layer_index,
        model_layer_id=model_layer_id,
    )
    valid = topk_ids[topk_ids >= 0].to(torch.int64).cpu()
    if valid.numel() and int(valid.max().item()) >= num_physical_experts:
        raise ValueError(
            "router topk_ids contain physical expert id outside "
            f"num_physical_experts={num_physical_experts}"
        )
    return torch.bincount(valid, minlength=num_physical_experts).to(torch.int64)


def _logical_counts_from_physical(
    physical_counts: torch.Tensor,
    *,
    physical_to_logical_map: torch.Tensor,
    layer_index: int,
    model_layer_id: int,
    num_logical_experts: int,
) -> torch.Tensor:
    for candidate in (layer_index, model_layer_id):
        if candidate is None:
            continue
        try:
            mapping = physical_to_logical_map[int(candidate)].to(torch.int64).cpu()
            break
        except (IndexError, TypeError):
            mapping = None
    else:
        mapping = None
    if mapping is None and int(physical_to_logical_map.shape[0]) == 1:
        mapping = physical_to_logical_map[0].to(torch.int64).cpu()
    if mapping is None:
        raise ValueError(
            f"No physical_to_logical_map row for layer={layer_index}, "
            f"model_layer_id={model_layer_id}"
        )
    valid = (mapping >= 0) & (mapping < num_logical_experts)
    logical_counts = torch.zeros(num_logical_experts, dtype=torch.int64)
    logical_counts.scatter_add_(
        0,
        mapping[valid],
        physical_counts.to(torch.int64)[valid],
    )
    return logical_counts


def _write_diagnostics_rows(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    existing_rows = []
    existing_fields: list[str] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or ())
            existing_rows = list(reader)
    fields = list(existing_fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)


def _record_misc_summary(record: dict) -> dict[str, Any]:
    misc_objects = [
        item for item in record.get("misc_objects", ()) if isinstance(item, dict)
    ]
    local_dispatch_items = [
        item for item in misc_objects if "local_physical_count_of_layer" in item
    ]
    layout_items = [
        item for item in misc_objects if item.get("aic_record_type") == "dispatch_layout"
    ]
    layer_ids = sorted(
        {
            int(item.get("layer_id"))
            for item in misc_objects
            if item.get("layer_id") is not None
        }
    )
    return {
        "misc_object_count": len(misc_objects),
        "misc_local_dispatch_count": len(local_dispatch_items),
        "misc_layout_count": len(layout_items),
        "misc_layer_ids_json": json.dumps(layer_ids, separators=(",", ":")),
    }


def materialize_replay_bundle(
    *,
    recorder_dir: str | Path,
    output_dir: str | Path,
    model: str,
    requested_ep_size: int,
    runtime_ep_size: int,
    enable_eplb: bool,
    topk: int,
    num_logical_experts: int,
    first_moe_layer_id: int,
    context_table_num_tokens: int,
    generation_table_num_tokens: int,
    workload_source: str = "runtime",
    diagnostics_path: str | Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Merge SGLang per-rank recorder files into one replay bundle.

    Returns the bundle path and perf-style summary rows.  A bundle can contain
    both context and generation samples from one Engine.generate call.
    """

    recorder_dir = Path(recorder_dir)
    output_dir = Path(output_dir)
    workload_source = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(workload_source).strip().lower()
    ) or "runtime"
    records_by_rank, physical_to_logical_map = _load_rank_records(recorder_dir)
    ranks = sorted(records_by_rank)
    if ranks != list(range(runtime_ep_size)):
        raise ValueError(
            f"Recorder ranks {ranks} do not match runtime_ep_size={runtime_ep_size}"
        )
    num_layers, num_physical_experts = physical_to_logical_map.shape
    if num_physical_experts % runtime_ep_size:
        raise ValueError(
            f"Physical experts {num_physical_experts} not divisible by EP={runtime_ep_size}"
        )
    num_local_physical_experts = num_physical_experts // runtime_ep_size

    by_rank_and_pass = {
        rank: {int(record["forward_pass_id"]): record for record in records}
        for rank, records in records_by_rank.items()
    }
    pass_ids = sorted(
        set.intersection(
            *(set(records) for records in by_rank_and_pass.values())
        )
    )
    samples = []
    summary_rows = []
    diagnostics_rows = []
    if num_layers > first_moe_layer_id:
        # Recorder map is indexed by the original model layer.
        replay_layers = [
            (layer_id, layer_id)
            for layer_id in range(first_moe_layer_id, num_layers)
        ]
    else:
        # Recorder map contains only MoE layers, compressed to start at zero.
        replay_layers = [
            (layer_index, first_moe_layer_id + layer_index)
            for layer_index in range(num_layers)
        ]
    try:
        for pass_id in pass_ids:
            first_record = by_rank_and_pass[0][pass_id]
            phase = _phase_from_forward_mode(first_record.get("forward_mode"))
            if phase is None:
                continue
            if phase == "context" and context_table_num_tokens <= 0:
                continue
            if phase == "generation" and generation_table_num_tokens <= 0:
                continue
            table_num_tokens = (
                context_table_num_tokens
                if phase == "context"
                else generation_table_num_tokens
            )
            for layer_index, output_layer_id in replay_layers:
                workloads = []
                for rank in ranks:
                    record = by_rank_and_pass[rank][pass_id]
                    if _phase_from_forward_mode(record.get("forward_mode")) != phase:
                        raise ValueError(
                            f"Forward mode mismatch for pass={pass_id}, rank={rank}"
                        )
                    misc_summary = _record_misc_summary(record)
                    router_physical_counts = _physical_counts_from_topk(
                        record,
                        layer_index=layer_index,
                        model_layer_id=output_layer_id,
                        num_physical_experts=num_physical_experts,
                    )
                    router_logical_counts = _logical_counts_from_physical(
                        router_physical_counts,
                        physical_to_logical_map=physical_to_logical_map,
                        layer_index=layer_index,
                        model_layer_id=output_layer_id,
                        num_logical_experts=num_logical_experts,
                    )
                    for count_kind, semantics, counts in (
                        (
                            "router_global_physical_from_topk",
                            "global_physical_per_recorder_rank",
                            router_physical_counts.tolist(),
                        ),
                        (
                            "router_logical_from_topk",
                            "logical_per_recorder_rank",
                            router_logical_counts.tolist(),
                        ),
                    ):
                        diagnostics_rows.append(
                            {
                                "model": model,
                                "workload_source": workload_source,
                                "phase": phase,
                                "forward_pass_id": pass_id,
                                "rank": rank,
                                "layer_index": layer_index,
                                "layer_id": output_layer_id,
                                "count_kind": count_kind,
                                "semantics": semantics,
                                "requested_ep_size": requested_ep_size,
                                "runtime_ep_size": runtime_ep_size,
                                "enable_eplb": bool(enable_eplb),
                                "topk": topk,
                                "table_num_tokens": table_num_tokens,
                                **misc_summary,
                                **_count_summary(
                                    counts,
                                    num_experts=num_logical_experts,
                                ),
                            }
                        )
                for rank in ranks:
                    record = by_rank_and_pass[rank][pass_id]
                    misc_summary = _record_misc_summary(record)
                    if phase == "context":
                        workload = _normal_workload(
                            record,
                            rank=rank,
                            layer_index=layer_index,
                            num_local_physical_experts=num_local_physical_experts,
                            allow_router_fallback=runtime_ep_size == 1,
                            allow_missing_dispatch_fallback=True,
                            model_layer_id=output_layer_id,
                        )
                    else:
                        workload = _low_latency_workload(
                            [
                                by_rank_and_pass[source_rank][pass_id]
                                for source_rank in ranks
                            ],
                            rank=rank,
                            layer_index=layer_index,
                            num_local_physical_experts=num_local_physical_experts,
                        )
                    workloads.append(workload)
                    diagnostics_rows.append(
                        {
                            "model": model,
                            "workload_source": workload_source,
                            "phase": phase,
                            "forward_pass_id": pass_id,
                            "rank": rank,
                            "layer_index": layer_index,
                            "layer_id": output_layer_id,
                            "count_kind": (
                                "router_rank_local_fallback"
                                if (
                                    workload.dispatch_layout
                                    and workload.dispatch_layout.get("aic_record_type")
                                    == "router_rank_local_fallback"
                                )
                                else "post_dispatch_rank_local"
                            ),
                            "semantics": (
                                "physical_rank_local_from_router_fallback"
                                if (
                                    workload.dispatch_layout
                                    and workload.dispatch_layout.get("aic_record_type")
                                    == "router_rank_local_fallback"
                                )
                                else "physical_rank_local_post_dispatch"
                            ),
                            "requested_ep_size": requested_ep_size,
                            "runtime_ep_size": runtime_ep_size,
                            "enable_eplb": bool(enable_eplb),
                            "topk": topk,
                            "table_num_tokens": table_num_tokens,
                            **misc_summary,
                            **_count_summary(
                                workload.num_recv_tokens_per_expert,
                                num_experts=num_local_physical_experts,
                            ),
                        }
                    )
                workloads = _validate_sample(
                    workloads,
                    ep_size=runtime_ep_size,
                    num_local_physical_experts=num_local_physical_experts,
                )
                sample_id = f"{phase}-pass{pass_id}-layer{output_layer_id}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "forward_pass_id": pass_id,
                        "phase": phase,
                        "table_num_tokens": table_num_tokens,
                        "layer_id": output_layer_id,
                        "ranks": [_serialize_workload(item) for item in workloads],
                    }
                )
                physical_counts = [
                    count
                    for workload in workloads
                    for count in workload.num_recv_tokens_per_expert
                ]
                total_assignments = sum(physical_counts)
                active_experts = sum(count > 0 for count in physical_counts)
                mean_assignments = total_assignments / max(1, len(physical_counts))
                max_assignments = max(physical_counts, default=0)
                summary_rows.append(
                    {
                        "phase": phase,
                        "num_tokens": table_num_tokens,
                        "layer_id": output_layer_id,
                        "total_assignments": total_assignments,
                        "active_experts": active_experts,
                        "max_assignments": max_assignments,
                        "mean_assignments": mean_assignments,
                        "expert_assignments_json": json.dumps(
                            physical_counts,
                            separators=(",", ":"),
                        ),
                        "sample_id": sample_id,
                    }
                )
    finally:
        if diagnostics_path is not None and diagnostics_rows:
            diagnostics_path = Path(diagnostics_path)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            _write_diagnostics_rows(path=diagnostics_path, rows=diagnostics_rows)

    if not samples:
        observed_modes = sorted(
            {
                str(record.get("forward_mode"))
                for records in records_by_rank.values()
                for record in records
            }
        )
        raise ValueError(
            f"No context or generation samples found in {recorder_dir}; "
            f"observed forward modes={observed_modes}, common pass ids={pass_ids}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"dsv3_{workload_source}_ep{requested_ep_size}_runtimeep{runtime_ep_size}_"
        f"eplb{int(enable_eplb)}_ctx{context_table_num_tokens}_"
        f"gen{generation_table_num_tokens}.pt"
    )
    bundle_path = output_dir / name
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "workload_source": workload_source,
        "requested_ep_size": requested_ep_size,
        "runtime_ep_size": runtime_ep_size,
        "enable_eplb": bool(enable_eplb),
        "topk": topk,
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": num_physical_experts,
        "num_local_physical_experts": num_local_physical_experts,
        "first_moe_layer_id": first_moe_layer_id,
        "physical_to_logical_map": physical_to_logical_map,
        "samples": samples,
    }
    torch.save(payload, bundle_path)
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_rows = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
    relative_bundle = bundle_path.name
    for row in summary_rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "distribution": "recorded",
                "workload_source": workload_source,
                "enable_eplb": bool(enable_eplb),
                "requested_ep_size": requested_ep_size,
                "runtime_ep_size": runtime_ep_size,
                "topk": topk,
                "num_experts": num_logical_experts,
                "replay_file": relative_bundle,
                "replay_sha256": digest,
            }
        )
    manifest_rows = [
        row for row in manifest_rows if row.get("replay_file") != relative_bundle
    ] + summary_rows
    manifest_fields = []
    for row in manifest_rows:
        for field in row:
            if field not in manifest_fields:
                manifest_fields.append(field)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_diagnostics_rows(path=diagnostics_path, rows=diagnostics_rows)
    return bundle_path, summary_rows


def _repeat_topk_rows(topk_ids: torch.Tensor, target_rows: int) -> torch.Tensor:
    topk_ids = topk_ids.to(torch.int32).cpu().contiguous()
    target_rows = max(0, int(target_rows))
    if target_rows == int(topk_ids.shape[0]):
        return topk_ids.clone()
    if target_rows == 0:
        return topk_ids[:0].clone()
    if int(topk_ids.shape[0]) == 0:
        return topk_ids.clone()
    repeat_factor = math.ceil(target_rows / int(topk_ids.shape[0]))
    return topk_ids.repeat((repeat_factor, 1))[:target_rows].contiguous()


def _scale_workload(
    value: dict[str, Any],
    *,
    ratio: float,
    num_local_physical_experts: int,
) -> dict[str, Any]:
    source_topk_ids = value["local_topk_ids"].to(torch.int32).cpu()
    target_rows = int(round(int(source_topk_ids.shape[0]) * ratio))
    local_topk_ids = _repeat_topk_rows(source_topk_ids, target_rows)
    counts = torch.bincount(
        local_topk_ids[local_topk_ids >= 0].to(torch.int64),
        minlength=num_local_physical_experts,
    ).to(torch.int32)
    return _serialize_workload(
        RankLocalWorkload(
            rank=int(value["rank"]),
            local_topk_ids=local_topk_ids,
            num_recv_tokens_per_expert=tuple(int(item) for item in counts.tolist()),
            masked_m=counts,
            expected_m=int(
                math.ceil(int(counts.sum().item()) / max(1, len(counts)))
            ),
            dispatch_layout=value.get("dispatch_layout"),
        )
    )


def materialize_extrapolated_replay_bundle(
    *,
    replay_dir: str | Path,
    model: str,
    requested_ep_size: int,
    runtime_ep_size: int,
    enable_eplb: bool,
    topk: int,
    num_logical_experts: int,
    context_table_num_tokens: int,
    workload_source: str = "runtime",
) -> tuple[Path, list[dict[str, Any]], int]:
    """Create a same-source replay bundle by scaling a nearby context sample.

    This is a collector recovery path for dummy SGLang recorder runs that fail
    after smaller token points have already produced rank-local replay bundles.
    It preserves the observed rank/expert incidence pattern while extending the
    token grid needed by standalone MoE kernel profiling.
    """

    replay_dir = Path(replay_dir)
    manifest_path = replay_dir / MANIFEST_FILENAME
    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    candidates = [
        row
        for row in manifest_rows
        if row.get("phase") == "context"
        and int(row.get("requested_ep_size", -1)) == int(requested_ep_size)
        and int(row.get("runtime_ep_size", -1)) == int(runtime_ep_size)
        and int(row.get("num_experts", -1)) == int(num_logical_experts)
        and int(row.get("topk", -1)) == int(topk)
        and (str(row.get("enable_eplb", "")).lower() in ("1", "true"))
        == bool(enable_eplb)
        and row.get("workload_source", "runtime") == workload_source
        and int(row.get("num_tokens", -1)) != int(context_table_num_tokens)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No donor replay available for extrapolation: "
            f"tokens={context_table_num_tokens}, ep={requested_ep_size}, "
            f"runtime_ep={runtime_ep_size}, experts={num_logical_experts}, "
            f"eplb={enable_eplb}, source={workload_source}"
        )

    def donor_score(row: dict[str, Any]) -> tuple[int, int]:
        tokens = int(row["num_tokens"])
        not_oversized = int(tokens <= int(context_table_num_tokens))
        distance = -abs(tokens - int(context_table_num_tokens))
        return (not_oversized, distance)

    donor_row = max(candidates, key=donor_score)
    donor_tokens = int(donor_row["num_tokens"])
    donor_bundle = replay_dir / donor_row["replay_file"]
    payload = torch.load(donor_bundle, map_location="cpu", weights_only=True)
    if int(payload["schema_version"]) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported replay schema {payload['schema_version']} in {donor_bundle}"
        )
    num_local_physical_experts = int(payload["num_local_physical_experts"])
    ratio = int(context_table_num_tokens) / max(1, donor_tokens)

    samples = []
    summary_rows = []
    for sample in payload["samples"]:
        if sample["phase"] != "context":
            continue
        ranks = [
            _scale_workload(
                rank_payload,
                ratio=ratio,
                num_local_physical_experts=num_local_physical_experts,
            )
            for rank_payload in sample["ranks"]
        ]
        ranks = [
            _serialize_workload(item)
            for item in _validate_sample(
                [_deserialize_workload(item) for item in ranks],
                ep_size=int(runtime_ep_size),
                num_local_physical_experts=num_local_physical_experts,
            )
        ]
        sample_id = (
            f"context-pass{sample.get('forward_pass_id', 1)}-"
            f"layer{sample['layer_id']}-extrapolated-from-{donor_tokens}"
        )
        samples.append(
            {
                "sample_id": sample_id,
                "forward_pass_id": int(sample.get("forward_pass_id", 1)),
                "phase": "context",
                "table_num_tokens": int(context_table_num_tokens),
                "layer_id": int(sample["layer_id"]),
                "ranks": ranks,
            }
        )
        physical_counts = [
            count
            for workload in ranks
            for count in workload["num_recv_tokens_per_expert"]
        ]
        total_assignments = sum(physical_counts)
        active_experts = sum(count > 0 for count in physical_counts)
        mean_assignments = total_assignments / max(1, len(physical_counts))
        summary_rows.append(
            {
                "phase": "context",
                "num_tokens": int(context_table_num_tokens),
                "layer_id": int(sample["layer_id"]),
                "total_assignments": total_assignments,
                "active_experts": active_experts,
                "max_assignments": max(physical_counts, default=0),
                "mean_assignments": mean_assignments,
                "expert_assignments_json": json.dumps(
                    physical_counts,
                    separators=(",", ":"),
                ),
                "sample_id": sample_id,
            }
        )
    if not samples:
        raise ValueError(f"Donor bundle has no context samples: {donor_bundle}")

    name = (
        f"dsv3_{workload_source}_ep{requested_ep_size}_runtimeep{runtime_ep_size}_"
        f"eplb{int(enable_eplb)}_ctx{context_table_num_tokens}_"
        f"gen{requested_ep_size}_extrapolated.pt"
    )
    bundle_path = replay_dir / name
    new_payload = dict(payload)
    new_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "workload_source": workload_source,
            "requested_ep_size": requested_ep_size,
            "runtime_ep_size": runtime_ep_size,
            "enable_eplb": bool(enable_eplb),
            "topk": topk,
            "num_logical_experts": num_logical_experts,
            "samples": samples,
            "extrapolated_from_num_tokens": donor_tokens,
            "extrapolation_method": "repeat_rank_local_rows",
        }
    )
    torch.save(new_payload, bundle_path)
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    relative_bundle = bundle_path.name
    for row in summary_rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "distribution": "recorded",
                "workload_source": workload_source,
                "enable_eplb": bool(enable_eplb),
                "requested_ep_size": requested_ep_size,
                "runtime_ep_size": runtime_ep_size,
                "topk": topk,
                "num_experts": num_logical_experts,
                "replay_file": relative_bundle,
                "replay_sha256": digest,
                "extrapolated_from_num_tokens": donor_tokens,
            }
        )
    manifest_rows = [
        row for row in manifest_rows if row.get("replay_file") != relative_bundle
    ] + summary_rows
    manifest_fields = []
    for row in manifest_rows:
        for field in row:
            if field not in manifest_fields:
                manifest_fields.append(field)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return bundle_path, summary_rows, donor_tokens


def _synthetic_global_topk_ids(
    *,
    table_num_tokens: int,
    topk: int,
    num_physical_experts: int,
    layer_id: int,
    phase: str,
    enable_eplb: bool,
    decode_effective_topk: float | None = None,
) -> torch.Tensor:
    """Build a deterministic profile-free router layout.

    This is not a server/profile-derived router trace.  It gives the
    single-card collector a stable EP1/2/4/8 workload with global physical
    expert IDs, which can then be split into rank-local replay samples.
    """

    rows = max(0, int(table_num_tokens))
    topk = max(1, int(topk))
    experts = max(1, int(num_physical_experts))
    output = torch.full((rows, topk), -1, dtype=torch.int32)
    phase_offset = 29 if phase == "generation" else 0
    eplb_offset = 11 if enable_eplb else 0
    stride = 37 if phase == "generation" else 53
    if phase == "generation" and decode_effective_topk is not None:
        effective_topk = max(1.0, min(float(topk), float(decode_effective_topk)))
    else:
        effective_topk = float(topk)
    hot_experts = [
        (layer_id * 17 + phase_offset + eplb_offset + item * stride) % experts
        for item in range(max(topk, 16))
    ]
    for token in range(rows):
        # Mix a small hot set with a full-expert sweep.  The hot slots keep the
        # layout non-uniform, while the sweep prevents a single-rank workload
        # from collapsing to only a few experts.
        base = (token * (topk + 3) + layer_id * 7 + phase_offset + eplb_offset) % experts
        active_slots = int(math.floor(effective_topk))
        fractional = effective_topk - active_slots
        if fractional > 0.0:
            period = max(1, int(round(1.0 / fractional)))
            if (token + layer_id + (1 if enable_eplb else 0)) % period == 0:
                active_slots += 1
        active_slots = max(1, min(topk, active_slots))
        for slot in range(active_slots):
            if slot < max(1, topk // 2):
                expert = hot_experts[(token + slot * 3) % len(hot_experts)]
            else:
                expert = (base + slot * stride + token // max(1, experts // 8)) % experts
            output[token, slot] = int(expert)
    return output


def _rank_local_workloads_from_global_topk(
    *,
    topk_ids: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    phase: str,
) -> tuple[RankLocalWorkload, ...]:
    num_local_physical_experts = num_physical_experts // ep_size
    workloads: list[RankLocalWorkload] = []
    for rank in range(ep_size):
        rank_begin = rank * num_local_physical_experts
        rank_end = rank_begin + num_local_physical_experts
        local_mask = (topk_ids >= rank_begin) & (topk_ids < rank_end)
        local_ids = torch.where(local_mask, topk_ids - rank_begin, -1).to(torch.int32)
        token_mask = local_mask.any(dim=1)
        local_topk_ids = _deduplicate_local_topk_ids(local_ids[token_mask])
        counts = torch.bincount(
            local_topk_ids[local_topk_ids >= 0].to(torch.int64),
            minlength=num_local_physical_experts,
        ).to(torch.int32)
        active_experts = int((counts > 0).sum().item())
        if phase == "generation":
            divisor = max(1, active_experts)
        else:
            divisor = max(1, num_local_physical_experts)
        workloads.append(
            RankLocalWorkload(
                rank=rank,
                local_topk_ids=local_topk_ids.contiguous(),
                num_recv_tokens_per_expert=tuple(
                    int(value) for value in counts.tolist()
                ),
                masked_m=counts,
                expected_m=int(math.ceil(int(counts.sum().item()) / divisor)),
                dispatch_layout={
                    "aic_record_type": "single_card_materialized",
                    "phase": phase,
                },
            )
        )
    return _validate_sample(
        workloads,
        ep_size=ep_size,
        num_local_physical_experts=num_local_physical_experts,
    )


def _deterministic_rank_total_shape(
    *,
    table_num_tokens: int,
    topk: int,
    ep_size: int,
    phase: str,
    enable_eplb: bool,
    layer_id: int,
) -> list[int]:
    # Normal prefill replay receives token-to-expert assignments after top-k
    # dispatch, so its rank-local total scales with tokens * topk.  For DeepEP
    # low-latency decode still executes top-k expert assignments in the WideEP
    # replay table.  Keep the default at the model topk so the materialized
    # single-card EP path has the same assignment-count semantics as the
    # ordinary collector and historical WideEP operator tables.  The knob stays
    # configurable for silicon bring-up experiments, but the default path must
    # not silently shrink decode workload shape.
    if phase == "generation":
        effective_topk = float(
            os.environ.get(
                "COLLECTOR_MOE_DISTRIBUTION_DETERMINISTIC_DECODE_EFFECTIVE_TOPK",
                str(topk),
            )
        )
        effective_topk = max(1.0, min(float(topk), effective_topk))
        total = max(0, int(round(int(table_num_tokens) * effective_topk)))
    else:
        total = max(0, int(table_num_tokens) * max(1, int(topk)))
    if ep_size <= 1:
        return [total]
    mean = total / max(1, ep_size)
    # Deterministic single-card EP simulation should not be rank-flat: real
    # router output typically has a few heavier destination ranks even when the
    # global assignment count is fixed.  These are fixed profile-free priors,
    # not learned from server/profile truth or bootstrap files.
    if phase == "generation":
        amplitude = 0.55 if enable_eplb else 0.48
    else:
        base_amplitude = 0.56 if enable_eplb else 0.54
        if ep_size <= 2:
            base_amplitude *= 0.90
        amplitude = min(0.68, base_amplitude)
    phase_shift = 0.41 if phase == "generation" else 0.0
    eplb_shift = 0.29 if enable_eplb else 0.0
    weights = []
    for rank in range(ep_size):
        angle = 2.0 * math.pi * (
            (rank + (layer_id % ep_size) * 0.37) / ep_size + phase_shift + eplb_shift
        )
        harmonic = math.cos(angle) + 0.35 * math.sin(2.0 * angle + 0.7)
        weights.append(max(0.18, 1.0 + amplitude * harmonic))
    scale = total / max(1.0, sum(weights))
    rank_totals = [max(1, int(round(weight * scale))) for weight in weights]
    diff = total - sum(rank_totals)
    order = sorted(range(ep_size), key=lambda rank: weights[rank], reverse=True)
    cursor = 0
    while diff:
        rank = order[cursor % ep_size]
        if diff > 0:
            rank_totals[rank] += 1
            diff -= 1
        elif rank_totals[rank] > 1:
            rank_totals[rank] -= 1
            diff += 1
        cursor += 1
        if cursor > ep_size * 16 and diff < 0:
            break
    return rank_totals


def _deterministic_counts_for_rank(
    *,
    total: int,
    num_local_physical_experts: int,
    phase: str,
    enable_eplb: bool,
    rank: int,
    layer_id: int,
) -> torch.Tensor:
    total = max(0, int(total))
    if total == 0:
        return torch.zeros(num_local_physical_experts, dtype=torch.int32)
    # Keep most local experts active for prefill, while decode stays sparse:
    # runtime low-latency replay commonly has only a few local experts active
    # per rank.  This is a deterministic structural prior, not fitted from
    # server/profile truth.
    if phase == "generation":
        if total < 16:
            active = int(round(total * 0.62))
            alpha = 0.65
        elif total < 64:
            active = int(round(math.sqrt(total) * 1.45))
            alpha = 0.85
        elif total < 256:
            active = int(round(math.sqrt(total) * 2.1))
            alpha = 0.95
        elif total < 384:
            active = int(round(math.sqrt(total) * 1.9))
            alpha = 1.75
        elif total < 768:
            active = int(round(math.sqrt(total) * 1.9))
            alpha = 1.45
        elif total < 1536:
            active = num_local_physical_experts
            alpha = 2.10
        else:
            active = num_local_physical_experts
            alpha = 1.75
        active = min(num_local_physical_experts, max(1, active))
    else:
        target_nonzero_mean = 5.7
        min_active = min(
            num_local_physical_experts,
            max(1, int(round(math.sqrt(total) * 1.8))),
        )
        active = min(
            num_local_physical_experts,
            max(min_active, int(round(total / target_nonzero_mean))),
        )
        alpha = 0.82
    if enable_eplb and phase != "generation":
        alpha *= 0.92
    weights = []
    for index in range(active):
        position = index / max(1, active - 1)
        ripple = 1.0 + 0.10 * math.sin((index + rank * 3 + layer_id) * 1.37)
        weights.append(max(0.05, (1.0 - position) ** alpha * ripple))
    scale = total / max(1.0, sum(weights))
    counts = [max(1, int(round(weight * scale))) for weight in weights]
    diff = total - sum(counts)
    cursor = 0
    while diff:
        if diff > 0:
            # Fill the upper tail first for decode so the deterministic
            # single-card materializer keeps a realistic hot-expert tail.  For
            # prefill, keep the previous broader mid-tail fill to avoid making
            # context too spiky.
            start = 0 if phase == "generation" else max(0, int(active * 0.08))
            stop = (
                max(start + 1, int(active * 0.18))
                if phase == "generation"
                else max(start + 1, int(active * 0.42))
            )
            candidates = list(range(start, min(stop, active)))
            pos = min(candidates, key=lambda item: counts[item])
            counts[pos] += 1
            diff -= 1
        else:
            candidates = [idx for idx in range(active) if counts[idx] > 1]
            if not candidates:
                break
            pos = max(candidates, key=lambda item: counts[item])
            counts[pos] -= 1
            diff += 1
        cursor += 1
        if cursor > max(active * 32, total * 2):
            break
    padded = counts + [0] * (num_local_physical_experts - len(counts))
    tensor = torch.tensor(padded[:num_local_physical_experts], dtype=torch.int32)
    shift = (rank * 7 + int(layer_id) * 3 + (5 if enable_eplb else 0)) % num_local_physical_experts
    return torch.roll(tensor, shifts=shift)


def _rank_local_workloads_from_deterministic_shape(
    *,
    table_num_tokens: int,
    topk: int,
    ep_size: int,
    num_physical_experts: int,
    phase: str,
    enable_eplb: bool,
    layer_id: int,
) -> tuple[RankLocalWorkload, ...]:
    num_local_physical_experts = num_physical_experts // ep_size
    rank_totals = _deterministic_rank_total_shape(
        table_num_tokens=table_num_tokens,
        topk=topk,
        ep_size=ep_size,
        phase=phase,
        enable_eplb=enable_eplb,
        layer_id=layer_id,
    )
    workloads = []
    for rank, total in enumerate(rank_totals):
        counts = _deterministic_counts_for_rank(
            total=int(total),
            num_local_physical_experts=num_local_physical_experts,
            phase=phase,
            enable_eplb=enable_eplb,
            rank=rank,
            layer_id=layer_id,
        )
        local_topk_ids = _topk_rows_from_counts(
            counts,
            topk=topk,
            one_assignment_per_row=True,
        )
        active_experts = int((counts > 0).sum().item())
        divisor = max(1, active_experts if phase == "generation" else num_local_physical_experts)
        workloads.append(
            RankLocalWorkload(
                rank=rank,
                local_topk_ids=local_topk_ids,
                num_recv_tokens_per_expert=tuple(int(value) for value in counts.tolist()),
                masked_m=counts,
                expected_m=int(math.ceil(int(counts.sum().item()) / divisor)),
                dispatch_layout={
                    "aic_record_type": "single_card_materialized_deterministic_shape",
                    "phase": phase,
                },
            )
        )
    return _validate_sample(
        workloads,
        ep_size=ep_size,
        num_local_physical_experts=num_local_physical_experts,
    )


def _topk_rows_from_counts(
    counts: torch.Tensor,
    *,
    topk: int,
    one_assignment_per_row: bool = False,
) -> torch.Tensor:
    values: list[int] = []
    for expert_id, count in enumerate(counts.tolist()):
        values.extend([int(expert_id)] * int(count))
    width = 1 if one_assignment_per_row else max(1, int(topk))
    rows = max(1, math.ceil(len(values) / width))
    output = torch.full((rows, int(topk)), -1, dtype=torch.int32)
    for index, expert_id in enumerate(values):
        output[index // width, index % width] = int(expert_id)
    return output.contiguous()


def materialize_synthetic_replay_bundle(
    *,
    output_dir: str | Path,
    model: str,
    requested_ep_size: int,
    enable_eplb: bool,
    topk: int,
    num_logical_experts: int,
    first_moe_layer_id: int,
    num_layers: int,
    phase: str,
    table_num_tokens: int,
    workload_source: str = "dummy",
    diagnostics_path: str | Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Materialize EP-rank replay without launching runtime EP ranks."""

    requested_ep_size = int(requested_ep_size)
    if requested_ep_size <= 0:
        raise ValueError(f"requested_ep_size must be positive, got {requested_ep_size}")
    if int(num_logical_experts) % requested_ep_size:
        raise ValueError(
            f"num_logical_experts={num_logical_experts} must be divisible by EP={requested_ep_size}"
        )
    phase = str(phase).lower()
    if phase not in {"context", "generation"}:
        raise ValueError(f"phase must be context/generation, got {phase!r}")
    materialization_method = "single_card_deterministic_router_layout"

    output_dir = Path(output_dir)
    workload_source = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(workload_source).strip().lower()
    ) or "dummy"
    num_layers = max(1, int(num_layers))
    num_logical_experts = int(num_logical_experts)
    physical_to_logical_map = torch.arange(
        num_logical_experts,
        dtype=torch.int64,
    ).repeat(num_layers, 1)
    replay_layers = [
        first_moe_layer_id + layer_index
        for layer_index in range(max(1, num_layers - first_moe_layer_id))
    ]
    if not replay_layers:
        replay_layers = [int(first_moe_layer_id)]

    samples = []
    summary_rows = []
    diagnostics_rows = []
    for layer_id in replay_layers:
        workloads = _rank_local_workloads_from_deterministic_shape(
            table_num_tokens=int(table_num_tokens),
            topk=topk,
            ep_size=requested_ep_size,
            num_physical_experts=num_logical_experts,
            phase=phase,
            enable_eplb=bool(enable_eplb),
            layer_id=int(layer_id),
        )
        sample_id = f"{phase}-materialized-layer{layer_id}"
        samples.append(
            {
                "sample_id": sample_id,
                "forward_pass_id": 1,
                "phase": phase,
                "table_num_tokens": int(table_num_tokens),
                "layer_id": int(layer_id),
                "ranks": [_serialize_workload(item) for item in workloads],
            }
        )
        physical_counts = [
            count
            for workload in workloads
            for count in workload.num_recv_tokens_per_expert
        ]
        total_assignments = sum(physical_counts)
        active_experts = sum(count > 0 for count in physical_counts)
        mean_assignments = total_assignments / max(1, len(physical_counts))
        summary_rows.append(
            {
                "phase": phase,
                "num_tokens": int(table_num_tokens),
                "layer_id": int(layer_id),
                "total_assignments": total_assignments,
                "active_experts": active_experts,
                "max_assignments": max(physical_counts, default=0),
                "mean_assignments": mean_assignments,
                "expert_assignments_json": json.dumps(
                    physical_counts,
                    separators=(",", ":"),
                ),
                "sample_id": sample_id,
            }
        )
        for rank_workload in workloads:
            diagnostics_rows.append(
                {
                    "model": model,
                    "workload_source": workload_source,
                    "phase": phase,
                    "forward_pass_id": 1,
                    "rank": int(rank_workload.rank),
                    "layer_index": int(layer_id),
                    "layer_id": int(layer_id),
                    "count_kind": "single_card_materialized_rank_local",
                    "semantics": "physical_rank_local_materialized",
                    "requested_ep_size": requested_ep_size,
                    "runtime_ep_size": 1,
                    "materialization_method": materialization_method,
                    "enable_eplb": bool(enable_eplb),
                    "topk": topk,
                    "table_num_tokens": int(table_num_tokens),
                    **_count_summary(
                        rank_workload.num_recv_tokens_per_expert,
                        num_experts=num_logical_experts // requested_ep_size,
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    context_tokens = int(table_num_tokens) if phase == "context" else 0
    generation_tokens = int(table_num_tokens) if phase == "generation" else 0
    name = (
        f"dsv3_{workload_source}_ep{requested_ep_size}_runtimeep1_"
        f"eplb{int(enable_eplb)}_ctx{context_tokens}_gen{generation_tokens}_"
        "materialized.pt"
    )
    bundle_path = output_dir / name
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "workload_source": workload_source,
        "requested_ep_size": requested_ep_size,
        "runtime_ep_size": requested_ep_size,
        "materialized_runtime_ep_size": 1,
        "enable_eplb": bool(enable_eplb),
        "topk": topk,
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": num_logical_experts,
        "num_local_physical_experts": num_logical_experts // requested_ep_size,
        "first_moe_layer_id": first_moe_layer_id,
        "physical_to_logical_map": physical_to_logical_map,
        "samples": samples,
        "materialization_method": materialization_method,
    }
    torch.save(payload, bundle_path)
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    relative_bundle = bundle_path.name
    for row in summary_rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "distribution": "recorded",
                "workload_source": workload_source,
                "enable_eplb": bool(enable_eplb),
                "requested_ep_size": requested_ep_size,
                "runtime_ep_size": requested_ep_size,
                "materialized_runtime_ep_size": 1,
                "materialization_method": materialization_method,
                "topk": topk,
                "num_experts": num_logical_experts,
                "replay_file": relative_bundle,
                "replay_sha256": digest,
            }
        )
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_rows = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
    manifest_rows = [
        row for row in manifest_rows if row.get("replay_file") != relative_bundle
    ] + summary_rows
    manifest_fields = []
    for row in manifest_rows:
        for field in row:
            if field not in manifest_fields:
                manifest_fields.append(field)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    if diagnostics_path is not None and diagnostics_rows:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_diagnostics_rows(path=diagnostics_path, rows=diagnostics_rows)
    return bundle_path, summary_rows


def select_replay_workloads(
    *,
    replay_dir: str | Path,
    phase: str,
    table_num_tokens: int,
    layer_id: int,
    ep_size: int,
    num_experts: int,
    enable_eplb: bool,
    workload_source: str | None = None,
) -> list[tuple[RankLocalWorkload, ...]]:
    """Select exact-match replay samples.

    No cross-phase, cross-EP, cross-EPLB, cross-layer, or nearest-token fallback
    is allowed.  Calibration must fail loudly when the matching real workload
    was not collected.
    """

    replay_dir = Path(replay_dir)
    manifest_path = replay_dir / MANIFEST_FILENAME
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row
        for row in rows
        if row["phase"] == phase
        and int(row["num_tokens"]) == int(table_num_tokens)
        and int(row["layer_id"]) == int(layer_id)
        and int(row["requested_ep_size"]) == int(ep_size)
        and int(row["num_experts"]) == int(num_experts)
        and (
            (str(row["enable_eplb"]).lower() in ("1", "true"))
            == bool(enable_eplb)
        )
        and (
            workload_source is None
            or row.get("workload_source", "runtime") == workload_source
        )
    ]
    if not matches:
        raise FileNotFoundError(
            "No exact rank-local MoE replay for "
            f"phase={phase}, tokens={table_num_tokens}, layer={layer_id}, "
            f"ep={ep_size}, experts={num_experts}, eplb={enable_eplb}, "
            f"source={workload_source or 'any'}"
        )
    samples = []
    seen = set()
    for row in matches:
        bundle_path = replay_dir / row["replay_file"]
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        if digest != row["replay_sha256"]:
            raise ValueError(f"Replay checksum mismatch: {bundle_path}")
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
        if int(payload["schema_version"]) not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported replay schema {payload['schema_version']} in {bundle_path}"
            )
        for sample in payload["samples"]:
            key = (row["replay_file"], sample["sample_id"])
            if key in seen:
                continue
            if (
                sample["phase"] == phase
                and int(sample["table_num_tokens"]) == int(table_num_tokens)
                and int(sample["layer_id"]) == int(layer_id)
            ):
                workloads = _validate_sample(
                    [_deserialize_workload(item) for item in sample["ranks"]],
                    ep_size=int(payload["runtime_ep_size"]),
                    num_local_physical_experts=int(
                        payload["num_local_physical_experts"]
                    ),
                )
                samples.append(workloads)
                seen.add(key)
    if not samples:
        raise FileNotFoundError("Manifest matched but bundle contained no matching samples")
    return samples
