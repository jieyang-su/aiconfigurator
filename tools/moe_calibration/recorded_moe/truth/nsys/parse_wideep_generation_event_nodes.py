#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a candidate WideEP generation routed/compute truth from nsys nodes.

The parser uses server-side WideEP compute NVTX markers only to learn the
CUDA graph node identity at graph-capture time:

1. find capture-time `aic_nsys/layer_N/routed/compute` markers before
   `cudaProfilerStart`;
2. collect CUDA graph nodes created inside each marker on the same globalTid;
3. map those capture nodes through `originalGraphNodeId` to replay
   `graphNodeId`s;
4. read replay kernels with those graphNodeIds after `cudaProfilerStart`;
5. split replay executions by launch-sized gaps and report union kernel time.

This remains a candidate path until multi-point/multi-session audits pass.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMPUTE_MARKER_RE = re.compile(r"^aic_nsys/layer_(?P<layer>\d+)/routed/compute$")
EXCLUDE_PATTERNS = (
    "deep_ep",
    "nccl",
    "nvshmem",
    "attention",
    "flash_attn",
    "flashinfer::norm",
    "rotary",
    "set_mla",
    "shared",
    "topk",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    return {int(row[0]): str(row[1]) for row in conn.execute("select id,value from StringIds")}


def _resolve_name(value: Any, string_ids: dict[int, str]) -> str:
    if isinstance(value, int):
        return string_ids.get(value, str(value))
    return str(value)


def _short_kernel_name(name: str) -> str:
    if name.startswith("void "):
        name = name[5:]
    if "<" in name:
        name = name.split("<", 1)[0]
    if "(" in name:
        name = name.split("(", 1)[0]
    return name[:140]


def _union_us(rows: list[dict[str, Any]]) -> float:
    intervals = sorted((int(row["start"]), int(row["end"])) for row in rows)
    if not intervals:
        return 0.0
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total / 1000.0


def _profile_start_ns(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where nameId=(select id from StringIds where value=? limit 1)",
        ("cudaProfilerStart_v4000",),
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("Could not find cudaProfilerStart_v4000 in runtime table")
    return int(row[0])


def _load_replay_kernel_nodes(
    conn: sqlite3.Connection, profile_start_ns: int
) -> set[int]:
    return {
        int(row[0])
        for row in conn.execute(
            "select distinct graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
            "where start>=? and graphNodeId is not null",
            (profile_start_ns,),
        )
    }


def _load_orig_to_replay(
    conn: sqlite3.Connection, replay_kernel_nodes: set[int]
) -> dict[tuple[int, int], set[int]]:
    out: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in conn.execute(
        "select graphNodeId,originalGraphNodeId,globalTid from CUDA_GRAPH_NODE_EVENTS "
        "where originalGraphNodeId is not null"
    ):
        graph_node = int(row["graphNodeId"])
        if graph_node in replay_kernel_nodes:
            out[(int(row["originalGraphNodeId"]), int(row["globalTid"]))].add(graph_node)
    return out


def _learn_layer_nodes(
    conn: sqlite3.Connection,
    profile_start_ns: int,
    orig_to_replay: dict[tuple[int, int], set[int]],
) -> tuple[dict[int, set[int]], list[dict[str, Any]]]:
    layer_nodes: dict[int, set[int]] = defaultdict(set)
    audit_rows: list[dict[str, Any]] = []
    for row in conn.execute(
        "select start,end,text,globalTid from NVTX_EVENTS "
        "where text like ? and start<? order by start",
        ("aic_nsys/layer_%/routed/compute", profile_start_ns),
    ):
        text = str(row["text"])
        match = COMPUTE_MARKER_RE.match(text)
        if not match:
            continue
        layer = int(match.group("layer"))
        global_tid = int(row["globalTid"])
        start = int(row["start"])
        end = int(row["end"])
        original_nodes = {
            int(node_row["graphNodeId"])
            for node_row in conn.execute(
                "select graphNodeId from CUDA_GRAPH_NODE_EVENTS "
                "where globalTid=? and start between ? and ?",
                (global_tid, start, end),
            )
        }
        replay_nodes: set[int] = set()
        for original in original_nodes:
            replay_nodes.update(orig_to_replay.get((original, global_tid), set()))
        if replay_nodes:
            layer_nodes[layer].update(replay_nodes)
            audit_rows.append(
                {
                    "layer": layer,
                    "globalTid": global_tid,
                    "marker_start": start,
                    "marker_duration_us": round((end - start) / 1000.0, 3),
                    "original_nodes": len(original_nodes),
                    "replay_nodes": len(replay_nodes),
                    "replay_node_ids": ";".join(str(node) for node in sorted(replay_nodes)),
                }
            )
    return layer_nodes, audit_rows


def _load_replay_kernels(
    conn: sqlite3.Connection,
    string_ids: dict[int, str],
    profile_start_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in conn.execute(
        "select start,end,deviceId,streamId,graphId,graphNodeId,"
        "coalesce(demangledName, shortName, mangledName) as name "
        "from CUPTI_ACTIVITY_KIND_KERNEL "
        "where start>=? and graphNodeId is not null order by start",
        (profile_start_ns,),
    ):
        name = _short_kernel_name(_resolve_name(row["name"], string_ids))
        rows.append(
            {
                "start": int(row["start"]),
                "end": int(row["end"]),
                "deviceId": int(row["deviceId"]),
                "streamId": int(row["streamId"]),
                "graphId": int(row["graphId"]),
                "graphNodeId": int(row["graphNodeId"]),
                "name": name,
            }
        )
    return rows


def _split_replay_groups(
    kernels: list[dict[str, Any]], gap_threshold_us: float
) -> list[list[dict[str, Any]]]:
    if not kernels:
        return []
    gap_ns = int(gap_threshold_us * 1000.0)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_end: int | None = None
    for kernel in sorted(kernels, key=lambda row: (row["start"], row["end"])):
        if last_end is not None and kernel["start"] - last_end > gap_ns:
            groups.append(current)
            current = []
        current.append(kernel)
        last_end = max(last_end or kernel["end"], kernel["end"])
    if current:
        groups.append(current)
    return groups


def _top_kernel_summary(kernels: list[dict[str, Any]]) -> str:
    counter: Counter[str] = Counter()
    for kernel in kernels:
        counter[str(kernel["name"])] += int(kernel["end"]) - int(kernel["start"])
    return "; ".join(
        f"{name}:{duration / 1000.0:.3f}us"
        for name, duration in counter.most_common(8)
    )


def _excluded_us(kernels: list[dict[str, Any]]) -> float:
    total = 0
    for kernel in kernels:
        lowered = str(kernel["name"]).lower()
        if any(pattern in lowered for pattern in EXCLUDE_PATTERNS):
            total += int(kernel["end"]) - int(kernel["start"])
    return total / 1000.0


def parse_candidate(
    sqlite_path: Path,
    *,
    ep: int,
    eplb: str,
    session: int,
    token: int,
    gap_threshold_us: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    string_ids = _load_string_ids(conn)
    profile_start_ns = _profile_start_ns(conn)
    replay_kernel_nodes = _load_replay_kernel_nodes(conn, profile_start_ns)
    orig_to_replay = _load_orig_to_replay(conn, replay_kernel_nodes)
    layer_nodes, mapping_rows = _learn_layer_nodes(
        conn, profile_start_ns, orig_to_replay
    )
    replay_kernels = _load_replay_kernels(conn, string_ids, profile_start_ns)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for layer, nodes in sorted(layer_nodes.items()):
        all_layer_kernels = [
            row for row in replay_kernels if int(row["graphNodeId"]) in nodes
        ]
        if not all_layer_kernels:
            continue

        layer_groups = _split_replay_groups(all_layer_kernels, gap_threshold_us)
        for occurrence, layer_kernels in enumerate(layer_groups, start=1):
            excluded = _excluded_us(layer_kernels)
            union = _union_us(layer_kernels)
            detail_rows.append(
                {
                    "ep": ep,
                    "eplb": eplb,
                    "session": session,
                    "token": token,
                    "occurrence": occurrence,
                    "layer": layer,
                    "graph_start_ns": min(row["start"] for row in layer_kernels),
                    "graph_end_ns": max(row["end"] for row in layer_kernels),
                    "layer_kernel_rows": len(layer_kernels),
                    "layer_graph_nodes": len(nodes),
                    "union_us": round(union, 3),
                    "span_us": round(
                        (max(row["end"] for row in layer_kernels) - min(row["start"] for row in layer_kernels))
                        / 1000.0,
                        3,
                    ),
                    "excluded_kernel_us": round(excluded, 3),
                    "candidate_status": "ok" if excluded == 0 else "has_excluded_kernel",
                    "top_kernels": _top_kernel_summary(layer_kernels),
                    "source_file": str(sqlite_path),
                }
            )

    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in detail_rows:
        if row["candidate_status"] == "ok":
            by_layer[int(row["layer"])].append(float(row["union_us"]))
    for layer, values in sorted(by_layer.items()):
        values = sorted(values)
        median = values[len(values) // 2] if len(values) % 2 else (
            values[len(values) // 2 - 1] + values[len(values) // 2]
        ) / 2.0
        summary_rows.append(
            {
                "ep": ep,
                "eplb": eplb,
                "session": session,
                "token": token,
                "layer": layer,
                "occurrences": len(values),
                "candidate_union_median_us": round(median, 3),
                "candidate_union_min_us": round(min(values), 3),
                "candidate_union_max_us": round(max(values), 3),
                "candidate_status": "candidate_not_frozen",
                "truth_ready": "false",
                "source_file": str(sqlite_path),
            }
        )
    conn.close()
    return mapping_rows, detail_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--eplb", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--mapping-out", type=Path, required=True)
    parser.add_argument("--detail-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--gap-threshold-us", type=float, default=500.0)
    args = parser.parse_args()

    mapping_rows, detail_rows, summary_rows = parse_candidate(
        args.sqlite,
        ep=args.ep,
        eplb=args.eplb,
        session=args.session,
        token=args.token,
        gap_threshold_us=args.gap_threshold_us,
    )
    _write_csv(args.mapping_out, mapping_rows)
    _write_csv(args.detail_out, detail_rows)
    _write_csv(args.summary_out, summary_rows)
    print(f"mapping_out={args.mapping_out}")
    print(f"detail_out={args.detail_out}")
    print(f"summary_out={args.summary_out}")
    print(f"mapping_rows={len(mapping_rows)} detail_rows={len(detail_rows)} summary_rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
