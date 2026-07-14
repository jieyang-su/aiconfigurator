import json
import sys
from pathlib import Path


COLLECTOR_DIR = Path(__file__).resolve().parents[3] / "collector"
if str(COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_DIR))

from moe_recorded_materialization import context_recorded_rows, generation_recorded_rows


def _row(token: int, *, distribution: str = "recorded_no_eplb") -> dict[str, str]:
    return {
        "framework": "SGLang",
        "version": "test",
        "device": "NVIDIA H20",
        "op_name": "wideep_context_moe",
        "kernel_source": "unit",
        "moe_dtype": "fp8_block",
        "num_tokens": str(token),
        "hidden_size": "7168",
        "inter_size": "2048",
        "topk": "8",
        "num_experts": "256",
        "moe_tp_size": "1",
        "moe_ep_size": "8",
        "distribution": distribution,
        "latency": str(token / 1000.0),
    }


def test_context_materialization_keeps_all_recorded_context_points():
    sparse_rows = [_row(token) for token in (64, 128, 512, 640, 2048)]
    dense_rows = [_row(token) for token in (2048, 4096, 5120)]

    rows = context_recorded_rows(
        sparse_rows=sparse_rows,
        dense_rows=dense_rows,
        distribution="recorded_no_eplb",
    )

    by_token = {int(float(row["num_tokens"])): row for row in rows}
    assert sorted(by_token) == [64, 128, 512, 640, 2048, 4096, 5120]
    assert by_token[64]["materialization_role"] == "context_sparse"
    assert by_token[640]["materialization_role"] == "context_sparse"
    assert by_token[2048]["materialization_role"] == "context_sparse"
    assert by_token[4096]["materialization_role"] == "context_dense"


def test_generation_materialization_exports_rank_envelope_diagnostics():
    row = _row(8)
    row.update(
        {
            "op_name": "wideep_generation_moe",
            "kernel_regime": "low_latency_masked_capacity_1024",
            "rank_mean_latency": "0.20",
            "rank_p90_latency": "0.30",
            "latency_raw_max": "0.40",
            "rank_sync_tail_mean": "0.05",
            "workload_rank_imbalance_max_over_mean": "1.25",
            "workload_rank_assignments_max": "10",
            "workload_expert_m_max": "3",
            "rank_steady_mean_ms_json": json.dumps(
                {"0": 0.10, "1": 0.20, "2": 0.25, "3": 0.60},
                separators=(",", ":"),
            ),
            "rank_steady_p90_ms_json": json.dumps(
                {"0": 0.11, "1": 0.22, "2": 0.30, "3": 0.66},
                separators=(",", ":"),
            ),
            "stage_mean_ms_json": json.dumps({"cuda_graph_replay": 0.20}),
            "stage_p90_ms_json": json.dumps({"cuda_graph_replay": 0.50}),
        }
    )

    rows = generation_recorded_rows(
        small_rows=[row],
        main_rows=[],
        distribution="recorded_no_eplb",
    )

    assert len(rows) == 1
    result = rows[0]
    assert result["aic_rank_steady_mean_min"] == "0.1"
    assert result["aic_rank_steady_mean_max"] == "0.6"
    assert result["aic_rank_envelope_ms"] == "0.5"
    assert result["aic_rank_bimodal_gap_ms"] == "0.35"
    assert result["aic_rank_bimodal_gap_over_mean"] == "1.21739130435"
    assert result["aic_stage_replay_p90_over_mean"] == "2.5"
