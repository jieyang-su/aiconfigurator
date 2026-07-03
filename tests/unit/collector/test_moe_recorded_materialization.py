import sys
from pathlib import Path


COLLECTOR_DIR = Path(__file__).resolve().parents[3] / "collector"
if str(COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_DIR))

from moe_recorded_materialization import context_recorded_rows


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
