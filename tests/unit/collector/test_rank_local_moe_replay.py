from pathlib import Path
import csv

import pytest
import torch

from collector.wideep.sglang.rank_local_moe_replay import (
    materialize_replay_bundle,
    select_replay_workloads,
)
from collector.sglang.collect_moe_distribution import (
    get_moe_distribution_test_cases,
    run_moe_distribution,
)
from collector.sglang.collect_moe import (
    _parse_rank_local_distribution,
    _rank_local_replay_distributions,
)
from collector.wideep.sglang.collect_deepep_moe import (
    _replay_kernel_template_fingerprint,
    _replay_workload_features,
    _summarize_replay_timings,
)
from collector.wideep.sglang.rank_local_moe_replay import RankLocalWorkload


def _write_rank_record(path: Path, rank: int) -> None:
    global_topk = torch.tensor(
        [[[0, 2], [1, 3], [0, 3]]],
        dtype=torch.int32,
    )
    normal = {
        0: {
            "layer_id": 0,
            "local_physical_count_of_layer": [2, 1],
            "recv_topk_ids": torch.tensor([[0, 1], [0, -1]], dtype=torch.int32),
            "num_tokens_per_rank": [0, 0],
            "num_tokens_per_rdma_rank": [0, 0],
            "num_tokens_per_expert": [0, 0, 0, 0],
        },
        1: {
            "layer_id": 0,
            "local_physical_count_of_layer": [1, 1],
            "recv_topk_ids": torch.tensor([[0, 1]], dtype=torch.int32),
            "num_tokens_per_rank": [0, 0],
            "num_tokens_per_rdma_rank": [0, 0],
            "num_tokens_per_expert": [0, 0, 0, 0],
        },
    }
    payload = {
        "last_physical_to_logical_map": torch.tensor(
            [[0, 1, 2, 3]],
            dtype=torch.int64,
        ),
        "records": [
            {
                "forward_pass_id": 10,
                "rank": rank,
                "forward_mode": "extend",
                "input_ids": [1, 2, 3],
                "topk_ids_of_layer": global_topk,
                "misc_objects": [normal[rank]],
                "global_physical_count": torch.tensor([[2, 1, 1, 1]]),
            },
            {
                "forward_pass_id": 11,
                "rank": rank,
                "forward_mode": "decode",
                "input_ids": [4, 5, 6],
                "topk_ids_of_layer": global_topk,
                "misc_objects": [],
                "global_physical_count": torch.tensor([[2, 1, 1, 2]]),
            },
        ],
    }
    torch.save(payload, path)


def test_materialize_and_select_exact_rank_local_replay(tmp_path: Path):
    recorder_dir = tmp_path / "recorder"
    replay_dir = tmp_path / "replay"
    recorder_dir.mkdir()
    for rank in range(2):
        _write_rank_record(
            recorder_dir / f"expert_distribution_recorder_test_{rank}.pt",
            rank,
        )

    bundle, rows = materialize_replay_bundle(
        recorder_dir=recorder_dir,
        output_dir=replay_dir,
        model="deepseek-v3",
        requested_ep_size=2,
        runtime_ep_size=2,
        enable_eplb=True,
        topk=2,
        num_logical_experts=4,
        first_moe_layer_id=3,
        context_table_num_tokens=6,
        generation_table_num_tokens=8,
    )

    assert bundle.exists()
    assert {row["phase"] for row in rows} == {"context", "generation"}

    context = select_replay_workloads(
        replay_dir=replay_dir,
        phase="context",
        table_num_tokens=6,
        layer_id=3,
        ep_size=2,
        num_experts=4,
        enable_eplb=True,
    )
    assert len(context) == 1
    assert [item.num_recv_tokens_per_expert for item in context[0]] == [
        (2, 1),
        (1, 1),
    ]
    assert context[0][0].local_topk_ids.tolist() == [[0, 1], [0, -1]]

    generation = select_replay_workloads(
        replay_dir=replay_dir,
        phase="generation",
        table_num_tokens=8,
        layer_id=3,
        ep_size=2,
        num_experts=4,
        enable_eplb=True,
    )
    # Decode compute on each destination rank receives assignments from every
    # source rank through the all-to-all dispatch.
    assert [item.masked_m.tolist() for item in generation[0]] == [[4, 2], [2, 4]]


def test_materialize_writes_count_semantics_diagnostics(tmp_path: Path):
    recorder_dir = tmp_path / "recorder"
    replay_dir = tmp_path / "replay"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    recorder_dir.mkdir()
    for rank in range(2):
        _write_rank_record(
            recorder_dir / f"expert_distribution_recorder_test_{rank}.pt",
            rank,
        )

    materialize_replay_bundle(
        recorder_dir=recorder_dir,
        output_dir=replay_dir,
        model="deepseek-v3",
        requested_ep_size=2,
        runtime_ep_size=2,
        enable_eplb=True,
        topk=2,
        num_logical_experts=4,
        first_moe_layer_id=3,
        context_table_num_tokens=6,
        generation_table_num_tokens=8,
        diagnostics_path=diagnostics_csv,
    )

    rows = list(csv.DictReader(diagnostics_csv.open()))
    kinds = {row["count_kind"] for row in rows}
    assert kinds == {
        "router_global_physical_from_topk",
        "router_logical_from_topk",
        "post_dispatch_rank_local",
    }
    context_rank0 = [
        row
        for row in rows
        if row["phase"] == "context"
        and row["rank"] == "0"
        and row["count_kind"] == "post_dispatch_rank_local"
    ][0]
    assert context_rank0["semantics"] == "physical_rank_local_post_dispatch"
    assert context_rank0["expert_assignments_json"] == "[2,1]"


def test_selection_does_not_fallback_across_eplb(tmp_path: Path):
    recorder_dir = tmp_path / "recorder"
    replay_dir = tmp_path / "replay"
    recorder_dir.mkdir()
    for rank in range(2):
        _write_rank_record(
            recorder_dir / f"expert_distribution_recorder_test_{rank}.pt",
            rank,
        )
    materialize_replay_bundle(
        recorder_dir=recorder_dir,
        output_dir=replay_dir,
        model="deepseek-v3",
        requested_ep_size=2,
        runtime_ep_size=2,
        enable_eplb=False,
        topk=2,
        num_logical_experts=4,
        first_moe_layer_id=3,
        context_table_num_tokens=6,
        generation_table_num_tokens=8,
    )

    with pytest.raises(FileNotFoundError):
        select_replay_workloads(
            replay_dir=replay_dir,
            phase="context",
            table_num_tokens=6,
            layer_id=3,
            ep_size=2,
            num_experts=4,
            enable_eplb=True,
        )


def test_distribution_default_matrix_covers_ep_and_eplb(monkeypatch):
    for name in (
        "COLLECTOR_MOE_DISTRIBUTION_TOKENS",
        "COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS",
        "COLLECTOR_MOE_DISTRIBUTION_EP_SIZES",
        "COLLECTOR_MOE_DISTRIBUTION_EP_SIZE",
        "COLLECTOR_WIDEEP_MOE_EP_SIZES",
        "COLLECTOR_MOE_DISTRIBUTION_EPLB_MODES",
    ):
        monkeypatch.delenv(name, raising=False)

    cases = get_moe_distribution_test_cases()
    ep_eplb = {(case[3], case[4]) for case in cases}
    phases = {case[5] for case in cases}
    assert ep_eplb == {
        (1, False),
        (2, False),
        (2, True),
        (4, False),
        (4, True),
        (8, False),
        (8, True),
    }
    assert phases == {"context", "generation"}


def test_ordinary_moe_discovers_rank_local_replay(monkeypatch, tmp_path: Path):
    replay_dir = tmp_path / "moe_token_distribution_replay"
    replay_dir.mkdir()
    manifest = replay_dir / "manifest.csv"
    manifest.write_text(
        "phase,num_tokens,layer_id,workload_source,enable_eplb,"
        "requested_ep_size,num_experts\n"
        "context,128,3,dummy,False,2,256\n"
        "context,128,3,dummy,True,2,256\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COLLECTOR_CURRENT_OUTPUT_DIR", str(tmp_path))

    assert _rank_local_replay_distributions(
        num_tokens=128,
        ep_size=2,
        num_experts=256,
    ) == [
        "recorded_dummy_rank_local_no_eplb",
        "recorded_dummy_rank_local_eplb",
    ]
    assert _parse_rank_local_distribution(
        "recorded_dummy_rank_local_eplb"
    ) == ("dummy", "context", True)
    assert _parse_rank_local_distribution(
        "recorded_dummy_rank_local_no_eplb"
    ) == ("dummy", "context", False)
    assert _parse_rank_local_distribution(
        "recorded_dummy_rank_local_eplb1"
    ) == ("dummy", "context", True)
    assert _parse_rank_local_distribution(
        "recorded_dummy_generation_rank_local_no_eplb"
    ) == ("dummy", "generation", False)


def test_distribution_rejects_ep_larger_than_visible_gpu_set(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("COLLECTOR_MOE_DISTRIBUTION_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("COLLECTOR_MOE_DISTRIBUTION_TP_SIZE", raising=False)

    with pytest.raises(ValueError, match="requires 2 visible GPUs"):
        run_moe_distribution(
            128,
            1,
            1,
            2,
            False,
            perf_filename=tmp_path / "distribution.csv",
        )


def test_ep1_context_can_replay_direct_router_record(tmp_path: Path):
    recorder_dir = tmp_path / "recorder"
    replay_dir = tmp_path / "replay"
    recorder_dir.mkdir()
    payload = {
        "last_physical_to_logical_map": torch.tensor(
            [[0, 1, 2, 3]],
            dtype=torch.int64,
        ),
        "records": [
            {
                "forward_pass_id": 20,
                "rank": 0,
                "forward_mode": "extend",
                "topk_ids_of_layer": torch.tensor(
                    [[[0, 2], [1, 3], [0, 3]]],
                    dtype=torch.int32,
                ),
                "misc_objects": [],
            }
        ],
    }
    torch.save(payload, recorder_dir / "expert_distribution_recorder_ep1.pt")

    materialize_replay_bundle(
        recorder_dir=recorder_dir,
        output_dir=replay_dir,
        model="deepseek-v3",
        requested_ep_size=1,
        runtime_ep_size=1,
        enable_eplb=True,
        topk=2,
        num_logical_experts=4,
        first_moe_layer_id=3,
        context_table_num_tokens=3,
        generation_table_num_tokens=1,
    )
    context = select_replay_workloads(
        replay_dir=replay_dir,
        phase="context",
        table_num_tokens=3,
        layer_id=3,
        ep_size=1,
        num_experts=4,
        enable_eplb=True,
    )
    assert context[0][0].num_recv_tokens_per_expert == (2, 1, 1, 2)


def test_replay_summary_preserves_rank_max_and_workload_features():
    sample = (
        RankLocalWorkload(
            rank=0,
            local_topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            num_recv_tokens_per_expert=(3, 1),
            masked_m=torch.tensor([3, 1], dtype=torch.int32),
            expected_m=2,
        ),
        RankLocalWorkload(
            rank=1,
            local_topk_ids=torch.tensor([[0, -1]], dtype=torch.int32),
            num_recv_tokens_per_expert=(1, 0),
            masked_m=torch.tensor([1, 0], dtype=torch.int32),
            expected_m=1,
        ),
    )
    features = _replay_workload_features([sample])
    assert features["workload_rank_assignments_max"] == 4
    assert features["workload_rank_imbalance_max_over_mean"] == pytest.approx(1.6)
    assert features["workload_active_experts_mean"] == pytest.approx(1.5)
    assert features["workload_expert_m_p90"] == pytest.approx(2.6)

    summary = _summarize_replay_timings(
        cold_rank_latencies=[2.0, 3.0],
        steady_round_rank_latencies=[[1.0, 2.0], [1.5, 2.5]],
        steady_round_rank_critical_latencies=[[0.8, 1.8], [1.3, 2.3]],
        rank_latency_history={0: [1.0, 1.5], 1: [2.0, 2.5]},
        rank_critical_latency_history={0: [0.8, 1.3], 1: [1.8, 2.3]},
        stage_duration_history={"gemm1": [0.8, 1.0], "gemm2": [0.7, 0.9]},
        multistream_round_latencies=[],
        multistream_probe_error=None,
        workload_features=features,
        capacity=1024,
        kernel_regime="low_latency_masked_capacity_1024",
    )
    assert summary["latency"] == pytest.approx(2.05)
    assert summary["latency_max"] == pytest.approx(2.3)
    assert summary["cold_latency"] == pytest.approx(3.0)
    assert summary["rank_sync_tail_mean"] == pytest.approx(0.5)
    assert summary["replay_policy"] == "rank_local_replay_v2"
    assert summary["primary_latency_source"] == "critical_path"
    assert summary["measurement_boundary"] == "run_moe_core_cuda_critical_path"
    assert summary["latency_stage_sum_rankmax"] == pytest.approx(2.25)

    with_jit_outlier = _summarize_replay_timings(
        cold_rank_latencies=[400.0],
        steady_round_rank_latencies=[
            [1.0, 2.0],
            [1.1, 2.1],
            [1.0, 250.0],
        ],
        steady_round_rank_critical_latencies=[
            [0.9, 1.9],
            [1.0, 2.0],
            [0.9, 249.0],
        ],
        rank_latency_history={0: [1.0, 1.1, 1.0], 1: [2.0, 2.1, 250.0]},
        rank_critical_latency_history={
            0: [0.9, 1.0, 0.9],
            1: [1.9, 2.0, 249.0],
        },
        stage_duration_history={"gemm1": [1.0, 1.1, 100.0]},
        multistream_round_latencies=[],
        multistream_probe_error=None,
        workload_features=features,
        capacity=0,
        kernel_regime="normal_contiguous",
    )
    assert with_jit_outlier["latency"] == pytest.approx(1.95)
    assert with_jit_outlier["latency_max"] == pytest.approx(2.0)
    assert with_jit_outlier["latency_raw_max"] == pytest.approx(249.0)
    assert with_jit_outlier["measurement_outliers_discarded"] == 1

    with_probe_failure = _summarize_replay_timings(
        cold_rank_latencies=[2.0, 3.0],
        steady_round_rank_latencies=[[1.0, 2.0], [1.5, 2.5]],
        steady_round_rank_critical_latencies=[[0.8, 1.8], [1.3, 2.3]],
        rank_latency_history={0: [1.0, 1.5], 1: [2.0, 2.5]},
        rank_critical_latency_history={0: [0.8, 1.3], 1: [1.8, 2.3]},
        stage_duration_history={"gemm1": [0.8, 1.0], "gemm2": [0.7, 0.9]},
        multistream_round_latencies=[],
        multistream_probe_error="CUDA error: an illegal memory access was encountered",
        workload_features=features,
        capacity=1024,
        kernel_regime="low_latency_masked_capacity_1024",
    )
    assert with_probe_failure["latency"] == pytest.approx(2.05)
    assert with_probe_failure["multistream_latency"] == 0.0
    assert with_probe_failure["multistream_measurement_boundary"] == "probe_failed"
    assert "illegal memory access" in with_probe_failure["multistream_probe_error"]

    fingerprint = _replay_kernel_template_fingerprint(
        phase="generation",
        replay_samples=[sample],
        capacity=1024,
        kernel_regime="low_latency_masked_capacity_1024",
        observed_kernel_wrappers=[
            "grouped_gemm_nt_f8f8bf16_masked",
            "sglang_per_token_group_quant_fp8",
        ],
        observed_dispatch_layouts=[
            {
                "hidden_states": {
                    "shape": [2, 1024, 7168],
                    "stride": [7340032, 7168, 1],
                    "dtype": "float8_e4m3fn",
                    "is_contiguous": True,
                },
                "hidden_states_scale": {
                    "shape": [2, 1024, 56],
                    "stride": [57344, 56, 1],
                    "dtype": "float32",
                    "is_contiguous": True,
                },
                "topk_ids": None,
            }
        ],
    )
    assert fingerprint["gemm_path"] == "masked"
    assert fingerprint["quant_scale_path"] == "recorded_fp8_hidden_scale_layout"
    assert fingerprint["template_expected_m_max"] == 2
    assert fingerprint["template_masked_m_max"] == 3
