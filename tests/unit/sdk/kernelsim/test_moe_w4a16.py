from __future__ import annotations

import warnings

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig, moe_latency_ms
from aiconfigurator_core.sdk.kernelsim.moe import model as moe_model

_SHAPE = {
    "num_tokens": 128,
    "hidden_size": 4096,
    "inter_size": 14336,
    "topk": 2,
    "num_experts": 8,
    "moe_tp_size": 1,
    "moe_ep_size": 1,
}


def test_w4a16_changes_only_weight_and_scale_traffic() -> None:
    bf16 = moe_model.work_terms("bf16_triton", **_SHAPE)
    w4a16 = moe_model.work_terms("w4a16_mxfp4_bf16_transfer", **_SHAPE)

    assert w4a16["required_peak_field"] == "bfloat16_tc_flops"
    assert w4a16["flops_total"] == bf16["flops_total"]
    for key in (
        "bytes_routing_logits",
        "bytes_routing_topk",
        "bytes_input_quant",
        "bytes_gemm1_input",
        "bytes_gemm1_output",
        "bytes_activation_quant",
        "bytes_gemm2_input",
        "bytes_gemm2_output",
        "bytes_combine",
    ):
        assert w4a16[key] == bf16[key]

    assert w4a16["bytes_gemm1_weight_values"] == bf16["bytes_gemm1_weight_values"] / 4
    assert w4a16["bytes_gemm2_weight_values"] == bf16["bytes_gemm2_weight_values"] / 4
    assert w4a16["bytes_gemm1_weight_scale"] > 0
    assert w4a16["bytes_gemm2_weight_scale"] > 0
    assert w4a16["bytes_total"] < bf16["bytes_total"]


def test_w4a16_reuses_bf16_parameters_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moe_model, "_w4a16_warning_emitted", False)
    kwargs = {
        **_SHAPE,
        "peak_flops_s": 989e12,
        "mem_bandwidth_bytes_s": 3.35e12,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = moe_model.estimate_sglang_moe("w4a16_mxfp4_bf16_transfer", **kwargs)
        second = moe_model.estimate_sglang_moe("w4a16_mxfp4_bf16_transfer", **kwargs)

    bf16_parameters = moe_model.get_moe_parameters().bf16_triton
    assert first.launch_us == bf16_parameters.t_launch_us
    assert first.eta_compute == bf16_parameters.eta_compute
    assert first.eta_memory == bf16_parameters.eta_mem
    assert first.scope == "provisional BF16 parameter transfer; EP=1 assumption"
    assert second.latency_us == first.latency_us
    transfer_warnings = [item for item in caught if item.category is moe_model.W4A16TransferModelWarning]
    assert len(transfer_warnings) == 1
    assert "EP>1" in str(transfer_warnings[0].message)
    assert "low confidence" in str(transfer_warnings[0].message)


@pytest.mark.parametrize(
    "quant_mode",
    (common.MoEQuantMode.w4a16_mxfp4, common.MoEQuantMode.w4a16_mxfp4_cutlass),
)
def test_analytical_adapter_accepts_w4a16_with_only_bf16_peak(quant_mode) -> None:
    gpu = {
        "bfloat16_tc_flops": 989e12,
        "mem_bw": 3.35e12,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", moe_model.W4A16TransferModelWarning)
        latency_ms = moe_latency_ms(
            **_SHAPE,
            quant_mode=quant_mode,
            gpu=gpu,
            config=AnalyticalConfig(),
        )
    assert latency_ms > 0


def test_unrelated_unsupported_quant_mode_remains_rejected() -> None:
    gpu = {"bfloat16_tc_flops": 989e12, "mem_bw": 3.35e12}
    with pytest.raises(ValueError, match="does not support quant mode 'int4_wo'"):
        moe_latency_ms(
            **_SHAPE,
            quant_mode=common.MoEQuantMode.int4_wo,
            gpu=gpu,
            config=AnalyticalConfig(),
        )
