from __future__ import annotations

import warnings

import pytest

from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.kernelsim import analytical
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


@pytest.fixture(autouse=True)
def reset_w8a16_warning() -> None:
    moe_model._w8a16_warning_emitted = False


def test_w8a16_changes_only_expert_weight_and_scale_traffic() -> None:
    bf16 = moe_model.work_terms("bf16_triton", **_SHAPE)
    w8a16 = moe_model.work_terms("w8a16_int8wo_bf16_transfer", **_SHAPE)

    assert w8a16["required_peak_field"] == "bfloat16_tc_flops"
    assert w8a16["flops_total"] == bf16["flops_total"]
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
        assert w8a16[key] == bf16[key]

    assert w8a16["bytes_gemm1_weight_values"] == bf16["bytes_gemm1_weight_values"] / 2
    assert w8a16["bytes_gemm2_weight_values"] == bf16["bytes_gemm2_weight_values"] / 2
    active = float(w8a16["active_experts"])
    hidden = _SHAPE["hidden_size"]
    inter = _SHAPE["inter_size"]
    assert w8a16["bytes_gemm1_weight_scale"] == 16.0 * active * inter
    assert w8a16["bytes_gemm2_weight_scale"] == 4.0 * active * hidden
    assert w8a16["bytes_total"] < bf16["bytes_total"]


def test_w8a16_reuses_bf16_parameters_and_marks_ep_extrapolation() -> None:
    kwargs = {
        **_SHAPE,
        "peak_flops_s": 989e12,
        "mem_bandwidth_bytes_s": 3.35e12,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = moe_model.estimate_sglang_moe(
            "w8a16_int8wo_bf16_transfer", **kwargs
        )
        ep_result = moe_model.estimate_sglang_moe(
            "w8a16_int8wo_bf16_transfer", **{**kwargs, "moe_ep_size": 2}
        )

    parameters = moe_model.get_moe_parameters().bf16_triton
    assert result.required_peak_field == "bfloat16_tc_flops"
    assert result.launch_us == parameters.t_launch_us
    assert result.eta_compute == parameters.eta_compute
    assert result.eta_memory == parameters.eta_mem
    assert result.scope == "provisional w8a16_int8wo BF16 parameter transfer; EP=1 assumption"
    assert "ideal uniform EP extrapolation" in ep_result.scope
    transfer = [item for item in caught if item.category is moe_model.W8A16TransferModelWarning]
    assert len(transfer) == 1
    assert "EP>1" in str(transfer[0].message)


def test_analytical_w8a16_requires_only_bf16_peak() -> None:
    gpu = {"bfloat16_tc_flops": 989e12, "mem_bw": 3.35e12}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", moe_model.W8A16TransferModelWarning)
        latency_ms = analytical.moe_latency_ms(
            **_SHAPE,
            quant_mode=common.MoEQuantMode.int8_wo,
            gpu=gpu,
            config=analytical.AnalyticalConfig(),
        )
    assert latency_ms > 0


def test_w8a16_is_not_claimed_as_a_calibrated_silicon_recipe() -> None:
    assert "uncalibrated" in moe_model.W8A16_TRANSFER_LIMITATION
    assert "collector" in moe_model.W8A16_TRANSFER_LIMITATION
    assert "not a Silicon calibration row" in moe_model.W8A16_TRANSFER_LIMITATION


def test_task_yaml_and_cli_propagate_moe_w8a16() -> None:
    from_yaml = Task.from_yaml({"moe_quant_mode": "int8_wo"})
    from_cli = Task.from_cli(moe_quant_mode=common.MoEQuantMode.int8_wo)

    assert from_yaml.moe_quant_mode is common.MoEQuantMode.int8_wo
    assert from_cli.moe_quant_mode is common.MoEQuantMode.int8_wo
    assert from_yaml.to_dict()["moe_quant_mode"] == "int8_wo"
