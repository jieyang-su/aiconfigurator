# SPDX-FileCopyrightText: Copyright (c) 2026 AIC contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.communication import P2P
from aiconfigurator.sdk.perf_database import load_system_spec
from aiconfigurator.sdk.rust_engine_step import should_use_rust_engine_step
from aiconfigurator.sdk.system_spec import (
    SystemSpec,
    is_blackwell_spec,
    is_hopper_spec,
    supports_fp4_mma,
    supports_fp8_mma,
    supports_mnnvl,
)
from aiconfigurator.sdk.utils import enumerate_parallel_config


def _spec(*, domestic: bool, capacity: int = 8) -> SystemSpec:
    gpu = {"sm_version": 90}
    node = {
        "num_gpus_per_node": capacity,
        "intra_node_bw": 400.0,
        "inter_node_bw": 25.0,
        "p2p_latency": 0.001,
    }
    if domestic:
        gpu.update(
            architecture_family="domestic",
            capability_overrides={"fp8_mma": True, "fp4_mma": False, "mnnvl": False},
        )
        node["topology_scope"] = "single_supernode"
    return SystemSpec({"gpu": gpu, "node": node})


def test_proxy_sm_does_not_claim_nvidia_architecture() -> None:
    domestic = _spec(domestic=True)
    assert not is_hopper_spec(domestic)
    assert not is_blackwell_spec(domestic)
    assert supports_fp8_mma(domestic)
    assert not supports_fp4_mma(domestic)
    assert not supports_mnnvl(domestic)

    hopper = _spec(domestic=False)
    assert is_hopper_spec(hopper)
    assert supports_fp8_mma(hopper)


@pytest.mark.parametrize(
    ("system_name", "fp8", "fp4"),
    [
        ("_dom_br100_64", False, False),
        ("_dom_br100_128", False, False),
        ("_dom_ascend_910c", False, False),
        ("_dom_klx_m300_512", True, False),
        ("_dom_ascend_950dt", True, True),
    ],
)
def test_packaged_domestic_specs_declare_explicit_capabilities(system_name: str, fp8: bool, fp4: bool) -> None:
    spec = load_system_spec(system_name)

    assert spec["gpu"]["architecture_family"] == "domestic"
    assert spec["gpu"]["capability_overrides"] == {
        "fp8_mma": fp8,
        "fp4_mma": fp4,
        "mnnvl": False,
    }
    assert spec["node"]["topology_scope"] == "single_supernode"
    assert spec["node"]["inter_node_bw"] == 0
    assert spec["node"]["num_gpus_per_node"] > 0
    assert not is_hopper_spec(spec)
    assert not is_blackwell_spec(spec)
    assert supports_fp8_mma(spec) is fp8
    assert supports_fp4_mma(spec) is fp4
    assert not supports_mnnvl(spec)


def test_single_supernode_rejects_oversized_groups() -> None:
    spec = _spec(domestic=True, capacity=8)
    spec.validate_parallelism(tp=2, pp=2, attention_dp=2, cp=1, moe_tp=2, moe_ep=2)
    with pytest.raises(ValueError, match="single-supernode capacity 8"):
        spec.validate_parallelism(tp=2, pp=4, attention_dp=2, cp=1, moe_tp=4, moe_ep=4)
    with pytest.raises(ValueError, match="communication group of 9 GPUs"):
        spec.get_p2p_bandwidth(9)


def test_pipeline_p2p_preserves_legacy_and_uses_supernode_bandwidth() -> None:
    message_bytes = 100
    legacy_db = SimpleNamespace(system_spec=_spec(domestic=False), _default_database_mode=common.DatabaseMode.SOL)
    domestic_db = SimpleNamespace(system_spec=_spec(domestic=True), _default_database_mode=common.DatabaseMode.SOL)

    legacy = P2P._query_p2p_table(legacy_db, message_bytes, common.DatabaseMode.SOL)
    domestic = P2P._query_p2p_table(domestic_db, message_bytes, common.DatabaseMode.SOL)

    assert float(legacy) == pytest.approx(message_bytes / 25.0 * 1000)
    assert float(domestic) == pytest.approx(message_bytes / 400.0 * 1000)


def test_parallel_enumeration_filters_single_supernode_overflow() -> None:
    configs = enumerate_parallel_config(
        num_gpu_list=[8, 16],
        tp_list=[2],
        pp_list=[2, 4],
        dp_list=[2],
        moe_tp_list=[2],
        moe_ep_list=[2],
        is_moe=True,
        backend=common.BackendName.sglang,
        single_supernode_gpus=8,
    )
    assert configs == [[2, 2, 2, 2, 2, 1]]


def test_domestic_system_forces_python_engine_step_even_when_rust_requested() -> None:
    database = SimpleNamespace(system="_dom_test", system_spec=_spec(domestic=True))
    runtime = RuntimeConfig(engine_step_backend="rust")
    assert not should_use_rust_engine_step(runtime, database)
