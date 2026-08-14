from aiconfigurator_core.sdk.perf_database import load_system_spec
from aiconfigurator_core.sdk.system_spec import (
    ParallelLayout,
    is_blackwell_spec,
    is_hopper_spec,
    supports_fp4_mma,
    supports_fp8_mma,
)


def test_h20_hardware_profile():
    spec = load_system_spec("h20_sxm")

    assert spec["gpu"]["mem_capacity"] == 96 * 1024**3
    assert spec["gpu"]["mem_bw"] == 4_000_000_000_000
    assert spec["gpu"]["bfloat16_tc_flops"] == 148_000_000_000_000
    assert spec["gpu"]["fp8_tc_flops"] == 296_000_000_000_000
    assert spec["node"]["num_gpus_per_node"] == 8
    assert is_hopper_spec(spec)
    assert not is_blackwell_spec(spec)
    assert supports_fp8_mma(spec)
    assert not supports_fp4_mma(spec)


def test_h20_tp_first_can_distinguish_node_locality():
    spec = load_system_spec("h20_sxm")
    layout = ParallelLayout(tp=8, pp=2, policy="tp_first")

    assert layout.bandwidth(spec, "tp", 8) == spec["node"]["intra_node_bw"]
    assert layout.bandwidth(spec, "pp", 2) == spec["node"]["inter_node_bw"]
