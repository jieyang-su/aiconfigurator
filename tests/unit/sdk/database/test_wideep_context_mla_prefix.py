from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.mla import (
    WideEPContextMLA,
    _WIDEEP_CONTEXT_MLA_PREFIX_DATA_KEY,
    load_wideep_context_mla_data,
)
from aiconfigurator.sdk.perf_database import PerformanceResult


def test_loader_keeps_prefix_shapes_separate_from_no_prefix_grid(tmp_path):
    path = tmp_path / "wideep_context_mla_perf.txt"
    path.write_text(
        "kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency\n"
        "fa3,fp8_block,fp8,16,1,3277,1,0,8.0\n"
        "fa3,fp8_block,fp8,16,1,3277,1,29491,9.7789\n"
    )

    data = load_wideep_context_mla_data(str(path))

    no_prefix = data["fa3"][common.FMHAQuantMode.fp8_block][
        common.KVCacheQuantMode.fp8
    ][16][3277][1]
    prefix = data[_WIDEEP_CONTEXT_MLA_PREFIX_DATA_KEY]["fa3"][
        common.FMHAQuantMode.fp8_block
    ][common.KVCacheQuantMode.fp8][16][29491][3277][1]
    assert no_prefix["latency"] == pytest.approx(8.0)
    assert prefix["latency"] == pytest.approx(9.7789)


def test_exact_prefix_shape_bypasses_no_prefix_area_scaling():
    prefix_data = {
        "fa3": {
            common.FMHAQuantMode.fp8_block: {
                common.KVCacheQuantMode.fp8: {
                    16: {29491: {3277: {1: {"latency": 9.7789, "energy": 0.0}}}}
                }
            }
        }
    }
    class PrefixWrapper(dict):
        loaded = True

        def raise_if_not_loaded(self):
            return None

    class PrefixData(dict):
        loaded = True

    wrapper = PrefixWrapper()
    wrapper.prefix_data = PrefixData(prefix_data)

    database = SimpleNamespace(
        _default_database_mode=common.DatabaseMode.SILICON,
        _wideep_context_mla_data=wrapper,
        _query_silicon_or_hybrid=lambda **kwargs: kwargs["get_silicon"](),
        _interp_pr=lambda latency, energy=0.0: PerformanceResult(
            latency,
            energy=energy,
            source="silicon",
        ),
    )

    with patch.object(WideEPContextMLA, "load_data", return_value=None):
        result = WideEPContextMLA._query_wideep_context_mla_table(
            database,
            b=1,
            s=3277,
            prefix=29491,
            tp_size=8,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8_block,
            attention_backend="fa3",
            database_mode=common.DatabaseMode.SILICON,
        )

    assert float(result) == pytest.approx(9.7789)
    assert result.source == "silicon"
