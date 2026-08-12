from __future__ import annotations

import unittest
import warnings
from typing import ClassVar

from aiconfigurator_core.sdk.kernelsim.dsa import (
    LIMITED_SCOPE_MESSAGE,
    DsaIndexModelWarning,
    IndexMqaShape,
    IndexTopKShape,
    estimate_index_mqa,
    estimate_index_topk,
    get_index_mqa_parameters,
    get_index_topk_parameters,
)


class DsaIndexKernelSimTests(unittest.TestCase):
    hardware: ClassVar[dict[str, float | int]] = {
        "sm_count": 132,
        "clock_hz": 1.83e9,
        "fp8_peak_flops_s": 1.978e15,
        "hbm_bandwidth_bytes_s": 3.35e12,
    }

    def mqa(self, shape, level="standard"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DsaIndexModelWarning)
            return estimate_index_mqa(shape, parameter_level=level, **self.hardware)

    def topk(self, shape, level="standard"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DsaIndexModelWarning)
            return estimate_index_topk(
                shape,
                hbm_bandwidth_bytes_s=self.hardware["hbm_bandwidth_bytes_s"],
                parameter_level=level,
            )

    def test_profiles_and_default(self) -> None:
        mqa = get_index_mqa_parameters()
        topk = get_index_topk_parameters()
        self.assertAlmostEqual(mqa.floor_paged_us, 3.6963528879)
        self.assertAlmostEqual(mqa.task_cycles_ragged, 414.7553222539)
        self.assertAlmostEqual(topk.natural_short_floor_paged_us, 3.2231622758)
        self.assertEqual(topk.natural_short_query_tile, 256)
        self.assertEqual(topk.natural_long_query_tile, 128)
        self.assertEqual(topk.tail_threshold, 32768)
        with self.assertRaisesRegex(ValueError, "standard, high, or low"):
            get_index_mqa_parameters("precise")

    def test_ragged_geometry_and_work_snapshot(self) -> None:
        shape = IndexMqaShape("ragged", 1, 16, 4096, 32)
        result = self.mqa(shape)
        self.assertEqual(result.valid_pairs, sum(range(4096 - 16 + 1, 4096 + 1)))
        self.assertEqual(result.chunk_rows, 16)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.critical_kv_blocks, 64)
        self.assertEqual(result.flops, result.valid_pairs * 2 * 32 * 128)
        self.assertGreater(result.modeled_bytes, 0)
        self.assertAlmostEqual(
            result.latency_us,
            result.floor_us + result.control_us,
        )

    def test_ragged_production_chunking(self) -> None:
        shape = IndexMqaShape("ragged", 16, 256, 65536, 32)
        result = self.mqa(shape)
        self.assertEqual(result.chunk_rows, 16)
        self.assertEqual(result.chunk_count, 256)
        self.assertGreater(result.critical_kv_blocks, 0)
        self.assertGreater(result.control_us, result.resource_us)

    def test_paged_shape_and_score_alignment(self) -> None:
        shape = IndexMqaShape("paged", 8, 2, 4097, 64)
        result = self.mqa(shape)
        self.assertEqual(result.valid_pairs, 8 * (2 * 4097 + 1))
        self.assertEqual(result.score_slots, 8 * 2 * 4160)
        self.assertEqual(result.chunk_count, 1)
        self.assertGreater(result.control_us, 0)
        self.assertEqual(result.roofline_branch, "task_service_layout")

    def test_mqa_validation_and_scope_warning(self) -> None:
        with self.assertRaisesRegex(ValueError, "next_n=1 or 2"):
            IndexMqaShape("paged", 1, 4, 4096, 64)
        with self.assertRaisesRegex(ValueError, "only the calibrated FP8"):
            IndexMqaShape("ragged", 1, 1, 4096, 64, dtype="bf16")
        shape = IndexMqaShape("paged", 1, 1, 4096, 48)
        with self.assertWarns(DsaIndexModelWarning):
            result = estimate_index_mqa(shape, **self.hardware)
        self.assertIn(LIMITED_SCOPE_MESSAGE, result.warnings)
        self.assertTrue(any("32/64" in message for message in result.warnings))

    def test_mqa_three_levels_are_monotonic(self) -> None:
        shapes = (
            IndexMqaShape("paged", 1, 1, 2048, 32),
            IndexMqaShape("paged", 128, 2, 65536, 64),
            IndexMqaShape("ragged", 1, 16, 4096, 32),
            IndexMqaShape("ragged", 16, 256, 65536, 64),
        )
        for shape in shapes:
            values = {level: self.mqa(shape, level).latency_us for level in ("low", "standard", "high")}
            self.assertLess(values["low"], values["standard"])
            self.assertLess(values["standard"], values["high"])

    def test_topk_work_and_standard_distribution(self) -> None:
        shape = IndexTopKShape("ragged", 2, 16, 4096, variant="fused")
        result = self.topk(shape)
        self.assertEqual(result.variant, "ragged_fused")
        self.assertEqual(result.query_rows, 32)
        self.assertEqual(result.score_slots, 32 * 2 * 4096)
        self.assertEqual(result.output_indices, 32 * 2048)
        self.assertEqual(result.score_distribution, "standard")
        self.assertEqual(result.latency_us, result.floor_us + result.scan_us)
        self.assertTrue(result.long_path)

    def test_topk_three_levels_are_monotonic(self) -> None:
        shapes = (
            IndexTopKShape("paged", 1, 1, 2048, variant="plain", score_distribution="top_last"),
            IndexTopKShape("paged", 512, 2, 65536, variant="fused", score_distribution="flat"),
            IndexTopKShape("ragged", 1, 1, 4096, variant="plain"),
            IndexTopKShape("ragged", 16, 256, 65536, variant="fused"),
        )
        for shape in shapes:
            values = {level: self.topk(shape, level).latency_us for level in ("low", "standard", "high")}
            self.assertLess(values["low"], values["standard"])
            self.assertLess(values["standard"], values["high"])

    def test_topk_validation_and_warning(self) -> None:
        with self.assertRaisesRegex(ValueError, "fused variant must match"):
            IndexTopKShape("paged", 1, 1, 4096, variant="ragged_fused")
        shape = IndexTopKShape("paged", 1, 1, 4096)
        with self.assertWarns(DsaIndexModelWarning):
            result = estimate_index_topk(shape, hbm_bandwidth_bytes_s=3.35e12)
        self.assertIn(LIMITED_SCOPE_MESSAGE, result.warnings)

    def test_topk_threshold_is_a_piecewise_boundary(self) -> None:
        short = self.topk(IndexTopKShape("paged", 1, 1, 2048, topk=2048))
        long = self.topk(IndexTopKShape("paged", 1, 1, 2049, topk=2048))
        self.assertFalse(short.long_path)
        self.assertTrue(long.long_path)
        self.assertGreater(long.latency_us, short.latency_us)

        changed = self.topk(IndexTopKShape("paged", 1, 1, 2048, topk=1024))
        self.assertTrue(any("calibrated K=2048" in message for message in changed.warnings))

    def test_standard_topk_uses_natural_recipe_without_row_borrowing(self) -> None:
        standard = self.topk(IndexTopKShape("paged", 1, 1, 65536))
        natural = self.topk(IndexTopKShape("paged", 1, 1, 65536, score_distribution="natural"))
        self.assertAlmostEqual(standard.latency_us, natural.latency_us)
        self.assertEqual(natural.row_penalty_us, 0.0)
        self.assertEqual(natural.model_recipe, "natural_wave_v3")

    def test_natural_topk_short_and_long_wave_geometry(self) -> None:
        short = self.topk(IndexTopKShape("ragged", 1, 512, 2048))
        self.assertEqual(short.query_tile, 256)
        self.assertEqual(short.waves, 2)
        self.assertEqual(short.tail_waves, 0)
        self.assertFalse(short.long_path)

        long = self.topk(IndexTopKShape("ragged", 1, 1024, 65536))
        self.assertEqual(long.query_tile, 128)
        self.assertEqual(long.waves, 8)
        self.assertEqual(long.tail_waves, 4)
        self.assertTrue(long.long_path)
        self.assertGreater(long.tail_us, 0.0)
        self.assertAlmostEqual(long.latency_us, 349.1969674, places=5)

    def test_natural_topk_body_scales_with_hbm_bandwidth(self) -> None:
        shape = IndexTopKShape("ragged", 1, 512, 32768)
        h100 = self.topk(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DsaIndexModelWarning)
            double_bw = estimate_index_topk(shape, hbm_bandwidth_bytes_s=6.7e12)
        self.assertAlmostEqual(double_bw.floor_us, h100.floor_us)
        self.assertAlmostEqual(double_bw.scan_us * 2, h100.scan_us)

    def test_natural_topk_extrapolation_warnings(self) -> None:
        result = self.topk(IndexTopKShape("ragged", 1, 16384, 1_048_576))
        self.assertTrue(any("524288-token" in message for message in result.warnings))
        self.assertTrue(any("8192-row" in message for message in result.warnings))

    def test_diagnostic_topk_keeps_v2_recipe(self) -> None:
        result = self.topk(IndexTopKShape("ragged", 1, 512, 65536, score_distribution="flat"))
        self.assertEqual(result.model_recipe, "v2_kernel_regime")
        self.assertEqual(result.query_tile, 0)


if __name__ == "__main__":
    unittest.main()
