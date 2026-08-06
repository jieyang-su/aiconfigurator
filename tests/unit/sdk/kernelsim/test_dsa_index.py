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
        self.assertEqual(get_index_mqa_parameters().eta_compute, 0.85)
        self.assertEqual(get_index_topk_parameters().floor_scale, 1.0)
        with self.assertRaisesRegex(ValueError, "standard, high, or low"):
            get_index_mqa_parameters("precise")

    def test_ragged_geometry_and_work_snapshot(self) -> None:
        shape = IndexMqaShape("ragged", 1, 16, 4096, 32)
        result = self.mqa(shape)
        self.assertEqual(result.valid_pairs, sum(range(4096 - 16 + 1, 4096 + 1)))
        self.assertEqual(result.chunk_rows, 16)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.critical_kv_blocks, 64)
        self.assertEqual(result.flops, 64 * 2 * 16 * 64 * 32 * 128)
        self.assertEqual(result.modeled_bytes, 64 * (64 * (128 + 4) + 16 * 64 * 4))
        self.assertAlmostEqual(
            result.latency_us,
            result.floor_us + result.resource_us + result.control_us,
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
        self.assertEqual(result.control_us, 0)

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


if __name__ == "__main__":
    unittest.main()
