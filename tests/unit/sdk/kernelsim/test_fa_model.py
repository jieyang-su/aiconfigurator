from __future__ import annotations

import unittest

from aiconfigurator_core.sdk.kernelsim.fa import (
    AttentionShape,
    HardwareSpec,
    ModelOptions,
    estimate_attention,
    get_reference_profile,
)


class FlashAttentionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a100 = HardwareSpec(108, 1.41e9, 196608, 41943040, 7.2192e12, 2.039e12, {"bf16": 312e12}, 77.96736e12)
        cls.h100 = HardwareSpec(
            132, 1.83e9, 262144, 52428800, 9.3696e12, 3.35e12, {"bf16": 989e12, "fp8": 1.979e15}, 123.7248e12
        )

    def test_shape_rejects_invalid_gqa(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            AttentionShape(1, 1, 128, 30, 8, 128, "bf16")

    def test_fp8_defaults_to_bf16_output(self) -> None:
        shape = AttentionShape(1, 1, 128, 32, 8, 128, "float8")
        self.assertEqual(shape.dtype, "fp8")
        self.assertEqual(shape.output_dtype, "bf16")
        self.assertEqual(AttentionShape.from_dict(shape.to_dict()), shape)

    def test_h100_decode_split_and_task_count(self) -> None:
        shape = AttentionShape(1, 1, 4096, 32, 8, 128, "bf16")
        result = estimate_attention(self.h100, shape)
        self.assertEqual(result.algorithm, "fa2")
        self.assertEqual(result.mode, "profiled")
        self.assertEqual(result.reference_profile["level"], "standard")
        self.assertEqual(result.tiles["br"], 128)
        self.assertEqual(result.tiles["bc"], 128)
        self.assertEqual(result.tiles["kv_splits"], 5)
        self.assertEqual(result.scheduling["base_cta_count_hq_mapping"], 32)
        self.assertEqual(result.scheduling["cta_count_hq_mapping_with_splits"], 160)
        self.assertEqual(result.scheduling["n_task_b_hkv_query_tiles_splits"], 40)
        self.assertAlmostEqual(result.scheduling["fractional_task_waves"], 40 / 132, places=15)

    def test_large_batch_short_decode_exposes_service_cost(self) -> None:
        shape = AttentionShape(512, 1, 8, 32, 8, 128, "fp8")
        raw = estimate_attention(self.h100, shape, ModelOptions(mode="analytical"))
        profiled = estimate_attention(self.h100, shape, ModelOptions(mode="profiled"))
        self.assertEqual(profiled.tiles["kv_splits"], 1)
        self.assertEqual(profiled.scheduling["n_task_b_hkv_query_tiles_splits"], 4096)
        self.assertGreater(profiled.resources_us["task_service_us"], 100)
        self.assertGreater(profiled.latency_us, raw.latency_us * 10)

    def test_fa2_snapshot_matches_llmcompass_neutral_semantics(self) -> None:
        shape = AttentionShape(1, 1, 4096, 32, 8, 128, "bf16")
        options = ModelOptions(
            algorithm="fa2",
            mode="analytical",
            include_kv_cache_update=False,
            assume_gqa_hbm_reuse=True,
            assume_gqa_l2_reuse=False,
        )
        result = estimate_attention(self.a100, shape, options)
        self.assertEqual(result.work["total_hbm_bytes"], 16950272)
        self.assertEqual(result.work["total_l2_requested_bytes"], 67281920)
        self.assertEqual(result.work["matrix_flops"], 67108864)
        self.assertEqual(result.work["vector_equivalent_flops"], 5227968)
        self.assertAlmostEqual(result.latency_us, 9.344281914893616, places=12)

    def test_fa3_snapshot_matches_llmcompass_neutral_semantics(self) -> None:
        shape = AttentionShape(1, 1, 4096, 32, 8, 128, "bf16")
        options = ModelOptions(
            algorithm="fa3",
            mode="analytical",
            include_kv_cache_update=False,
            assume_gqa_hbm_reuse=True,
            assume_gqa_l2_reuse=False,
        )
        result = estimate_attention(self.h100, shape, options)
        self.assertEqual(result.tiles["kv_splits"], 5)
        self.assertEqual(result.work["total_hbm_bytes"], 16991488)
        self.assertEqual(result.work["total_l2_requested_bytes"], 67323136)
        self.assertEqual(result.work["matrix_flops"], 67108864)
        self.assertEqual(result.work["vector_equivalent_flops"], 5246752)
        self.assertAlmostEqual(result.latency_us, 7.215541325136612, places=12)

    def test_fa3_overlap_never_increases_compute_time(self) -> None:
        shape = AttentionShape(2, 1024, 1024, 24, 8, 128, "fp8")
        fa2 = estimate_attention(self.h100, shape, ModelOptions(algorithm="fa2", mode="analytical"))
        fa3 = estimate_attention(self.h100, shape, ModelOptions(algorithm="fa3", mode="analytical"))
        self.assertLessEqual(
            fa3.resources_us["raw_compute_combined_us"],
            fa2.resources_us["raw_compute_combined_us"],
        )

    def test_gqa_reuse_reduces_requested_bytes(self) -> None:
        shape = AttentionShape(1, 1, 4096, 32, 8, 128, "bf16")
        reused = estimate_attention(self.h100, shape, ModelOptions(mode="analytical"))
        duplicated = estimate_attention(
            self.h100,
            shape,
            ModelOptions(
                mode="analytical",
                assume_gqa_hbm_reuse=False,
                assume_gqa_l2_reuse=False,
            ),
        )
        self.assertLess(reused.work["total_hbm_bytes"], duplicated.work["total_hbm_bytes"])
        self.assertLess(
            reused.work["total_l2_requested_bytes"],
            duplicated.work["total_l2_requested_bytes"],
        )

    def test_asymmetric_prefill_counts_qk_pv_and_storage_separately(self) -> None:
        shape = AttentionShape(
            1,
            1,
            1,
            2,
            2,
            192,
            "bf16",
            value_head_dim=128,
            kv_storage_dim=320,
        )
        result = estimate_attention(
            self.h100,
            shape,
            ModelOptions(
                algorithm="fa3",
                mode="analytical",
                br=128,
                bc=128,
                account_for_parallelism=False,
                include_kv_cache_update=False,
            ),
        )
        self.assertEqual(result.work["qk_flops"], 768)
        self.assertEqual(result.work["pv_flops"], 512)
        self.assertEqual(result.work["matrix_flops"], 1280)
        self.assertEqual(result.work["mainloop_hbm_bytes"], 2560)
        self.assertEqual(result.work["live_kv_cache_update_bytes"], 0)

    def test_mla_decode_shared_latent_kv_is_stored_once(self) -> None:
        shape = AttentionShape(
            1,
            1,
            128,
            8,
            1,
            576,
            "bf16",
            value_head_dim=512,
            kv_storage_dim=576,
        )
        result = estimate_attention(
            self.h100,
            shape,
            ModelOptions(
                algorithm="fa3",
                mode="analytical",
                decode_splits=1,
                account_for_parallelism=False,
            ),
        )
        self.assertEqual(result.work["qk_flops"], 1_179_648)
        self.assertEqual(result.work["pv_flops"], 1_048_576)
        self.assertEqual(result.work["mainloop_hbm_bytes"], 164_864)
        self.assertEqual(result.work["live_kv_cache_update_bytes"], 2_304)
        self.assertEqual(result.scheduling["n_task_b_hkv_query_tiles_splits"], 1)

    def test_query_tile_l2_reuse_only_reduces_hbm_traffic(self) -> None:
        shape = AttentionShape(1, 256, 256, 1, 1, 128, "bf16")
        base_options = dict(
            algorithm="fa3",
            mode="analytical",
            br=128,
            bc=128,
            account_for_parallelism=False,
            include_kv_cache_update=False,
        )
        repeated = estimate_attention(
            self.h100,
            shape,
            ModelOptions(**base_options),
        )
        reused = estimate_attention(
            self.h100,
            shape,
            ModelOptions(**base_options, assume_query_tile_l2_reuse=True),
        )
        self.assertEqual(
            repeated.work["total_l2_requested_bytes"],
            reused.work["total_l2_requested_bytes"],
        )
        self.assertEqual(
            repeated.work["total_hbm_bytes"] - reused.work["total_hbm_bytes"],
            65_536,
        )

    def test_three_generic_levels_are_monotonic(self) -> None:
        shapes = (
            AttentionShape(2, 1024, 1024, 24, 8, 128, "bf16"),
            AttentionShape(32, 1, 4096, 32, 8, 128, "bf16"),
        )
        for algorithm in ("fa2", "fa3"):
            for shape in shapes:
                predictions = {
                    level: estimate_attention(
                        self.h100,
                        shape,
                        ModelOptions(algorithm=algorithm, estimate_level=level),
                    ).latency_us
                    for level in ("low", "standard", "high")
                }
                self.assertLessEqual(predictions["low"], predictions["standard"])
                self.assertLessEqual(predictions["standard"], predictions["high"])

    def test_generic_profile_values_are_hardware_agnostic(self) -> None:
        standard = get_reference_profile("standard")
        self.assertEqual(standard.level, "standard")
        self.assertEqual(standard.fixed_overhead_us, 12.5)
        self.assertEqual(standard.kv_task_cycles, 6000.0)
        self.assertNotIn("source", standard.to_dict())
        self.assertNotIn("architecture", standard.to_dict())

    def test_hardware_identity_fields_are_rejected(self) -> None:
        payload = self.h100.to_dict()
        payload["architecture_family"] = "hopper"
        with self.assertRaisesRegex(ValueError, "unknown hardware fields"):
            HardwareSpec.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
