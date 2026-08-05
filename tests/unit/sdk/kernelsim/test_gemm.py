from __future__ import annotations

import unittest

from aiconfigurator_core.sdk.kernelsim.gemm import model


class GemmEmpiricalFinalTests(unittest.TestCase):
    def test_precise_parameters_match_final_refits(self) -> None:
        bf16 = model.get_bf16_parameters("precise")
        self.assertEqual(bf16.t_launch_us, 2.0777777827803106)
        self.assertEqual(bf16.eta_mem, 0.7682798890205326)
        self.assertEqual(bf16.eta_compute, 0.8953233499328733)

        deepgemm = model.get_deepgemm_parameters("precise")
        self.assertEqual(deepgemm.t_floor_hopper_us, 3.8656665322681274)
        self.assertEqual(deepgemm.t_floor_blackwell_us, 10.378777877324156)
        self.assertEqual(deepgemm.eta_mem_hopper, 0.8581147783419518)
        self.assertEqual(deepgemm.eta_mem_blackwell, 0.48039321452379946)
        self.assertEqual(deepgemm.eta_compute, 0.6486839517063081)
        self.assertEqual(deepgemm.eta_quant_hopper, 0.6765141735130472)
        self.assertEqual(deepgemm.eta_quant_blackwell, 0.3368848516246155)
        self.assertEqual(deepgemm.rho_transition, 2.1264741995206062)

        sglang = model.get_sglang_fp8_parameters("precise")
        self.assertEqual(sglang.t_floor_us, 4.985881534715493)
        self.assertEqual(sglang.eta_mem, 0.7011993788391371)
        self.assertEqual(sglang.eta_compute, 0.6165923972877984)
        self.assertEqual(sglang.eta_quant, 0.4626841110323739)
        self.assertEqual(sglang.rho_transition, 1.5129826018401102)

    def test_default_standard_preserves_v1_outputs(self) -> None:
        self.assertIs(model.BF16_SUM_3P, model.BF16_SUM_3P_STANDARD)
        self.assertIs(model.DEEPGEMM_MAX_8P, model.DEEPGEMM_MAX_8P_STANDARD)
        self.assertIs(model.SGLANG_FP8_MAX_5P, model.SGLANG_FP8_MAX_5P_STANDARD)
        self.assertEqual(
            model.estimate_bf16_gemm(4096, 4096, 4096, 989e12, 3.35e12).latency_us,
            195.43278540768858,
        )
        self.assertEqual(
            model.estimate_deepgemm_fp8(4096, 4096, 4096, 1.979e15, 3.35e12, "hopper").latency_us,
            141.31664252283954,
        )
        self.assertEqual(
            model.estimate_deepgemm_fp8(4096, 4096, 4096, 4.5e15, 8e12, "blackwell").latency_us,
            93.91004300283592,
        )
        self.assertEqual(
            model.estimate_sglang_fp8(4096, 4096, 4096, 1.979e15, 3.35e12).latency_us,
            156.93580043573968,
        )

    def test_bf16_engineering_levels_are_monotonic(self) -> None:
        shapes = (
            (1, 128, 128),
            (33, 3072, 16384),
            (256, 7168, 16384),
            (4096, 4096, 4096),
            (131072, 512, 512),
        )
        for shape in shapes:
            predictions = {
                level: model.bf16_gemm_latency_us(
                    *shape,
                    peak_bf16_flops=989e12,
                    mem_bandwidth_bytes_s=3.35e12,
                    parameter_level=level,
                )
                for level in ("low", "standard", "high")
            }
            self.assertLessEqual(predictions["low"], predictions["standard"])
            self.assertLessEqual(predictions["standard"], predictions["high"])

    def test_deepgemm_engineering_levels_are_monotonic(self) -> None:
        shapes = (
            (1, 128, 128),
            (8, 2112, 7168),
            (33, 3072, 16384),
            (256, 7168, 16384),
            (4096, 4096, 4096),
        )
        hardware = {
            "hopper": (1.979e15, 3.35e12),
            "blackwell": (4.5e15, 8.0e12),
        }
        for architecture, (peak, bandwidth) in hardware.items():
            for shape in shapes:
                predictions = {
                    level: model.deepgemm_fp8_latency_us(
                        *shape,
                        peak_fp8_flops=peak,
                        mem_bandwidth_bytes_s=bandwidth,
                        architecture=architecture,
                        parameter_level=level,
                    )
                    for level in ("low", "standard", "high")
                }
                self.assertLessEqual(predictions["low"], predictions["standard"])
                self.assertLessEqual(predictions["standard"], predictions["high"])

    def test_sglang_engineering_levels_are_monotonic(self) -> None:
        shapes = (
            (1, 64, 128),
            (33, 3072, 16384),
            (257, 7168, 16384),
            (4096, 4096, 4096),
            (131072, 512, 512),
        )
        for shape in shapes:
            predictions = {
                level: model.sglang_fp8_latency_us(
                    *shape,
                    peak_fp8_flops=1.979e15,
                    mem_bandwidth_bytes_s=3.35e12,
                    parameter_level=level,
                )
                for level in ("low", "standard", "high")
            }
            self.assertLessEqual(predictions["low"], predictions["standard"])
            self.assertLessEqual(predictions["standard"], predictions["high"])

    def test_custom_parameters_remain_supported(self) -> None:
        custom = model.Bf16Sum3PParameters(
            t_launch_us=7.0,
            eta_mem=0.5,
            eta_compute=0.6,
        )
        result = model.estimate_bf16_gemm(64, 128, 256, 312e12, 2e12, custom)
        self.assertEqual(result.launch_us, 7.0)

        with self.assertRaisesRegex(ValueError, "custom params"):
            model.estimate_bf16_gemm(
                64,
                128,
                256,
                312e12,
                2e12,
                custom,
                parameter_level="high",
            )

    def test_level_and_deepgemm_domain_validation(self) -> None:
        self.assertIs(
            model.get_bf16_parameters(" STANDARD "),
            model.BF16_SUM_3P_STANDARD,
        )
        with self.assertRaisesRegex(ValueError, "parameter_level"):
            model.get_sglang_fp8_parameters("median")
        with self.assertRaisesRegex(ValueError, "n and k >= 128"):
            model.estimate_deepgemm_fp8(64, 127, 256, 1e15, 3e12, "hopper")
        with self.assertRaisesRegex(ValueError, "hopper.*blackwell"):
            model.estimate_deepgemm_fp8(64, 128, 256, 1e15, 3e12, "ampere")

    def test_deepgemm_accepts_ragged_n_and_uses_ceil_scale_groups(self) -> None:
        result = model.estimate_deepgemm_fp8(8, 2112, 7168, 1.979e15, 3.35e12, "hopper")
        scale_a_bytes = 4 * 8 * 56
        scale_b_bytes = 4 * 17 * 56
        expected_bytes = 8 * 7168 + 2112 * 7168 + 2 * 8 * 2112 + scale_a_bytes + scale_b_bytes
        self.assertEqual(result.flops, 2 * 8 * 2112 * 7168)
        self.assertEqual(result.gemm_bytes, expected_bytes)
        self.assertGreater(result.latency_us, 0)


if __name__ == "__main__":
    unittest.main()
