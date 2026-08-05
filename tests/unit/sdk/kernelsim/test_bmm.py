from __future__ import annotations

import inspect
import math
import unittest
import warnings
from dataclasses import fields

from aiconfigurator_core.sdk.kernelsim.bmm import model


class MlaBmmEmpiricalFinalTests(unittest.TestCase):
    def test_precise_parameters_match_final_refit(self) -> None:
        params = model.get_mla_bmm_parameters("precise")
        self.assertEqual(params.t_floor_bf16_us, 4.1978933693220215)
        self.assertEqual(params.t_floor_fp8_us, 7.136533359686535)
        self.assertEqual(params.eta_mem_bf16, 0.8810094265528273)
        self.assertEqual(params.eta_mem_fp8, 0.2194306605990924)
        self.assertEqual(params.eta_compute, 0.65)

    def test_bf16_engineering_eta_is_corrected_but_fp8_is_unchanged(self) -> None:
        self.assertEqual(model.BMM_LOW.eta_mem_bf16, 0.80)
        self.assertEqual(model.BMM_STANDARD.eta_mem_bf16, 0.70)
        self.assertEqual(model.BMM_HIGH.eta_mem_bf16, 0.56)
        self.assertEqual(model.BMM_LOW.eta_mem_fp8, 0.28)
        self.assertEqual(model.BMM_STANDARD.eta_mem_fp8, 0.22)
        self.assertEqual(model.BMM_HIGH.eta_mem_fp8, 0.18)

    def test_default_is_standard_and_snapshot_is_stable(self) -> None:
        self.assertIs(model.MLA_BMM_FULL_ROOFLINE, model.MLA_BMM_STANDARD)
        with self.assertWarns(model.BmmFp8ReliabilityWarning):
            result = model.estimate_mla_bmm(
                512,
                128,
                "pre",
                "fp8",
                peak_flops_s=1.979e15,
                mem_bandwidth_bytes_s=3.35e12,
            )
        expected_flops = 2 * 128 * 512 * 512 * 128
        expected_bytes = 4 * 128 * 512 * 128 + 2 * 128 * 512 * 512 + 8 * 128 + 128 * 128 * 512 + 4 * 128 * 4
        expected_us = (
            7.1
            + max(
                expected_flops / 1.979e15 / 0.65,
                expected_bytes / 3.35e12 / 0.22,
            )
            * 1e6
        )
        self.assertEqual(result.flops, expected_flops)
        self.assertEqual(result.logical_bytes, expected_bytes)
        self.assertAlmostEqual(result.latency_us, expected_us, places=12)
        self.assertEqual(result.parameter_level, "standard")
        self.assertIn(model.FP8_WARNING_MESSAGE, result.warnings)

    def test_three_engineering_levels_are_monotonic(self) -> None:
        cases = (
            (1, 1),
            (8, 128),
            (512, 32),
            (8192, 128),
            (20480, 1),
        )
        hardware = (
            {"bf16": 989e12, "fp8": 1.979e15, "bw": 3.35e12},
            {"bf16": 2.25e15, "fp8": 4.5e15, "bw": 8.0e12},
        )
        for rates in hardware:
            for dtype in ("bf16", "fp8"):
                for op in ("pre", "post"):
                    for tokens, heads in cases:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", model.BmmFp8ReliabilityWarning)
                            values = {
                                level: model.mla_bmm_latency_us(
                                    tokens,
                                    heads,
                                    op,
                                    dtype,
                                    rates[dtype],
                                    rates["bw"],
                                    parameter_level=level,
                                )
                                for level in ("low", "standard", "high")
                            }
                        self.assertLessEqual(values["low"], values["standard"])
                        self.assertLessEqual(values["standard"], values["high"])

    def test_pre_and_post_share_parameters(self) -> None:
        for dtype in ("bf16", "fp8"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", model.BmmFp8ReliabilityWarning)
                pre = model.estimate_mla_bmm(128, 32, "pre", dtype, 2e15, 4e12)
                post = model.estimate_mla_bmm(128, 32, "post", dtype, 2e15, 4e12)
            self.assertEqual(pre.floor_us, post.floor_us)
            self.assertEqual(pre.eta_memory, post.eta_memory)
            self.assertEqual(pre.eta_compute, post.eta_compute)
            self.assertEqual((pre.k, pre.n), (128, 512))
            self.assertEqual((post.k, post.n), (512, 128))

    def test_work_terms_match_collector_boundary(self) -> None:
        bf16 = model.work_terms(17, 8, "mla_gen_post", "bfloat16")
        self.assertEqual(bf16["flops"], 2 * 8 * 17 * 512 * 128)
        self.assertEqual(bf16["token_bytes"], 2 * 8 * 17 * (512 + 128))
        self.assertEqual(bf16["weight_bytes"], 2 * 8 * 512 * 128)
        self.assertEqual(bf16["quant_bytes"], 0)

        fp8 = model.work_terms(17, 8, "pre", "float8_e4m3fn")
        scale_a = 4 * 8
        scale_b = 4 * 8 * math.ceil(512 / 128) * math.ceil(128 / 128)
        self.assertEqual(fp8["quant_bytes"], 3 * 8 * 17 * 128 + scale_a)
        self.assertEqual(fp8["weight_bytes"], 8 * 128 * 512 + scale_b)
        self.assertEqual(fp8["logical_bytes"], fp8["token_bytes"] + fp8["weight_bytes"])

    def test_generic_bmm_accepts_arbitrary_geometry(self) -> None:
        terms = model.bmm_work_terms(3, 17, 257, 65, "bf16")
        self.assertEqual(terms["flops"], 2 * 3 * 17 * 257 * 65)
        self.assertEqual(terms["logical_bytes"], 2 * 3 * (17 * 65 + 65 * 257 + 17 * 257))
        self.assertEqual(terms["scope"], "generic_shape_extrapolation")
        result = model.estimate_bmm(3, 17, 257, 65, "bf16", 1e15, 3e12)
        self.assertEqual((result.batch, result.m, result.n, result.k), (3, 17, 257, 65))
        self.assertEqual(result.op, "generic")
        self.assertIn(model.GENERIC_SHAPE_WARNING, result.warnings)

    def test_generic_fp8_uses_ceil_scales_and_warns(self) -> None:
        batch, m, n, k = 3, 17, 257, 65
        terms = model.bmm_work_terms(batch, m, n, k, "fp8")
        scale_a = 4 * batch
        scale_b = 4 * batch * math.ceil(n / 128) * math.ceil(k / 128)
        self.assertEqual(terms["quant_bytes"], 3 * batch * m * k + scale_a)
        self.assertEqual(terms["weight_bytes"], batch * k * n + scale_b)
        with self.assertWarnsRegex(model.BmmFp8ReliabilityWarning, "low predictive confidence"):
            result = model.estimate_bmm(batch, m, n, k, "fp8", 2e15, 4e12)
        self.assertEqual(result.recipe, "sglang_quant_bmm_fp8")
        self.assertIn(model.FP8_WARNING_MESSAGE, result.warnings)

    def test_legacy_wrapper_matches_generic_interface(self) -> None:
        legacy = model.estimate_mla_bmm(31, 7, "post", "bf16", 2e15, 4e12)
        generic = model.estimate_bmm(7, 31, 128, 512, "bf16", 2e15, 4e12)
        self.assertEqual(legacy.latency_us, generic.latency_us)
        self.assertEqual(legacy.op, "post")
        self.assertEqual(legacy.scope, "calibrated_deepseek_matrix_geometry")

    def test_public_model_has_no_hardware_specific_parameters(self) -> None:
        parameter_names = {field.name for field in fields(model.BmmParameters)}
        result_names = {field.name for field in fields(model.BmmLatencyBreakdown)}
        signature_names = set(inspect.signature(model.estimate_bmm).parameters)
        forbidden = {"architecture", "generation", "device", "hopper", "blackwell"}
        for name in parameter_names | result_names | signature_names:
            self.assertTrue(forbidden.isdisjoint(name.lower().split("_")), name)

    def test_custom_parameters_and_validation(self) -> None:
        custom = model.MlaBmmParameters(t_floor_bf16_us=11.0)
        result = model.estimate_mla_bmm(1, 1, "pre", "bf16", 1e15, 3e12, custom)
        self.assertEqual(result.parameter_level, "custom")
        self.assertEqual(result.floor_us, 11.0)
        with self.assertRaisesRegex(ValueError, "custom params"):
            model.estimate_mla_bmm(
                1,
                1,
                "pre",
                "bf16",
                1e15,
                3e12,
                custom,
                parameter_level="high",
            )
        with self.assertRaisesRegex(ValueError, "parameter_level"):
            model.get_mla_bmm_parameters("median")
        with self.assertRaisesRegex(ValueError, "num_tokens"):
            model.work_terms(0, 1, "pre", "bf16")
        with self.assertRaisesRegex(ValueError, "peak_flops_s"):
            model.estimate_mla_bmm(1, 1, "pre", "bf16", 0, 3e12)


if __name__ == "__main__":
    unittest.main()
