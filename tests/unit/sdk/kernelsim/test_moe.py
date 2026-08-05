from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import fields

from aiconfigurator_core.sdk.kernelsim.moe import model


class SglangMoeEmpiricalFinalTests(unittest.TestCase):
    def test_precise_parameters_match_refit(self) -> None:
        params = model.get_moe_parameters("precise")
        self.assertEqual(params.bf16_triton.t_launch_us, 49.20565631866641)
        self.assertEqual(params.fp8_block_triton.eta_compute, 0.7260468956413237)
        self.assertEqual(params.nvfp4_cutedsl.eta_mem, 0.7578316517757806)

    def test_recipe_requires_direct_dtype_peak(self) -> None:
        self.assertEqual(model.required_peak_field("bf16_triton"), "bfloat16_tc_flops")
        self.assertEqual(model.required_peak_field("fp8_block_triton"), "fp8_tc_flops")
        self.assertEqual(model.required_peak_field("nvfp4_cutedsl"), "fp4_tc_flops")
        result = model.estimate_sglang_moe(
            "nvfp4_cutedsl",
            128,
            7168,
            2048,
            8,
            256,
            1,
            1,
            peak_flops_s=15e15,
            mem_bandwidth_bytes_s=8e12,
            parameter_level="precise",
        )
        self.assertAlmostEqual(
            result.compute_us,
            result.flops / 15e15 / model.MOE_PRECISE.nvfp4_cutedsl.eta_compute * 1e6,
            places=12,
        )

    def test_bf16_full_boundary_traffic(self) -> None:
        t, h, j, topk, experts = 8, 16, 32, 2, 4
        assignments = t * topk
        terms = model.work_terms("bf16_triton", t, h, j, topk, experts)
        expected = {
            "bytes_routing_logits": 6 * t * experts,
            "bytes_routing_topk": 32 * assignments,
            "bytes_gemm1_input": 2 * assignments * h,
            "bytes_gemm1_weight_values": 4 * experts * h * j,
            "bytes_gemm1_output": 4 * assignments * j,
            "bytes_activation_quant": 6 * assignments * j,
            "bytes_gemm2_input": 2 * assignments * j,
            "bytes_gemm2_weight_values": 2 * experts * h * j,
            "bytes_gemm2_output": 2 * assignments * h,
            "bytes_combine": 2 * assignments * h + 2 * t * h,
        }
        for key, value in expected.items():
            self.assertEqual(terms[key], value, key)
        self.assertEqual(terms["bytes_total"], sum(expected.values()))
        self.assertEqual(terms["flops_total"], 6 * assignments * h * j)

    def test_topk_one_has_no_separate_combine(self) -> None:
        terms = model.work_terms("bf16_triton", 8, 16, 32, 1, 4)
        self.assertEqual(terms["bytes_combine"], 0.0)

    def test_fp8_block_quant_and_scale_traffic(self) -> None:
        t, h, j, topk, experts = 3, 256, 384, 2, 4
        assignments = t * topk
        terms = model.work_terms("fp8_block_triton", t, h, j, topk, experts)
        self.assertEqual(terms["bytes_input_quant"], 3 * t * h + 4 * t * 2)
        self.assertEqual(terms["bytes_gemm1_input"], assignments * h + 4 * assignments * 2)
        self.assertEqual(terms["bytes_gemm1_weight_values"], 2 * experts * h * j)
        self.assertEqual(terms["bytes_gemm1_weight_scale"], 4 * experts * 6 * 2)
        self.assertEqual(terms["bytes_activation_quant"], 9 * assignments * j + 4 * assignments * 3)
        self.assertEqual(terms["bytes_gemm2_weight_scale"], 4 * experts * 2 * 3)

    def test_nvfp4_value_and_scale_bytes_are_separate(self) -> None:
        t, h, j, topk, experts = 8, 256, 384, 2, 4
        assignments = t * topk
        terms = model.work_terms("nvfp4_cutedsl", t, h, j, topk, experts)
        scale_bytes_w1_per_expert = math.ceil((2 * j) / 128) * 128 * math.ceil(math.ceil(h / 16) / 4) * 4
        scale_bytes_w2_per_expert = math.ceil(h / 128) * 128 * math.ceil(math.ceil(j / 16) / 4) * 4
        self.assertEqual(terms["bytes_gemm1_weight_values"], experts * h * j)
        self.assertEqual(terms["bytes_gemm2_weight_values"], 0.5 * experts * h * j)
        self.assertEqual(terms["bytes_gemm1_weight_scale"], experts * scale_bytes_w1_per_expert)
        self.assertEqual(terms["bytes_gemm2_weight_scale"], experts * scale_bytes_w2_per_expert)
        self.assertEqual(terms["bytes_input_quant"], 2.5 * assignments * h + assignments * 16)
        self.assertEqual(terms["bytes_control"], 16 * experts + 16 * experts)
        self.assertEqual(terms["bytes_routing_logits"], 0.0)
        self.assertEqual(terms["bytes_combine"], 0.0)

    def test_three_engineering_levels_are_monotonic(self) -> None:
        shapes = (
            (1, 2048, 768, 8, 128, 4, 1),
            (128, 4096, 14336, 2, 8, 8, 1),
            (20480, 7168, 2048, 8, 256, 32, 1),
            (1024, 7168, 2048, 8, 256, 1, 8),
        )
        rates = {
            "bf16_triton": (2.5e15, 8e12),
            "fp8_block_triton": (5e15, 8e12),
            "nvfp4_cutedsl": (15e15, 8e12),
        }
        for recipe, (peak, bandwidth) in rates.items():
            for shape in shapes:
                values = {
                    level: model.sglang_moe_latency_us(
                        recipe,
                        *shape,
                        peak,
                        bandwidth,
                        parameter_level=level,
                    )
                    for level in ("low", "standard", "high")
                }
                self.assertLessEqual(values["low"], values["standard"])
                self.assertLessEqual(values["standard"], values["high"])

    def test_ep_scope_and_ideal_work_scaling_are_explicit(self) -> None:
        ep1 = model.estimate_sglang_moe("bf16_triton", 1024, 7168, 2048, 8, 256, 1, 1, 989e12, 3.35e12)
        ep8 = model.estimate_sglang_moe("bf16_triton", 1024, 7168, 2048, 8, 256, 1, 8, 989e12, 3.35e12)
        self.assertEqual(ep1.scope, "measured EP=1")
        self.assertEqual(ep8.scope, "ideal uniform EP extrapolation")
        self.assertEqual(ep8.local_assignments, ep1.local_assignments / 8)

    def test_no_hardware_identity_or_bf16_multiplier_in_api(self) -> None:
        names = set(inspect.signature(model.estimate_sglang_moe).parameters)
        names |= {field.name for field in fields(model.MoeParameters)}
        forbidden = {"device", "architecture", "generation", "hopper", "blackwell", "compute_multiplier"}
        for name in names:
            self.assertTrue(forbidden.isdisjoint(name.lower().split("_")), name)
        self.assertIn("peak_flops_s", names)
        self.assertNotIn("peak_bf16_flops_s", names)

    def test_custom_parameters_and_validation(self) -> None:
        custom = model.MoeParameters(
            bf16_triton=model.RecipeParameters(10.0, 0.5, 0.5),
            fp8_block_triton=model.MOE_STANDARD.fp8_block_triton,
            nvfp4_cutedsl=model.MOE_STANDARD.nvfp4_cutedsl,
        )
        result = model.estimate_sglang_moe("bf16_triton", 8, 16, 32, 2, 4, 1, 1, 1e15, 3e12, custom)
        self.assertEqual(result.parameter_level, "custom")
        self.assertEqual(result.launch_us, 10.0)
        with self.assertRaisesRegex(ValueError, "custom params"):
            model.estimate_sglang_moe(
                "bf16_triton",
                8,
                16,
                32,
                2,
                4,
                1,
                1,
                1e15,
                3e12,
                custom,
                parameter_level="high",
            )
        with self.assertRaisesRegex(ValueError, "inter_size"):
            model.work_terms("bf16_triton", 8, 16, 31, 2, 4, 2, 1)
        with self.assertRaisesRegex(ValueError, "peak_flops_s"):
            model.estimate_sglang_moe("bf16_triton", 8, 16, 32, 2, 4, 1, 1, 0.0, 3e12)


if __name__ == "__main__":
    unittest.main()
