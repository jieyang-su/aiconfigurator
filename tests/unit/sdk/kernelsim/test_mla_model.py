from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aiconfigurator_core.sdk.kernelsim.fa import (
    FP8_UNSUPPORTED_MESSAGE,
    HardwareSpec,
    MlaModelOptions,
    MlaRequest,
    all_mla_profiles,
    estimate_mla,
    get_mla_reference_profile,
)


class MlaModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a100 = HardwareSpec(108, 1.41e9, 196608, 41943040, 7.2192e12, 2.039e12, {"bf16": 312e12}, 77.96736e12)
        cls.h100 = HardwareSpec(
            132, 1.83e9, 262144, 52428800, 9.3696e12, 3.35e12, {"bf16": 989e12, "fp8": 1.979e15}, 123.7248e12
        )

    def test_prefill_and_decode_geometry(self) -> None:
        prefill = estimate_mla(
            self.h100,
            MlaRequest("prefill", 2, 16, 1024),
            MlaModelOptions(algorithm="fa3"),
        )
        self.assertEqual(prefill.geometry["head_dim"], 192)
        self.assertEqual(prefill.geometry["value_head_dim"], 128)
        self.assertEqual(prefill.geometry["kv_storage_dim"], 320)
        self.assertEqual(prefill.geometry["kv_heads"], 16)
        self.assertFalse(prefill.geometry["include_kv_cache_update"])

        decode = estimate_mla(
            self.h100,
            MlaRequest("decode", 2, 16, 4096),
            MlaModelOptions(algorithm="fa3"),
        )
        self.assertEqual(decode.geometry["head_dim"], 576)
        self.assertEqual(decode.geometry["value_head_dim"], 512)
        self.assertEqual(decode.geometry["kv_storage_dim"], 576)
        self.assertEqual(decode.geometry["kv_heads"], 1)
        self.assertTrue(decode.geometry["include_kv_cache_update"])
        self.assertEqual(decode.tiles["kv_splits"], 5)
        self.assertEqual(decode.scheduling["n_task_b_hkv_query_tiles_splits"], 10)

    def test_prefill_rounded_wave(self) -> None:
        result = estimate_mla(
            self.h100,
            MlaRequest("prefill", 1, 16, 2048),
            MlaModelOptions(algorithm="fa3"),
        )
        self.assertEqual(result.scheduling["cta_count_hq_mapping_with_splits"], 256)
        self.assertAlmostEqual(result.resource["cta_waves"], 256 / 132, places=14)
        self.assertAlmostEqual(result.resource["wave_rounding_factor"], 1.03125, places=14)
        self.assertAlmostEqual(
            result.components_us["selected_resource_before_efficiency"],
            result.components_us["raw_resource"] * 1.03125,
            places=12,
        )

    def test_prefix_prefill_uses_independent_query_and_kv_lengths(self) -> None:
        query_length = 7168
        kv_length = 8192
        result = estimate_mla(
            self.h100,
            MlaRequest(
                "prefill",
                4,
                32,
                kv_length,
                query_length=query_length,
            ),
            MlaModelOptions(algorithm="fa3"),
        )
        expected_scores_per_head = query_length * (kv_length - query_length) + query_length * (query_length + 1) // 2
        self.assertEqual(result.geometry["query_length"], query_length)
        self.assertEqual(result.geometry["kv_length_total"], kv_length)
        self.assertEqual(
            result.work["exact_valid_score_elements"],
            4 * 32 * expected_scores_per_head,
        )
        self.assertGreater(result.latency_us, 0)

    def test_query_length_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be <= sequence_length"):
            MlaRequest("prefill", 1, 16, 1024, query_length=1025)
        with self.assertRaisesRegex(ValueError, "decode query_length must be 1"):
            MlaRequest("decode", 1, 16, 1024, query_length=2)

    def test_three_bf16_levels_are_monotonic(self) -> None:
        requests = (
            MlaRequest("prefill", 1, 16, 128),
            MlaRequest("prefill", 8, 32, 4096),
            MlaRequest("decode", 1, 8, 256),
            MlaRequest("decode", 64, 32, 32768),
        )
        for hardware in (self.a100, self.h100):
            for algorithm in ("fa2", "fa3"):
                for request in requests:
                    values = {
                        level: estimate_mla(
                            hardware,
                            request,
                            MlaModelOptions(algorithm=algorithm, estimate_level=level),
                        ).latency_us
                        for level in ("low", "standard", "high")
                    }
                    self.assertLessEqual(values["low"], values["standard"])
                    self.assertLessEqual(values["standard"], values["high"])

    def test_generic_profiles_are_hardware_agnostic(self) -> None:
        self.assertEqual([profile.level for profile in all_mla_profiles()], ["low", "standard", "high"])
        standard = get_mla_reference_profile("standard")
        self.assertEqual(standard.prefill_bf16.resource_efficiency, 0.70)
        self.assertEqual(standard.decode_bf16.kv_task_cycles, 7000.0)
        serialized = json.dumps(standard.to_dict())
        for identity in ("H100", "NVIDIA", "SGLang", "architecture"):
            self.assertNotIn(identity, serialized)

    def test_fp8_is_rejected_with_compatibility_error(self) -> None:
        for dtype in ("fp8", "float8"):
            with self.assertRaisesRegex(ValueError, "not supported by the production model") as caught:
                MlaRequest("prefill", 1, 16, 1024, dtype)
            self.assertEqual(str(caught.exception), FP8_UNSUPPORTED_MESSAGE)

    def test_request_json_is_strict_and_bf16_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            path.write_text(
                json.dumps(
                    {
                        "requests": [
                            {
                                "phase": "prefill",
                                "batch_size": 1,
                                "local_query_heads": 8,
                                "sequence_length": 4096,
                                "query_length": 3072,
                                "dtype": "bfloat16",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            requests = MlaRequest.many_from_json(path)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].dtype, "bf16")
            self.assertEqual(requests[0].effective_query_length, 3072)


if __name__ == "__main__":
    unittest.main()
