#!/usr/bin/env python3
"""Reusable recorded MoE latency materializer.

Modes covered here:
  - ordinary_context
  - ordinary_generation
  - wideep_context
  - wideep_generation
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["platform", "family", "phase", "ep", "eplb", "token"]
EPS = 1e-9


ORDINARY_COMMON_FILTER = {
    "op_name": "moe",
    "kernel_source": "sglang_fused_moe_triton",
    "moe_dtype": "fp8_block",
    "hidden_size": 7168,
    "inter_size": 2048,
    "topk": 8,
    "num_experts": 256,
    "moe_tp_size": 1,
}

WIDEEP_CONTEXT_COMMON_FILTER = {
    "op_name": "moe_context",
    "moe_dtype": "fp8_block",
    "hidden_size": 7168,
    "inter_size": 2048,
    "topk": 8,
    "num_experts": 256,
    "moe_tp_size": 1,
}

WIDEEP_GENERATION_COMMON_FILTER = {
    "op_name": "moe_generation",
    "moe_dtype": "fp8_block",
    "hidden_size": 7168,
    "inter_size": 2048,
    "topk": 8,
    "num_experts": 256,
    "moe_tp_size": 1,
}


OC_FEATURES = [
    "const",
    "log2_token",
    "log2_token_sq",
    "inv_sqrt_token",
    "log2_ep",
    "eplb_on",
    "logt_logep",
    "g8",
    "g32",
    "g64",
    "g512",
    "g4096",
    "g_tail",
    "g8_ep",
    "g32_ep",
    "g64_ep",
    "g512_ep",
    "g4096_ep",
    "g_tail_ep",
    "log_raw_latency_us",
    "log_rank_mean_us",
    "log_rank_max_us",
    "log_ratio_rankmax_mean",
    "log_rows_max",
    "log_masked_m_max",
    "log_active_experts_max",
    "log_assignments_max",
]

OG_FEATURES = [
    "const",
    "log2_token",
    "log2_token_sq",
    "inv_sqrt_token",
    "log2_ep",
    "eplb_on",
    "tiny",
    "small",
    "mid",
    "tail",
    "ep1",
    "ep2_or_less",
    "tiny_x_ep1",
    "small_x_ep1",
    "mid_x_ep2",
    "tail_x_ep1",
    "tail_x_ep2_or_less",
    "log2_token_x_tiny_x_ep1",
    "log2_token_sq_x_tiny_x_ep1",
    "log2_token_x_mid_x_ep2",
    "log2_token_sq_x_mid_x_ep2",
    "log2_token_x_tail_x_ep2_or_less",
    "log_raw_current",
    "log_ord_max_mean",
    "log1p_ord_spread_max",
]

OG_EP1_RECORDED_ONLY_FEATURES = [
    "const",
    "log2_token",
    "log2_token_sq",
    "inv_sqrt_token",
    "g32",
    "g64",
    "g128",
    "g512",
    "g4096",
    "g_tail",
    "log_raw_latency_us",
    "log_masked_over_rows",
    "log_rows_over_token",
]

OG_EPGE2_RECORDED_ONLY_FEATURES = [
    "const",
    "log2_token",
    "log2_token_sq",
    "inv_sqrt_token",
    "log2_ep",
    "eplb_on",
    "logt_logep",
    "g32",
    "g64",
    "g512",
    "g4096",
    "g_tail",
    "g32_ep",
    "g64_ep",
    "g512_ep",
    "g4096_ep",
    "g_tail_ep",
    "log_raw_latency_us",
    "log_rank_mean_us",
    "log_rank_max_us",
    "log_ratio_max_mean",
    "log_rows",
    "log_masked",
    "log_assign",
    "log_active",
    "log_rows_over_token",
    "log_masked_over_rows",
]

WC_FEATURES = [
    "const",
    "log2_token",
    "log2_token_sq",
    "inv_sqrt_token",
    "log2_ep",
    "eplb_on",
    "logt_logep",
    "g8",
    "g32",
    "g64",
    "g512",
    "g4096",
    "g_tail",
    "g8_ep",
    "g32_ep",
    "g64_ep",
    "g512_ep",
    "g4096_ep",
    "g_tail_ep",
    "log_raw_latency_us",
    "log_raw_latency_stage_sum_rankmax_us",
    "log_raw_origin_latency_us",
    "log_raw_rank_sync_tail_mean_us",
    "log_raw_rank_mean_latency_us",
    "log_raw_rank_p90_latency_us",
    "g8_log_sync_tail",
    "g8_log_stage",
    "log_stage_raw",
    "log_origin_raw",
    "log1p_sync_stage",
    "log_workload_rank_assignments_max",
    "log_workload_rank_imbalance_max_over_mean",
    "log_workload_expert_m_max",
    "log_workload_expected_m_max",
]


MODEL_SPECS = {
    "ordinary_context": {
        "active": True,
        "mode": "correction",
        "base": "raw_latency_us",
        "features": OC_FEATURES,
        "lambda": 0.01,
        "version": "moe_recorded_v4_ordinary_context_current_clean_small_endpoint_left_curve",
        "pending": "",
        "kernel_source": None,
    },
    "ordinary_generation": {
        "active": True,
        "mode": "ordinary_generation_recorded_only",
        "base": None,
        "features": [],
        "lambda": 0.0,
        "version": "moe_recorded_v4_ordinary_generation_scaled_source",
        "pending": "",
        "kernel_source": None,
    },
    "wideep_context": {
        "active": True,
        "mode": "direct",
        "base": None,
        "features": WC_FEATURES,
        "lambda": 0.01,
        "version": "moe_recorded_v4_wideep_context_current_clean_shape_endpoint",
        "pending": "",
        "kernel_source": "aic_recorded_wideep_context_v3",
    },
    "wideep_generation": {
        "active": True,
        "mode": "wideep_generation_practical",
        "base": None,
        "features": [],
        "lambda": 0.0,
        "version": "moe_recorded_v1_wideep_generation_nsys",
        "pending": "",
        "kernel_source": "aic_recorded_wideep_generation_v1",
    },
}


WIDEEP_GENERATION_COEFFICIENTS = {
    "capacity_256": {
        "coef": np.array([2.768979687, -0.2878345676, -4.748652345], dtype=float),
        "features": ["const", "log2_token", "tiny_decay"],
        "source": "blend_kernel_stage_g0.4",
        "note": "fixed_source_coef_platform_holdout_pass_max_19p11",
    },
    "capacity_512": {
        "coef": np.array([0.5791240778, -0.07309484336, -2.3023977, 0.1899448125], dtype=float),
        "features": ["const", "log2_token", "tiny_decay", "log_rank_over_stage"],
        "source": "raw_stage_kernel_sum_mean_us",
        "note": "fixed_source_coef_platform_holdout_pass_max_12p49",
    },
    "capacity_1024": {
        "coef": np.array(
            [
                -1.23553770754262,
                2.30135588051729,
                0.0146325760539454,
                -3.70661312262781,
                -0.00363238046908589,
                0.0438977281618966,
                0.5313698327863,
                1.5941094983589,
                -0.645759758139618,
                -0.504381515560275,
            ],
            dtype=float,
        ),
        "features": [
            "const",
            "log_source",
            "log2_token",
            "log2_ep",
            "eplb_on",
            "logt_logep",
            "tiny_decay",
            "tiny_by_logep",
            "log_rank_over_stage",
            "log_sync_tail_over_stage",
        ],
        "source": "raw_latency_stage_sum_rankmax_us",
        "note": "h20_only_fit_materializer_recovered_pending_cross_hw",
    },
}

WIDEEP_CONTEXT_TOKEN8_ENDPOINT = {
    "features": [
        "const",
        "log2_ep",
        "eplb_on",
        "log_raw_rank_mean_latency_us",
        "log_sync_rank",
        "sync_x_logep",
    ],
    "coef": np.array(
        [
            -0.02489739238684799,
            0.41546863245511495,
            0.053483782300209545,
            1.1716601176049615,
            -0.051931602572053946,
            0.41245915397138905,
        ],
        dtype=float,
    ),
    "version": "wideep_context_token8_endpoint_sync_ratio_lam0.01",
    "note": "fit_all_h20_h100_current_aic_latest_fixed_truth_20260726",
}

WIDEEP_CONTEXT_CURRENT_CLEAN_SHAPE_ENDPOINT = {
    "features": [
        "const",
        "log_current",
        "log2_token",
        "log2_ep",
        "eplb_on",
        "tiny8",
        "small16",
    ],
    "coef": np.array(
        [
            0.071735,
            0.988741,
            0.002611,
            -0.007626,
            -0.005870,
            0.039760,
            -0.026449,
        ],
        dtype=float,
    ),
    "version": "wideep_context_log_current_shape_endpoint_lambda1",
    "note": "current_clean_h20_h100_fit_all_holdout_pass_20260726",
}

ORDINARY_CONTEXT_SCALED_LEFT_ENDPOINT = {
    "features": [
        "const",
        "log_raw",
        "log_norm",
        "log_token",
        "log_ep",
        "eplb_on",
        "log_assign",
    ],
    "coef": np.array(
        [
            0.208623904,
            0.866321394,
            0.416517716,
            0.202285687,
            -0.374672371,
            0.00131419,
            -0.01047179,
        ],
        dtype=float,
    ),
    "version": "ordinary_context_current_log_raw_norm_token_le4",
    "note": "current_clean_h20_h100_fit_all_20260726",
}

ORDINARY_CONTEXT_TOKEN8_SCALE = 0.96

ORDINARY_GENERATION_SCALED_SOURCE_ENDPOINT = {
    "features": [
        "const",
        "log2_token",
        "log2_token_sq",
        "log2_ep",
        "eplb_on",
        "log_raw_us",
        "log_scaled_norm_us",
        "log_scaled_over_raw",
        "log_assign_over_token",
        "log_masked_over_rows",
        "log_active_over_token",
        "log2_token_x_log2_ep",
        "log_scaled_norm_us_x_log2_ep",
        "log_raw_us_x_log2_ep",
        "t_le4_x_log2_ep",
        "t_le4_x_eplb_on",
    ],
    "coef": np.array(
        [
            0.2909728768558627,
            0.1819767654470345,
            -0.039432899491997075,
            0.372671784302541,
            0.02867113858551338,
            0.12401856869859026,
            0.09837640573413221,
            -0.025642162966318775,
            -0.20358613720475555,
            0.1769261838025238,
            -0.3673594953603439,
            -0.016150467484538293,
            0.033781990499846336,
            -0.14403372736284517,
            -0.03905668035011354,
            -0.0017509146020911423,
        ],
        dtype=float,
    ),
    "version": "ordinary_generation_scaled_interact_log_scale_raw_lambda0p1",
    "note": "current_clean_h20_h100_fit_all_20260726",
}

ORDINARY_GENERATION_RECORDED_ONLY_COEFFICIENTS = {
    "ep1": {
        "features": OG_EP1_RECORDED_ONLY_FEATURES,
        "coef": np.array(
            [
                0.4021457590808713,
                0.010565192602124955,
                -0.03198890907272515,
                -0.38126194080527753,
                0.2655402331837651,
                0.6013166774632395,
                1.393602895160139,
                3.0889676650518654,
                8.000275470373797,
                -5.653691598511734,
                0.17987848427228778,
                1.3613735664677138,
                -6.377409097117888,
            ],
            dtype=float,
        ),
        "base": "raw_latency_us",
        "note": "ordinary_generation_ep1_raw_shape_more_tiny_fit_all_replay",
    },
    "ep_ge2": {
        "features": OG_EPGE2_RECORDED_ONLY_FEATURES,
        "coef": np.array(
            [
                0.25586935989717724,
                0.05467055977632946,
                -0.019923750141667505,
                0.11505276474413684,
                0.03441520957701139,
                -0.026681978758624166,
                -0.006909542307440893,
                -0.08011665708504413,
                0.13880587758498616,
                -0.44290792250455224,
                -0.05893168572814729,
                -0.012152937831880114,
                0.08271136501451173,
                -0.16099592396473303,
                -0.03867480164081231,
                0.202942922109872,
                0.0027198958692562377,
                -0.1904075671480705,
                0.22600970720915667,
                -0.19040756715255744,
                -0.4164172743538873,
                -0.08989891593587228,
                0.025595608482418004,
                0.3598023326115254,
                0.0487383085024386,
                -0.12779366032136535,
                0.11549452443360511,
            ],
            dtype=float,
        ),
        "base": "raw_latency_us",
        "note": "ordinary_generation_epge2_smooth_fit_all_replay",
    },
}


# Hardware-independent left-endpoint override for ordinary generation.
#
# Scope:
#   - EP1 token=1/2/4/8/16
#   - EP>=2 token=1/2/4
#   - EP>=2 token=8/16 intentionally stay on the v2 materializer curve
#   - token>=32 stays unchanged
#
# These scales are fitted from the formal H20/H100 fixed truth available on
# 2026-07-24. H100 EP4 token=1/2/4/16 is still pending a four-GPU rerun, so this
# block must be re-audited after that truth is added.
ORDINARY_GENERATION_LEFT_ENDPOINT_RAW_SCALES = {
    ("ep1_t_le4", 1): 1.149344209967539,
    ("ep1_t_le4", 2): 1.678041325382361,
    ("ep1_t_le4", 4): 2.707079355477595,
    ("ep1_t_8_16", 8): 2.088116598193495,
    ("ep1_t_8_16", 16): 2.6769625637485204,
    ("ep_ge2_t_le4", 1): 1.117891833256459,
    ("ep_ge2_t_le4", 2): 1.1273416102840854,
    ("ep_ge2_t_le4", 4): 1.3994893268311062,
}


RECORDED_MOE_COEFFICIENTS = {
    "ordinary_context": np.array(
        [
            1.1851882275070984,
            0.42116050026923973,
            -0.0027830367741772167,
            0.43979362104748987,
            0.1524772894569722,
            -0.0074812033206768771,
            -0.047618096793698023,
            0.57696230386938319,
            0.58156929634277377,
            0.18499399028629013,
            0.041275017533878793,
            -0.060523644697117483,
            0.17391596607676774,
            -0.4623800603770547,
            -0.099776861343475215,
            -0.11159415496185653,
            -0.083532883269015895,
            -0.077948975511258609,
            0.051775973849493805,
            -0.11650136807984553,
            0.061521312722954628,
            -0.11650136798987623,
            -0.17802268089664247,
            -0.49856076194946458,
            0.06696020129909501,
            -0.0020937976053057843,
            -0.051980649625972014,
        ],
        dtype=float,
    ),
    "ordinary_generation": np.array(
        [
            -0.0750744517201863,
            0.022227758545851701,
            -0.00083033033590479996,
            0.83383546247379292,
            -0.0216782684122989,
            -0.0053973365906604997,
            -0.061440139941074202,
            -0.068794013462740694,
            -0.034094406697907698,
            0.027813968418825601,
            -0.0828424546414743,
            -0.041322565460901399,
            -0.2950916136520072,
            -0.20938194211122549,
            -0.046112102374612697,
            0.086297238377275501,
            -0.14993841205739711,
            0.13167818351692809,
            -0.0074796006131041997,
            0.029564343636235502,
            -0.0022610375787993002,
            0.0016476584383931,
            0.13347962240350331,
            -0.17518142544187129,
            0.0118943586823802,
        ],
        dtype=float,
    ),
    "wideep_context": np.array(
        [
            -0.013044303419859338,
            0.044000592863125673,
            -0.0067839058909310206,
            -0.017317077252471631,
            -0.077711030203734932,
            -0.0046136197445944922,
            0.0042870936295930041,
            -0.17736058771945939,
            0.76052772114624423,
            0.24497563875740569,
            -0.061938133998234876,
            0.025109666566000912,
            -0.1929730775826147,
            -0.33175408638499976,
            -0.012709949195645253,
            -0.049752410190554465,
            -0.017141555460131543,
            -0.052936682466814829,
            0.0049137861414372897,
            0.031401577060235816,
            0.67779871775838541,
            -0.78831960007665436,
            -0.012147733182604177,
            0.46181212787060327,
            0.72189369416035265,
            0.22717617741706983,
            0.076849836716431294,
            0.64639714118794833,
            -0.81972117652346443,
            -0.10637684133824096,
            -0.22472706485731353,
            -0.28196626782291362,
            0.1363010066780099,
            0.23730910747646436,
        ],
        dtype=float,
    ),
    "wideep_generation": np.array([], dtype=float),
}


def default_betas() -> dict[str, np.ndarray]:
    return {family: beta.copy() for family, beta in RECORDED_MOE_COEFFICIENTS.items()}


def eplb_from_distribution(distribution: object) -> str | None:
    if distribution == "recorded_eplb":
        return "on"
    if distribution == "recorded_no_eplb":
        return "off"
    return None


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def passes_filter(df: pd.DataFrame, criteria: dict[str, object]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, expected in criteria.items():
        if isinstance(expected, int):
            mask &= pd.to_numeric(df[col], errors="coerce") == expected
        else:
            mask &= df[col] == expected
    return mask


def add_common_key(df: pd.DataFrame, platform: str, family: str | None = None) -> pd.DataFrame:
    d = df.copy()
    d["platform"] = platform
    if family is None:
        d["family"] = np.where(d["phase"] == "context", "ordinary_context", "ordinary_generation")
    else:
        d["family"] = family
        d["phase"] = "context"
    d["ep"] = pd.to_numeric(d["moe_ep_size"], errors="coerce").astype("Int64")
    d["token"] = pd.to_numeric(d["num_tokens"], errors="coerce").astype("Int64")
    d["eplb"] = d["distribution"].map(eplb_from_distribution)
    return d


def load_ordinary_sources(platform: str, final_path: Path, raw_path: Path) -> pd.DataFrame:
    final = pd.read_csv(final_path)
    raw = pd.read_csv(raw_path)
    final = final[
        passes_filter(final, ORDINARY_COMMON_FILTER)
        & final["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
        & final["phase"].isin(["context", "generation"])
    ].copy()
    raw = raw[
        passes_filter(raw, ORDINARY_COMMON_FILTER)
        & raw["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
        & raw["phase"].isin(["context", "generation"])
    ].copy()
    final = add_common_key(final, platform)
    raw = add_common_key(raw, platform)
    final_small = final[KEY + ["latency"]].rename(columns={"latency": "current_latency_ms"})
    raw_cols = [
        "latency",
        "ordinary_rank_local_latency_mean",
        "ordinary_rank_local_latency_p90",
        "ordinary_rank_local_latency_max",
        "ordinary_rank_local_latency_spread",
        "ordinary_rank_local_workload_count_mean",
        "ordinary_rank_local_rows_max",
        "ordinary_rank_local_rows_mean",
        "ordinary_rank_local_rows_sum_mean",
        "ordinary_rank_local_masked_m_max",
        "ordinary_rank_local_active_experts_max",
        "ordinary_rank_local_assignments_max",
        "ordinary_rank_local_tiny_amortized_latency_mean",
        "ordinary_rank_local_tiny_amortized_latency_p90",
        "ordinary_rank_local_tiny_amortized_latency_max",
        "ordinary_rank_local_tiny_amortized_latency_min",
        "ordinary_rank_local_tiny_amortized_loop_count",
        "ordinary_rank_local_tiny_scaled_latency_mean",
        "ordinary_rank_local_tiny_scaled_latency_p90",
        "ordinary_rank_local_tiny_scaled_latency_max",
        "ordinary_rank_local_tiny_scaled_latency_min",
        "ordinary_rank_local_tiny_scaled_latency_max_normalized",
        "ordinary_rank_local_tiny_scaled_scale_at_max_rank",
        "ordinary_rank_local_tiny_scaled_target_assignments",
    ]
    for col in raw_cols:
        if col not in raw.columns:
            raw[col] = np.nan
    raw_small = raw[KEY + raw_cols].rename(
        columns={
            "latency": "raw_latency_ms",
            "ordinary_rank_local_latency_mean": "raw_ordinary_rank_local_latency_mean_ms",
            "ordinary_rank_local_latency_p90": "raw_ordinary_rank_local_latency_p90_ms",
            "ordinary_rank_local_latency_max": "raw_ordinary_rank_local_latency_max_ms",
            "ordinary_rank_local_latency_spread": "raw_ordinary_rank_local_latency_spread_ms",
            "ordinary_rank_local_workload_count_mean": "raw_ordinary_rank_local_workload_count_mean",
            "ordinary_rank_local_rows_max": "raw_ordinary_rank_local_rows_max",
            "ordinary_rank_local_rows_mean": "raw_ordinary_rank_local_rows_mean",
            "ordinary_rank_local_rows_sum_mean": "raw_ordinary_rank_local_rows_sum_mean",
            "ordinary_rank_local_masked_m_max": "raw_ordinary_rank_local_masked_m_max",
            "ordinary_rank_local_active_experts_max": "raw_ordinary_rank_local_active_experts_max",
            "ordinary_rank_local_assignments_max": "raw_ordinary_rank_local_assignments_max",
            "ordinary_rank_local_tiny_amortized_latency_mean": (
                "raw_ordinary_rank_local_tiny_amortized_latency_mean_ms"
            ),
            "ordinary_rank_local_tiny_amortized_latency_p90": (
                "raw_ordinary_rank_local_tiny_amortized_latency_p90_ms"
            ),
            "ordinary_rank_local_tiny_amortized_latency_max": (
                "raw_ordinary_rank_local_tiny_amortized_latency_max_ms"
            ),
            "ordinary_rank_local_tiny_amortized_latency_min": (
                "raw_ordinary_rank_local_tiny_amortized_latency_min_ms"
            ),
            "ordinary_rank_local_tiny_amortized_loop_count": (
                "raw_ordinary_rank_local_tiny_amortized_loop_count"
            ),
            "ordinary_rank_local_tiny_scaled_latency_mean": (
                "raw_ordinary_rank_local_tiny_scaled_latency_mean_ms"
            ),
            "ordinary_rank_local_tiny_scaled_latency_p90": (
                "raw_ordinary_rank_local_tiny_scaled_latency_p90_ms"
            ),
            "ordinary_rank_local_tiny_scaled_latency_max": (
                "raw_ordinary_rank_local_tiny_scaled_latency_max_ms"
            ),
            "ordinary_rank_local_tiny_scaled_latency_min": (
                "raw_ordinary_rank_local_tiny_scaled_latency_min_ms"
            ),
            "ordinary_rank_local_tiny_scaled_latency_max_normalized": (
                "raw_ordinary_rank_local_tiny_scaled_latency_max_normalized_ms"
            ),
            "ordinary_rank_local_tiny_scaled_scale_at_max_rank": (
                "raw_ordinary_rank_local_tiny_scaled_scale_at_max_rank"
            ),
            "ordinary_rank_local_tiny_scaled_target_assignments": (
                "raw_ordinary_rank_local_tiny_scaled_target_assignments"
            ),
        }
    )
    merged = final_small.merge(raw_small, on=KEY, how="left")
    return merged


def load_wideep_context_sources(platform: str, final_path: Path, raw_path: Path) -> pd.DataFrame:
    final = pd.read_csv(final_path)
    raw = pd.read_csv(raw_path)
    final = final[
        passes_filter(final, WIDEEP_CONTEXT_COMMON_FILTER)
        & final["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
    ].copy()
    raw = raw[
        passes_filter(raw, WIDEEP_CONTEXT_COMMON_FILTER)
        & raw["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
    ].copy()
    final = add_common_key(final, platform, "wideep_context")
    raw = add_common_key(raw, platform, "wideep_context")
    final_small = final[KEY + ["latency"]].rename(columns={"latency": "current_latency_ms"})
    raw_cols = [
        "latency",
        "origin_latency",
        "latency_stage_sum_rankmax",
        "rank_sync_tail_mean",
        "rank_mean_latency",
        "rank_p90_latency",
        "multistream_latency",
        "workload_rank_assignments_max",
        "workload_rank_imbalance_max_over_mean",
        "workload_expert_m_max",
        "workload_expected_m_max",
    ]
    for col in raw_cols:
        if col not in raw.columns:
            raw[col] = np.nan
    raw_small = raw[KEY + raw_cols].rename(
        columns={
            "latency": "raw_latency_ms",
            "origin_latency": "raw_origin_latency_ms",
            "latency_stage_sum_rankmax": "raw_latency_stage_sum_rankmax_ms",
            "rank_sync_tail_mean": "raw_rank_sync_tail_mean_ms",
            "rank_mean_latency": "raw_rank_mean_latency_ms",
            "rank_p90_latency": "raw_rank_p90_latency_ms",
            "multistream_latency": "raw_multistream_latency_ms",
            "workload_rank_assignments_max": "raw_workload_rank_assignments_max",
            "workload_rank_imbalance_max_over_mean": "raw_workload_rank_imbalance_max_over_mean",
            "workload_expert_m_max": "raw_workload_expert_m_max",
            "workload_expected_m_max": "raw_workload_expected_m_max",
        }
    )
    merged = final_small.merge(raw_small, on=KEY, how="left")
    return merged


def load_wideep_generation_sources(platform: str, final_path: Path, raw_path: Path) -> pd.DataFrame:
    final = pd.read_csv(final_path)
    raw = pd.read_csv(raw_path)
    final = final[
        passes_filter(final, WIDEEP_GENERATION_COMMON_FILTER)
        & final["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
    ].copy()
    raw = raw[
        passes_filter(raw, WIDEEP_GENERATION_COMMON_FILTER)
        & raw["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
    ].copy()
    final = add_common_key(final, platform, "wideep_generation")
    final["phase"] = "generation"
    raw = add_common_key(raw, platform, "wideep_generation")
    raw["phase"] = "generation"
    final_small = final[KEY + ["latency"]].rename(columns={"latency": "current_latency_ms"})
    raw_cols = [
        "latency",
        "origin_latency",
        "rank_mean_latency",
        "latency_stage_sum_rankmax",
        "stage_kernel_sum_mean",
        "rank_sync_tail_mean",
        "replay_capacity",
        "kernel_regime",
    ]
    raw_small = raw[KEY + raw_cols].rename(
        columns={
            "latency": "raw_latency_ms",
            "origin_latency": "raw_origin_latency_ms",
            "rank_mean_latency": "raw_rank_mean_latency_ms",
            "latency_stage_sum_rankmax": "raw_latency_stage_sum_rankmax_ms",
            "stage_kernel_sum_mean": "raw_stage_kernel_sum_mean_ms",
            "rank_sync_tail_mean": "raw_rank_sync_tail_mean_ms",
            "replay_capacity": "raw_replay_capacity",
            "kernel_regime": "raw_kernel_regime",
        }
    )
    merged = final_small.merge(raw_small, on=KEY, how="left")
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in list(d.columns):
        if col.endswith("_ms"):
            d[col[:-3] + "_us"] = to_float(d[col]) * 1000.0
    token = to_float(d["token"]).clip(lower=1.0)
    ep = to_float(d["ep"]).clip(lower=1.0)
    d["const"] = 1.0
    d["log2_token"] = np.log2(token)
    d["log2_token_sq"] = d["log2_token"] ** 2
    d["inv_sqrt_token"] = 1.0 / np.sqrt(token)
    d["log2_ep"] = np.log2(ep)
    d["eplb_on"] = (d["eplb"] == "on").astype(float)

    regimes = {
        "tiny": token <= 64,
        "small": token <= 128,
        "mid": (token > 128) & (token <= 1024),
        "dense": (token > 1024) & (token < 4096),
        "tail": token >= 4096,
        "ep1": ep == 1,
        "ep2": ep == 2,
        "ep2_or_less": ep <= 2,
        "ep4_or_more": ep >= 4,
        "ep8_or_more": ep >= 8,
    }
    for name, mask in regimes.items():
        d[name] = mask.astype(float)

    cross_features = {}
    for left in ["tiny", "small", "mid", "dense", "tail"]:
        for right in ["ep1", "ep2", "ep2_or_less", "ep4_or_more", "ep8_or_more"]:
            interaction = d[left] * d[right]
            cross_features[f"{left}_x_{right}"] = interaction
            cross_features[f"log2_token_x_{left}_x_{right}"] = d["log2_token"] * interaction
            cross_features[f"log2_token_sq_x_{left}_x_{right}"] = d["log2_token_sq"] * interaction
    d = pd.concat([d, pd.DataFrame(cross_features, index=d.index)], axis=1).copy()

    def log_positive(col: str, out: str) -> None:
        if col not in d:
            d[out] = np.nan
            return
        d[out] = np.log(to_float(d[col]).clip(lower=EPS))

    def log_ratio(num: str, den: str, out: str) -> None:
        if num not in d or den not in d:
            d[out] = np.nan
            return
        ratio = (to_float(d[num]) / to_float(d[den])).replace([np.inf, -np.inf], np.nan)
        d[out] = np.log(ratio.clip(lower=EPS))

    def log1p_ratio(num: str, den: str, out: str) -> None:
        if num not in d or den not in d:
            d[out] = np.nan
            return
        ratio = (to_float(d[num]) / to_float(d[den])).replace([np.inf, -np.inf], np.nan)
        d[out] = np.log1p(ratio.clip(lower=0.0))

    log_positive("current_latency_us", "log_current_latency_us")
    log_positive("raw_latency_us", "log_raw_latency_us")
    log_positive("raw_latency_stage_sum_rankmax_us", "log_raw_latency_stage_sum_rankmax_us")
    log_positive("raw_origin_latency_us", "log_raw_origin_latency_us")
    log_ratio("raw_latency_us", "current_latency_us", "log_raw_current")
    log_ratio("raw_ordinary_rank_local_latency_max_us", "raw_ordinary_rank_local_latency_mean_us", "log_ord_max_mean")
    log_ratio("raw_ordinary_rank_local_latency_max_us", "raw_ordinary_rank_local_latency_mean_us", "log_ratio_max_mean")
    log1p_ratio("raw_ordinary_rank_local_latency_spread_us", "raw_ordinary_rank_local_latency_max_us", "log1p_ord_spread_max")
    log_ratio("raw_origin_latency_us", "raw_latency_us", "log_origin_raw")
    log_ratio("raw_latency_stage_sum_rankmax_us", "raw_latency_us", "log_stage_raw")
    log1p_ratio("raw_rank_sync_tail_mean_us", "raw_latency_stage_sum_rankmax_us", "log1p_sync_stage")
    log_ratio("raw_multistream_latency_us", "raw_latency_us", "log_ms_raw")
    log_ratio("raw_rank_mean_latency_us", "raw_latency_stage_sum_rankmax_us", "log_rank_over_stage")
    log_ratio("raw_rank_sync_tail_mean_us", "raw_latency_stage_sum_rankmax_us", "log_sync_tail_over_stage")
    log_positive("raw_rank_sync_tail_mean_us", "log_raw_rank_sync_tail_mean_us")
    log_positive("raw_rank_mean_latency_us", "log_raw_rank_mean_latency_us")
    log_positive("raw_rank_p90_latency_us", "log_raw_rank_p90_latency_us")
    log_positive("raw_workload_rank_assignments_max", "log_workload_rank_assignments_max")
    log_positive("raw_workload_rank_imbalance_max_over_mean", "log_workload_rank_imbalance_max_over_mean")
    log_positive("raw_workload_expert_m_max", "log_workload_expert_m_max")
    log_positive("raw_workload_expected_m_max", "log_workload_expected_m_max")
    d["left_endpoint"] = (to_float(d["token"]) <= 8).astype(float)
    d["logt_logep"] = d["log2_token"] * d["log2_ep"]
    d["tiny_decay"] = 1.0 / np.sqrt(to_float(d["token"]).clip(lower=1.0))
    d["tiny_by_logep"] = d["tiny_decay"] * d["log2_ep"]
    for name, center, width in (
        ("g8", 3.0, 1.0),
        ("g32", 5.0, 1.0),
        ("g64", 6.0, 1.0),
        ("g128", 7.0, 1.0),
        ("g512", 9.0, 1.2),
        ("g4096", 12.0, 1.3),
        ("g_tail", 14.0, 1.6),
    ):
        d[name] = np.exp(-0.5 * ((d["log2_token"] - center) / width) ** 2)
        d[f"{name}_ep"] = d[name] * d["log2_ep"]
    d["g8_log_sync_tail"] = d["g8"] * d["log_raw_rank_sync_tail_mean_us"]
    d["g8_log_stage"] = d["g8"] * d["log_raw_latency_stage_sum_rankmax_us"]
    log_positive("raw_ordinary_rank_local_latency_mean_us", "log_rank_mean_us")
    log_positive("raw_ordinary_rank_local_latency_max_us", "log_rank_max_us")
    log_ratio("raw_ordinary_rank_local_latency_max_us", "raw_ordinary_rank_local_latency_mean_us", "log_ratio_rankmax_mean")
    log_positive("raw_ordinary_rank_local_rows_max", "log_rows")
    log_positive("raw_ordinary_rank_local_rows_max", "log_rows_max")
    log_positive("raw_ordinary_rank_local_masked_m_max", "log_masked")
    log_positive("raw_ordinary_rank_local_masked_m_max", "log_masked_m_max")
    log_positive("raw_ordinary_rank_local_assignments_max", "log_assign")
    log_positive("raw_ordinary_rank_local_assignments_max", "log_assignments_max")
    log_positive("raw_ordinary_rank_local_active_experts_max", "log_active")
    log_positive("raw_ordinary_rank_local_active_experts_max", "log_active_experts_max")
    log_positive(
        "raw_ordinary_rank_local_tiny_amortized_latency_mean_us",
        "log_tiny_amortized_mean_us",
    )
    log_positive(
        "raw_ordinary_rank_local_tiny_amortized_latency_max_us",
        "log_tiny_amortized_max_us",
    )
    log_ratio(
        "raw_ordinary_rank_local_tiny_amortized_latency_max_us",
        "raw_ordinary_rank_local_tiny_amortized_latency_mean_us",
        "log_tiny_amortized_max_mean",
    )
    log_positive(
        "raw_ordinary_rank_local_tiny_scaled_latency_max_us",
        "log_tiny_scaled_max_us",
    )
    log_positive(
        "raw_ordinary_rank_local_tiny_scaled_latency_mean_us",
        "log_tiny_scaled_mean_us",
    )
    log_positive(
        "raw_ordinary_rank_local_tiny_scaled_latency_max_normalized_us",
        "log_tiny_scaled_max_normalized_us",
    )
    d["log_scaled_norm_us"] = d["log_tiny_scaled_max_normalized_us"]
    log_ratio(
        "raw_ordinary_rank_local_tiny_scaled_latency_max_normalized_us",
        "raw_latency_us",
        "log_scaled_over_raw",
    )
    log_positive(
        "raw_ordinary_rank_local_tiny_scaled_scale_at_max_rank",
        "log_tiny_scaled_scale",
    )
    log_ratio("raw_ordinary_rank_local_rows_max", "token", "log_rows_over_token")
    log_ratio("raw_ordinary_rank_local_masked_m_max", "raw_ordinary_rank_local_rows_max", "log_masked_over_rows")
    log_ratio("raw_ordinary_rank_local_assignments_max", "token", "log_assign_over_token")
    log_ratio("raw_ordinary_rank_local_active_experts_max", "token", "log_active_over_token")
    d = d.copy()
    d["log2_token_x_log2_ep"] = d["log2_token"] * d["log2_ep"]
    d["log_scaled_norm_us_x_log2_ep"] = d["log_scaled_norm_us"] * d["log2_ep"]
    d["log_raw_us_x_log2_ep"] = d["log_raw_latency_us"] * d["log2_ep"]
    d["t_le4"] = (token <= 4).astype(float)
    d["t_le4_x_log2_ep"] = d["t_le4"] * d["log2_ep"]
    d["t_le4_x_eplb_on"] = d["t_le4"] * d["eplb_on"]
    return d


def fit_coefficients(join: pd.DataFrame, family: str) -> np.ndarray:
    spec = MODEL_SPECS[family]
    if family in {"ordinary_generation", "wideep_generation"}:
        return np.array([], dtype=float)
    features = spec["features"]
    d = join[(join["family"] == family) & (join["token"] != 4)].copy()
    d = add_features(d)
    needed = features + ["truth_median_us"]
    if spec["base"]:
        needed.append(spec["base"])
    train = d.dropna(subset=needed).copy()
    if spec["base"]:
        train = train[to_float(train[spec["base"]]) > 0]
    train = train[to_float(train["truth_median_us"]) > 0]
    if len(train) < len(features) + 3:
        raise RuntimeError(f"not enough training rows for {family}: {len(train)} < {len(features)+3}")
    x = train[features].to_numpy(float)
    if spec["mode"] == "correction":
        y = np.log(to_float(train["truth_median_us"]).to_numpy() / to_float(train[spec["base"]]).to_numpy())
    else:
        y = np.log(to_float(train["truth_median_us"]).to_numpy())
    lam = float(spec["lambda"])
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def wideep_generation_regime(row: pd.Series) -> str | None:
    capacity = row.get("raw_replay_capacity")
    if capacity is not None and pd.notna(capacity):
        try:
            parsed = int(float(capacity))
            if parsed in (256, 512, 1024):
                return f"capacity_{parsed}"
        except (TypeError, ValueError):
            pass
    kernel_regime = str(row.get("raw_kernel_regime", ""))
    if "capacity_256" in kernel_regime:
        return "capacity_256"
    if "capacity_512" in kernel_regime:
        return "capacity_512"
    if "capacity_1024" in kernel_regime:
        return "capacity_1024"
    return None


def ordinary_generation_regime(row: pd.Series) -> str:
    return "ep1" if int(row["ep"]) == 1 else "ep_ge2"


def ordinary_context_scaled_left_endpoint_override(row: pd.Series) -> float | None:
    if int(row["token"]) > 4:
        return None
    raw_us = float(row.get("raw_latency_us", np.nan))
    norm_us = float(row.get("raw_ordinary_rank_local_tiny_scaled_latency_max_normalized_us", np.nan))
    assignments = float(row.get("raw_ordinary_rank_local_assignments_max", np.nan))
    if not math.isfinite(raw_us) or raw_us <= 0:
        return None
    if not math.isfinite(norm_us) or norm_us <= 0:
        return None
    if not math.isfinite(assignments) or assignments <= 0:
        return None

    token = float(row["token"])
    ep = float(row["ep"])
    log_norm = math.log(norm_us)
    log_token = math.log(max(token, 1.0))
    log_ep = math.log(max(ep, 1.0))
    values = {
        "const": 1.0,
        "log_raw": math.log(raw_us),
        "log_norm": log_norm,
        "log_token": log_token,
        "log_ep": log_ep,
        "eplb_on": float(row["eplb_on"]),
        "log_assign": math.log(max(assignments, 1.0)),
    }
    endpoint = ORDINARY_CONTEXT_SCALED_LEFT_ENDPOINT
    value_us = math.exp(
        float(np.array([values[f] for f in endpoint["features"]], dtype=float) @ endpoint["coef"])
    )
    return value_us if math.isfinite(value_us) and value_us > 0 else None


def ordinary_context_postprocess(row: pd.Series, value_us: float) -> float:
    if int(row["token"]) == 8:
        return value_us * ORDINARY_CONTEXT_TOKEN8_SCALE
    return value_us


def ordinary_generation_scaled_source_override(row: pd.Series) -> float | None:
    if int(row["token"]) > 16:
        return None
    source_us = float(row.get("raw_latency_us", np.nan))
    if not math.isfinite(source_us) or source_us <= 0:
        return None
    endpoint = ORDINARY_GENERATION_SCALED_SOURCE_ENDPOINT
    missing = []
    values = {}
    for feature in endpoint["features"]:
        if feature == "log_raw_us":
            value = row.get("log_raw_latency_us")
        else:
            value = row.get(feature)
        if value is None or not math.isfinite(float(value)):
            missing.append(feature)
        else:
            values[feature] = float(value)
    if missing:
        return None
    correction = math.exp(
        float(np.array([values[f] for f in endpoint["features"]], dtype=float) @ endpoint["coef"])
    )
    value_us = source_us * correction
    return value_us if math.isfinite(value_us) and value_us > 0 else None


def ordinary_generation_left_endpoint_override(row: pd.Series) -> float | None:
    token = int(row["token"])
    ep = int(row["ep"])
    if token > 16:
        return None
    if ep == 1:
        group = "ep1_t_le4" if token <= 4 else "ep1_t_8_16"
    elif token <= 4:
        group = "ep_ge2_t_le4"
    else:
        return None
    scale = ORDINARY_GENERATION_LEFT_ENDPOINT_RAW_SCALES.get((group, token))
    if scale is None:
        return None
    source_us = float(row["raw_latency_us"])
    if not math.isfinite(source_us) or source_us <= 0:
        return None
    return source_us * scale


def wideep_context_current_clean_shape_endpoint(row: pd.Series, current_us: float) -> float:
    if not math.isfinite(current_us) or current_us <= 0:
        return current_us
    values = {
        "const": 1.0,
        "log_current": math.log(current_us),
        "log2_token": float(row["log2_token"]),
        "log2_ep": float(row["log2_ep"]),
        "eplb_on": float(row["eplb_on"]),
        "tiny8": 1.0 if int(row["token"]) == 8 else 0.0,
        "small16": 1.0 if int(row["token"]) == 16 else 0.0,
    }
    endpoint = WIDEEP_CONTEXT_CURRENT_CLEAN_SHAPE_ENDPOINT
    value_us = math.exp(
        float(np.array([values[f] for f in endpoint["features"]], dtype=float) @ endpoint["coef"])
    )
    return value_us if math.isfinite(value_us) and value_us > 0 else current_us


def wideep_context_token8_endpoint_override(row: pd.Series) -> float | None:
    if int(row["token"]) != 8:
        return None
    source_us = float(row.get("raw_rank_mean_latency_us", np.nan))
    sync_us = float(row.get("raw_rank_sync_tail_mean_us", np.nan))
    if not math.isfinite(source_us) or source_us <= 0:
        return None
    if not math.isfinite(sync_us) or sync_us <= 0:
        return None
    values = {
        "const": 1.0,
        "log2_ep": float(row["log2_ep"]),
        "eplb_on": float(row["eplb_on"]),
        "log_raw_rank_mean_latency_us": math.log(source_us),
        "log_sync_rank": math.log(max(sync_us / source_us, EPS)),
        "sync_x_logep": math.log(max(sync_us / source_us, EPS)) * float(row["log2_ep"]),
    }
    endpoint = WIDEEP_CONTEXT_TOKEN8_ENDPOINT
    x = np.array([values[f] for f in endpoint["features"]], dtype=float)
    value_us = math.exp(float(x @ endpoint["coef"]))
    return value_us if math.isfinite(value_us) and value_us > 0 else None


def predict(df: pd.DataFrame, family: str, beta: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    spec = MODEL_SPECS[family]
    d = add_features(df[df["family"] == family].copy())
    invalid_rows = []
    pred = pd.Series(index=d.index, dtype=float)
    required = list(spec["features"])
    if spec["base"]:
        required.append(spec["base"])
    if family == "ordinary_generation":
        required = ["raw_latency_us"]
    if family == "wideep_generation":
        required = [
            "raw_latency_stage_sum_rankmax_us",
            "raw_stage_kernel_sum_mean_us",
            "raw_rank_mean_latency_us",
            "raw_rank_sync_tail_mean_us",
        ]
    for idx, row in d.iterrows():
        missing = []
        if family == "wideep_generation" and wideep_generation_regime(row) is None:
            missing.append("raw_replay_capacity_or_raw_kernel_regime")
        for col in required:
            value = row.get(col)
            if value is None or not np.isfinite(float(value)):
                missing.append(col)
            elif col.endswith("_us") and float(value) <= 0:
                missing.append(col)
        if missing:
            invalid_rows.append({**{k: row.get(k) for k in KEY}, "missing_or_invalid": ";".join(missing)})
            continue
        if family == "ordinary_generation":
            override_us = ordinary_generation_scaled_source_override(row)
            if override_us is not None:
                value_us = override_us
            else:
                regime = ordinary_generation_regime(row)
                og = ORDINARY_GENERATION_RECORDED_ONLY_COEFFICIENTS[regime]
                missing_features = []
                for feature in og["features"]:
                    value = row.get(feature)
                    if value is None or not np.isfinite(float(value)):
                        missing_features.append(feature)
                if missing_features:
                    invalid_rows.append({**{k: row.get(k) for k in KEY}, "missing_or_invalid": ";".join(missing_features)})
                    continue
                source_us = float(row["raw_latency_us"])
                if not math.isfinite(source_us) or source_us <= 0:
                    invalid_rows.append({**{k: row.get(k) for k in KEY}, "missing_or_invalid": "ordinary_generation_source"})
                    continue
                override_us = ordinary_generation_left_endpoint_override(row)
                if override_us is not None:
                    value_us = override_us
                else:
                    x = np.array([float(row[f]) for f in og["features"]], dtype=float)
                    value_us = source_us * math.exp(float(x @ og["coef"]))
        elif family == "wideep_generation":
            regime = wideep_generation_regime(row)
            wg = WIDEEP_GENERATION_COEFFICIENTS[regime]
            if regime == "capacity_256":
                source_us = (float(row["raw_stage_kernel_sum_mean_us"]) ** 0.6) * (
                    float(row["raw_latency_stage_sum_rankmax_us"]) ** 0.4
                )
            elif regime == "capacity_512":
                source_us = float(row["raw_stage_kernel_sum_mean_us"])
            else:
                source_us = float(row["raw_latency_stage_sum_rankmax_us"])
            if not math.isfinite(source_us) or source_us <= 0:
                invalid_rows.append({**{k: row.get(k) for k in KEY}, "missing_or_invalid": "wideep_generation_source"})
                continue
            if regime == "capacity_1024":
                values = {
                    "const": 1.0,
                    "log_source": math.log(source_us),
                    "log2_token": float(row["log2_token"]),
                    "log2_ep": float(row["log2_ep"]),
                    "eplb_on": float(row["eplb_on"]),
                    "logt_logep": float(row["logt_logep"]),
                    "tiny_decay": float(row["tiny_decay"]),
                    "tiny_by_logep": float(row["tiny_by_logep"]),
                    "log_rank_over_stage": float(row["log_rank_over_stage"]),
                    "log_sync_tail_over_stage": float(row["log_sync_tail_over_stage"]),
                }
                value_us = math.exp(float(np.array([values[f] for f in wg["features"]], dtype=float) @ wg["coef"]))
            else:
                values = {
                    "const": 1.0,
                    "log2_token": float(row["log2_token"]),
                    "tiny_decay": float(row["tiny_decay"]),
                    "log_rank_over_stage": float(row["log_rank_over_stage"]),
                }
                correction = math.exp(float(np.array([values[f] for f in wg["features"]], dtype=float) @ wg["coef"]))
                value_us = source_us * correction
        else:
            if family == "ordinary_context":
                override_us = ordinary_context_scaled_left_endpoint_override(row)
            elif family == "wideep_context":
                override_us = wideep_context_token8_endpoint_override(row)
            else:
                override_us = None
            if override_us is not None:
                value_us = override_us
            else:
                x = row[spec["features"]].to_numpy(float)
                if spec["mode"] == "correction":
                    value_us = float(row[spec["base"]]) * math.exp(float(x @ beta))
                else:
                    value_us = math.exp(float(x @ beta))
            if family == "ordinary_context":
                value_us = ordinary_context_postprocess(row, value_us)
            elif family == "wideep_context":
                value_us = wideep_context_current_clean_shape_endpoint(row, value_us)
        if not math.isfinite(value_us) or value_us <= 0:
            invalid_rows.append({**{k: row.get(k) for k in KEY}, "missing_or_invalid": "predicted_latency"})
            continue
        pred.loc[idx] = value_us
    return pred, pd.DataFrame(invalid_rows)


def _write_duplicate_source_audit(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(
            columns=KEY
            + [
                "model",
                "model_version",
                "duplicate_rows",
                "selected_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "spread_pct",
            ]
        ).to_csv(path, index=False)


def _collapse_duplicate_predictions(
    key_to_pred: pd.DataFrame,
    *,
    family: str,
    model_version: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Collapse duplicate semantic keys without depending on row order."""

    duplicate_audit: list[dict[str, object]] = []
    if key_to_pred.empty:
        return key_to_pred, duplicate_audit

    collapsed_rows = []
    for key_values, group in key_to_pred.groupby(KEY, dropna=False, sort=True):
        latencies = to_float(group["p2_latency_ms"]).dropna().sort_values()
        if latencies.empty:
            continue
        selected = float(latencies.median())
        if len(group) > 1:
            min_latency = float(latencies.min())
            max_latency = float(latencies.max())
            spread_pct = (
                (max_latency - min_latency) / selected * 100.0
                if selected > 0
                else float("inf")
            )
            duplicate_audit.append(
                {
                    **dict(zip(KEY, key_values, strict=True)),
                    "model": family,
                    "model_version": model_version,
                    "duplicate_rows": int(len(group)),
                    "selected_latency_ms": selected,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "spread_pct": spread_pct,
                }
            )
        row = {key: value for key, value in zip(KEY, key_values, strict=True)}
        row["p2_latency_ms"] = selected
        collapsed_rows.append(row)
    return pd.DataFrame(collapsed_rows), duplicate_audit


def _apply_ordinary_context_left_curve_guard(key_to_pred: pd.DataFrame) -> pd.DataFrame:
    """Keep token=1 on the same small-token curve as token=2.

    The DSV3 ordinary-context token=1 endpoint is short enough that the scaled
    source probe can occasionally land on the token=2-like branch.  This guard
    is deliberately local and hardware-independent: it only compares adjacent
    AIC predictions with the same platform/EP/EPLB and never uses truth.
    """

    if key_to_pred.empty:
        return key_to_pred
    out = key_to_pred.copy()
    cap_ratio = 0.65
    for group_key, group in out.groupby(["platform", "family", "phase", "ep", "eplb"], dropna=False):
        _, family, phase, ep, _ = group_key
        if family != "ordinary_context" or phase != "context" or int(ep) > 4:
            continue
        token1_idx = group.index[group["token"].astype(int).eq(1)]
        token2 = group.loc[group["token"].astype(int).eq(2), "p2_latency_ms"]
        if len(token1_idx) != 1 or len(token2) != 1:
            continue
        cap = float(token2.iloc[0]) * cap_ratio
        current = float(out.loc[token1_idx[0], "p2_latency_ms"])
        if math.isfinite(cap) and cap > 0 and current > cap:
            out.loc[token1_idx[0], "p2_latency_ms"] = cap
    return out


def materialize_table(
    df: pd.DataFrame,
    source_rows: pd.DataFrame,
    families: list[str],
    betas: dict[str, np.ndarray],
    output_path: Path,
    invalid_path: Path,
    duplicate_path: Path | None = None,
) -> None:
    out = df.copy()
    invalid_parts = []
    duplicate_parts = []
    if "origin_latency" not in out.columns:
        out["origin_latency"] = out.get("latency", "")
    if "aic_latency_source" not in out.columns:
        out["aic_latency_source"] = ""
    if "aic_latency_policy" not in out.columns:
        out["aic_latency_policy"] = ""
    for family in families:
        spec = MODEL_SPECS[family]
        if not spec.get("active", True):
            raise RuntimeError(f"{family} materializer is inactive; do not write unvalidated clean latency")
        pred, invalid = predict(source_rows, family, betas[family])
        if not invalid.empty:
            invalid_parts.append(invalid.assign(model=family, model_version=spec["version"]))
        valid = pred.dropna()
        key_to_pred = source_rows.loc[valid.index, KEY].copy()
        key_to_pred["p2_latency_ms"] = valid.to_numpy() / 1000.0
        key_to_pred, duplicate_audit = _collapse_duplicate_predictions(
            key_to_pred,
            family=family,
            model_version=spec["version"],
        )
        if family == "ordinary_context":
            key_to_pred = _apply_ordinary_context_left_curve_guard(key_to_pred)
        duplicate_parts.extend(duplicate_audit)
        for _, row in key_to_pred.iterrows():
            mask = pd.Series(True, index=out.index)
            if family.startswith("ordinary"):
                mask &= out["op_name"] == "moe"
                mask &= out["phase"] == row["phase"]
            elif family == "wideep_generation":
                mask &= out["op_name"] == "moe_generation"
            else:
                mask &= out["op_name"] == "moe_context"
            mask &= out["distribution"] == ("recorded_eplb" if row["eplb"] == "on" else "recorded_no_eplb")
            mask &= pd.to_numeric(out["moe_ep_size"], errors="coerce") == int(row["ep"])
            mask &= pd.to_numeric(out["num_tokens"], errors="coerce") == int(row["token"])
            if family.startswith("ordinary"):
                mask &= passes_filter(out, ORDINARY_COMMON_FILTER)
            elif family == "wideep_generation":
                mask &= passes_filter(out, WIDEEP_GENERATION_COMMON_FILTER)
            else:
                mask &= passes_filter(out, WIDEEP_CONTEXT_COMMON_FILTER)
            p2_latency_ms = float(row["p2_latency_ms"])
            out.loc[mask, "latency"] = p2_latency_ms
            out.loc[mask, "aic_critical_path_latency"] = p2_latency_ms
            out.loc[mask, "aic_latency_source"] = "recorded_required_sources"
            out.loc[mask, "aic_latency_policy"] = spec["version"]
            if spec["kernel_source"]:
                out.loc[mask, "kernel_source"] = spec["kernel_source"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    if invalid_parts:
        pd.concat(invalid_parts, ignore_index=True).to_csv(invalid_path, index=False)
    else:
        pd.DataFrame(columns=KEY + ["missing_or_invalid", "model", "model_version"]).to_csv(invalid_path, index=False)
    if duplicate_path is not None:
        _write_duplicate_source_audit(duplicate_path, duplicate_parts)
