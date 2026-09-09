// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, table-free analytical kernels used by `DatabaseMode::Analytical`.
//!
//! The analytical mode deliberately depends only on the system YAML, operator
//! shape, and the policy carried on the engine spec.  It never opens a perf
//! parquet file.  The formulas are conservative roofline models with explicit
//! launch/efficiency terms; collected tables remain owned by SILICON mode.

use std::collections::BTreeMap;

use crate::common::enums::{CommQuantMode, ComputeDtype, QuantMapping};
use crate::common::error::AicError;
use crate::common::system_spec::{quant_tc_flops, SystemSpec};

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticalConfig {
    pub level: String,
    pub fp8_gemm_recipe: String,
    pub attention_algorithm: String,
    pub sparse_attention_head_quantum: Option<u32>,
    pub communication_mode: String,
    pub moe_dispatch_dtype: String,
    pub moe_combine_dtype: String,
    pub wideep_dispatch_dtype: String,
    pub wideep_combine_dtype: String,
    pub communication_placement: String,
}

impl Default for AnalyticalConfig {
    fn default() -> Self {
        Self {
            level: "standard".to_string(),
            fp8_gemm_recipe: "sglang".to_string(),
            attention_algorithm: "fa2".to_string(),
            sparse_attention_head_quantum: None,
            communication_mode: "empirical".to_string(),
            moe_dispatch_dtype: "half".to_string(),
            moe_combine_dtype: "half".to_string(),
            wideep_dispatch_dtype: "half".to_string(),
            wideep_combine_dtype: "half".to_string(),
            communication_placement: "independent".to_string(),
        }
    }
}

impl AnalyticalConfig {
    /// Decode the flat `EngineConfig.extra` representation. Unknown keys are
    /// ignored so adding a Python-side policy field stays backward compatible.
    pub fn from_extra(extra: &BTreeMap<String, String>) -> Self {
        let mut config = Self::default();
        if let Some(value) = extra.get("analytical_level") {
            config.level = value.clone();
        }
        if let Some(value) = extra.get("analytical_fp8_gemm_recipe") {
            config.fp8_gemm_recipe = value.clone();
        }
        if let Some(value) = extra.get("analytical_attention_algorithm") {
            config.attention_algorithm = value.clone();
        }
        if let Some(value) = extra.get("analytical_sparse_attention_head_quantum") {
            config.sparse_attention_head_quantum = match value.trim().to_ascii_lowercase().as_str() {
                "" | "none" | "null" => None,
                value => value.parse::<u32>().ok().filter(|quantum| matches!(quantum, 64 | 128)),
            };
        }
        if let Some(value) = extra.get("analytical_communication_mode") {
            config.communication_mode = value.clone();
        }
        for (key, target) in [
            ("analytical_moe_dispatch_dtype", &mut config.moe_dispatch_dtype),
            ("analytical_moe_combine_dtype", &mut config.moe_combine_dtype),
            ("analytical_wideep_dispatch_dtype", &mut config.wideep_dispatch_dtype),
            ("analytical_wideep_combine_dtype", &mut config.wideep_combine_dtype),
        ] {
            if let Some(value) = extra.get(key) {
                *target = value.clone();
            }
        }
        if let Some(value) = extra.get("analytical_communication_placement") {
            config.communication_placement = value.clone();
        }
        config.normalize();
        config
    }

    fn normalize(&mut self) {
        self.level = self.level.trim().to_ascii_lowercase();
        if !matches!(self.level.as_str(), "low" | "standard" | "high") {
            self.level = "standard".to_string();
        }
        self.fp8_gemm_recipe = self.fp8_gemm_recipe.trim().to_ascii_lowercase();
        self.attention_algorithm = self.attention_algorithm.trim().to_ascii_lowercase();
        self.communication_mode = self.communication_mode.trim().to_ascii_lowercase();
        for dtype in [
            &mut self.moe_dispatch_dtype,
            &mut self.moe_combine_dtype,
            &mut self.wideep_dispatch_dtype,
            &mut self.wideep_combine_dtype,
        ] {
            let normalized = dtype.trim().to_ascii_lowercase();
            *dtype = if matches!(normalized.as_str(), "half" | "fp8" | "int8") {
                normalized
            } else {
                "half".to_string()
            }
        }
        self.communication_placement = self.communication_placement.trim().to_ascii_lowercase();
    }

    fn profile(self: &Self) -> (f64, f64, f64) {
        // (memory efficiency, compute efficiency, launch floor in us)
        match self.level.as_str() {
            "low" => (0.88, 0.96, 1.5),
            "high" => (0.58, 0.70, 3.0),
            _ => (0.72, 0.86, 2.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AnalyticalConfig;

    #[test]
    fn extra_round_trip_carries_extended_policy() {
        let extra = [
            ("analytical_level", "high"),
            ("analytical_sparse_attention_head_quantum", "128"),
            ("analytical_moe_dispatch_dtype", "fp8"),
            ("analytical_moe_combine_dtype", "int8"),
            ("analytical_wideep_dispatch_dtype", "fp8"),
            ("analytical_wideep_combine_dtype", "half"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();

        let config = AnalyticalConfig::from_extra(&extra);
        assert_eq!(config.level, "high");
        assert_eq!(config.sparse_attention_head_quantum, Some(128));
        assert_eq!(config.moe_dispatch_dtype, "fp8");
        assert_eq!(config.moe_combine_dtype, "int8");
        assert_eq!(config.wideep_dispatch_dtype, "fp8");
        assert_eq!(config.wideep_combine_dtype, "half");
    }

    #[test]
    fn invalid_extended_policy_uses_safe_defaults() {
        let extra = [("analytical_sparse_attention_head_quantum", "32"), ("analytical_moe_dispatch_dtype", "bf16")]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();

        let config = AnalyticalConfig::from_extra(&extra);
        assert_eq!(config.sparse_attention_head_quantum, None);
        assert_eq!(config.moe_dispatch_dtype, "half");
    }
}

fn positive(value: f64, name: &str) -> Result<f64, AicError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(AicError::InvalidEngineConfig(format!(
            "analytical hardware field {name} must be positive, got {value}"
        )))
    }
}

fn peak_for(spec: &SystemSpec, mapping: QuantMapping) -> Result<f64, AicError> {
    quant_tc_flops(spec, mapping)
}

fn named_peak(value: Option<f64>, name: &str) -> Result<f64, AicError> {
    positive(
        value.ok_or_else(|| {
            AicError::MissingSystemFlops(format!(
                "analytical recipe requires gpu.{name}, but the system YAML does not define it"
            ))
        })?,
        name,
    )
}

fn mxfp4_scale_bytes(rows: f64, cols: f64) -> f64 {
    rows * (cols / 32.0).ceil()
}

fn nvfp4_scale_bytes(rows: f64, cols: f64) -> f64 {
    let rounded_rows = (rows / 128.0).ceil() * 128.0;
    let scale_cols = (cols / 16.0).ceil();
    let rounded_scale_cols = (scale_cols / 4.0).ceil() * 4.0;
    rounded_rows * rounded_scale_cols
}

#[derive(Clone, Copy, Debug)]
struct GemmProfile {
    floor_us: f64,
    eta_mem: f64,
    eta_compute: f64,
    eta_quant: f64,
    rho_transition: f64,
}

fn gemm_profile(config: &AnalyticalConfig, recipe: &str, architecture: Option<&str>) -> Result<GemmProfile, AicError> {
    let profile = match (recipe, architecture, config.level.as_str()) {
        ("bf16", _, "low") => GemmProfile { floor_us: 1.5, eta_mem: 0.96, eta_compute: 1.0, eta_quant: 1.0, rho_transition: 0.0 },
        ("bf16", _, "high") => GemmProfile { floor_us: 3.0, eta_mem: 0.60, eta_compute: 0.70, eta_quant: 1.0, rho_transition: 0.0 },
        ("bf16", _, _) => GemmProfile { floor_us: 2.0, eta_mem: 0.77, eta_compute: 0.90, eta_quant: 1.0, rho_transition: 0.0 },
        ("sglang", _, "low") => GemmProfile { floor_us: 4.0, eta_mem: 0.88, eta_compute: 0.78, eta_quant: 0.58, rho_transition: 1.1 },
        ("sglang", _, "high") => GemmProfile { floor_us: 6.5, eta_mem: 0.58, eta_compute: 0.52, eta_quant: 0.38, rho_transition: 1.9 },
        ("sglang", _, _) => GemmProfile { floor_us: 5.0, eta_mem: 0.70, eta_compute: 0.62, eta_quant: 0.46, rho_transition: 1.5 },
        ("deepgemm", Some("hopper"), "low") => GemmProfile { floor_us: 3.0, eta_mem: 1.0, eta_compute: 0.81, eta_quant: 0.85, rho_transition: 1.6 },
        ("deepgemm", Some("blackwell"), "low") => GemmProfile { floor_us: 8.0, eta_mem: 0.60, eta_compute: 0.81, eta_quant: 0.43, rho_transition: 1.6 },
        ("deepgemm", Some("hopper"), "high") => GemmProfile { floor_us: 5.0, eta_mem: 0.72, eta_compute: 0.54, eta_quant: 0.57, rho_transition: 2.6 },
        ("deepgemm", Some("blackwell"), "high") => GemmProfile { floor_us: 13.0, eta_mem: 0.40, eta_compute: 0.54, eta_quant: 0.28, rho_transition: 2.6 },
        ("deepgemm", Some("hopper"), _) => GemmProfile { floor_us: 4.0, eta_mem: 0.86, eta_compute: 0.65, eta_quant: 0.68, rho_transition: 2.1 },
        ("deepgemm", Some("blackwell"), _) => GemmProfile { floor_us: 10.0, eta_mem: 0.48, eta_compute: 0.65, eta_quant: 0.34, rho_transition: 2.1 },
        ("deepgemm", _, _) => {
            return Err(AicError::InvalidEngineConfig(
                "DeepGEMM analytical recipe requires hopper or blackwell architecture".into(),
            ))
        }
        (unknown, _, _) => {
            return Err(AicError::InvalidEngineConfig(format!(
                "unknown analytical GEMM recipe {unknown:?}"
            )))
        }
    };
    Ok(profile)
}

fn transition_seconds(profile: GemmProfile, floor_seconds: f64, body_seconds: f64) -> f64 {
    profile.rho_transition * floor_seconds * body_seconds / (body_seconds + floor_seconds)
}

fn roofline_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    flops: f64,
    bytes: f64,
    peak_flops: f64,
    launch_us: f64,
    sum_components: bool,
) -> Result<f64, AicError> {
    let (eta_mem, eta_compute, default_launch_us) = config.profile();
    let peak_flops = positive(peak_flops, "tensor-core FLOPS")?;
    let mem_bw = positive(spec.gpu.mem_bw, "mem_bw")?;
    let compute_ms = flops / (peak_flops * eta_compute) * 1000.0;
    let memory_ms = bytes / (mem_bw * eta_mem) * 1000.0;
    let body_ms = if sum_components {
        compute_ms + memory_ms
    } else {
        compute_ms.max(memory_ms)
    };
    Ok(body_ms + launch_us.max(default_launch_us) / 1000.0)
}

pub fn memory_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    bytes: f64,
) -> Result<f64, AicError> {
    let (eta_mem, _, launch_us) = config.profile();
    let mem_bw = positive(spec.gpu.mem_bw, "mem_bw")?;
    Ok(bytes / (mem_bw * eta_mem) * 1000.0 + launch_us / 1000.0)
}

pub fn gemm_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    mapping: QuantMapping,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64, AicError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "analytical GEMM shape must be positive, got m={m}, n={n}, k={k}"
        )));
    }
    let m = f64::from(m);
    let n = f64::from(n);
    let k = f64::from(k);
    let flops = 2.0 * m * n * k;
    let bandwidth = positive(spec.gpu.mem_bw, "mem_bw")?;

    match mapping.name {
        // Python estimate_bf16_gemm: full BF16 input, weight, and output
        // traffic, with a serial memory+compute body.
        "bfloat16" => {
            let profile = gemm_profile(config, "bf16", None)?;
            let peak = peak_for(spec, mapping)?;
            let bytes = 2.0 * (m * k + k * n + m * n);
            let memory_s = bytes / bandwidth / profile.eta_mem;
            let compute_s = flops / peak / profile.eta_compute;
            Ok((profile.floor_us * 1e-6 + memory_s + compute_s) * 1000.0)
        }
        // Python estimate_w8a16_gemm: only weight values and one FP32 scale
        // per output channel are changed; compute and launch remain BF16.
        "int8_wo" => {
            let profile = gemm_profile(config, "bf16", None)?;
            let peak = peak_for(spec, mapping)?;
            let bytes = 2.0 * m * k + n * k + 4.0 * n + 2.0 * m * n;
            let memory_s = bytes / bandwidth / profile.eta_mem;
            let compute_s = flops / peak / profile.eta_compute;
            Ok((profile.floor_us * 1e-6 + memory_s + compute_s) * 1000.0)
        }
        "fp8" | "fp8_static" | "fp8_block" | "fp8_ootb" => {
            if config.fp8_gemm_recipe == "sglang" {
                let profile = gemm_profile(config, "sglang", None)?;
                let peak = peak_for(spec, mapping)?;
                let bytes = m * k + n * k + 2.0 * m * n + 4.0 * m + 4.0 * n;
                let quant_bytes = 3.0 * m * k + 4.0 * m;
                let memory_s = bytes / bandwidth / profile.eta_mem;
                let compute_s = flops / peak / profile.eta_compute;
                let quant_s = quant_bytes / bandwidth / profile.eta_quant;
                let floor_s = profile.floor_us * 1e-6;
                let body_s = quant_s + memory_s.max(compute_s);
                let transition_s = transition_seconds(profile, floor_s, body_s);
                Ok((floor_s + body_s + transition_s) * 1000.0)
            } else {
                let architecture = match config.fp8_gemm_recipe.as_str() {
                    "deepgemm-hopper" => "hopper",
                    "deepgemm-blackwell" => "blackwell",
                    other => {
                        return Err(AicError::InvalidEngineConfig(format!(
                            "unknown analytical FP8 GEMM recipe {other:?}"
                        )))
                    }
                };
                if n < 128.0 || k < 128.0 {
                    return Err(AicError::InvalidEngineConfig(
                        "DeepGEMM model requires n and k >= 128".into(),
                    ));
                }
                let profile = gemm_profile(config, "deepgemm", Some(architecture))?;
                let peak = peak_for(spec, mapping)?;
                let scale_a_bytes = 4.0 * m * (k / 128.0).ceil();
                let scale_b_bytes = 4.0 * (n / 128.0).ceil() * (k / 128.0).ceil();
                let bytes = m * k + n * k + 2.0 * m * n + scale_a_bytes + scale_b_bytes;
                let quant_bytes = 3.0 * m * k + scale_a_bytes;
                let memory_s = bytes / bandwidth / profile.eta_mem;
                let compute_s = flops / peak / profile.eta_compute;
                let quant_s = quant_bytes / bandwidth / profile.eta_quant;
                let floor_s = profile.floor_us * 1e-6;
                let body_s = quant_s + memory_s.max(compute_s);
                let transition_s = transition_seconds(profile, floor_s, body_s);
                Ok((floor_s + body_s + transition_s) * 1000.0)
            }
        }
        unsupported => Err(AicError::InvalidEngineConfig(format!(
            "ANALYTICAL GEMM does not support quant mode {unsupported:?}; supported modes are BF16, W8A16, and FP8 variants"
        ))),
    }
}

pub fn attention_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    batch: u32,
    query_length: u32,
    kv_length: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    kv_bytes_per_element: f64,
    compute_mapping: QuantMapping,
    ) -> Result<f64, AicError> {
    if batch == 0 || query_length == 0 || kv_length == 0 || query_heads == 0 || head_dim == 0 {
        return Err(AicError::InvalidEngineConfig("analytical attention shape must be positive".into()));
    }
    let b = f64::from(batch);
    let q = f64::from(query_length);
    let kv = f64::from(kv_length);
    let heads = f64::from(query_heads);
    let kv_heads = f64::from(kv_heads.max(1));
    let dim = f64::from(head_dim);
    let flops = 4.0 * b * heads * q * kv * dim;
    let bytes = b * (heads * q * dim * 2.0 + kv_heads * kv * dim * kv_bytes_per_element);
    let launch_us = if config.attention_algorithm == "fa3" { 3.0 } else { 2.0 };
    roofline_ms(
        spec,
        config,
        flops,
        bytes,
        peak_for(spec, compute_mapping)?,
        launch_us,
        false,
    )
}

#[derive(Clone, Copy, Debug)]
struct FaParts {
    raw_hbm_s: f64,
    raw_l2_s: f64,
    compute_s: f64,
    raw_bottleneck: u8,
    cta_count: f64,
    fractional_task_waves: f64,
}

fn dtype_bytes(dtype: &str) -> Result<f64, AicError> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "fp8" | "float8" | "float8_e4m3fn" => Ok(1.0),
        "fp16" | "float16" | "bf16" | "bfloat16" => Ok(2.0),
        "fp32" | "float32" => Ok(4.0),
        other => Err(AicError::InvalidEngineConfig(format!("unsupported analytical dtype {other:?}"))),
    }
}

fn attention_peak(spec: &SystemSpec, dtype: &str) -> Result<f64, AicError> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "fp8" | "float8" | "float8_e4m3fn" => named_peak(spec.gpu.fp8_tc_flops, "fp8_tc_flops"),
        "fp16" | "float16" | "bf16" | "bfloat16" => {
            named_peak(spec.gpu.bfloat16_tc_flops, "bfloat16_tc_flops")
        }
        "fp32" | "float32" => named_peak(spec.gpu.bfloat16_tc_flops, "bfloat16_tc_flops"),
        other => Err(AicError::InvalidEngineConfig(format!("unsupported analytical dtype {other:?}"))),
    }
}

/// Return the matrix-input dtype name used by the detailed attention model.
/// Quant mappings keep the storage and compute dtypes separate, so callers
/// must choose this from the compute side of the mapping rather than from the
/// KV-cache representation.
pub(crate) fn dtype_name(mapping: QuantMapping) -> Result<&'static str, AicError> {
    match mapping.compute_dtype {
        Some(ComputeDtype::Bfloat16) => Ok("bf16"),
        Some(ComputeDtype::Fp8) => Ok("fp8"),
        Some(ComputeDtype::Fp4) => Err(AicError::InvalidEngineConfig(
            "the analytical attention model does not support FP4 matrix inputs".into(),
        )),
        Some(ComputeDtype::Int8) => Err(AicError::InvalidEngineConfig(
            "the analytical attention model does not support INT8 matrix inputs".into(),
        )),
        None => Err(AicError::InvalidEngineConfig(
            "analytical attention requires a compute dtype".into(),
        )),
    }
}

fn ceil_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}

fn floor_power_of_two(value: u32) -> u32 {
    if value == 0 {
        return 1;
    }
    1u32 << (31 - value.leading_zeros())
}

fn fa_parts(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    batch: u32,
    query_length: u32,
    kv_length: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    value_head_dim: u32,
    kv_storage_dim: u32,
    dtype: &str,
    causal: bool,
    kv_cache_bytes_per_token: Option<f64>,
    include_kv_cache_update: bool,
    profiled: bool,
    query_tile_l2_reuse: bool,
) -> Result<FaParts, AicError> {
    if batch == 0
        || query_length == 0
        || kv_length == 0
        || query_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_head_dim == 0
        || kv_storage_dim == 0
    {
        return Err(AicError::InvalidEngineConfig("analytical attention shape must be positive".into()));
    }
    if query_heads % kv_heads != 0 {
        return Err(AicError::InvalidEngineConfig("query_heads must be divisible by kv_heads".into()));
    }
    if causal && kv_length < query_length {
        return Err(AicError::InvalidEngineConfig("causal attention requires kv_length >= query_length".into()));
    }

    let input_bytes = dtype_bytes(dtype)?;
    let output_bytes = if input_bytes == 1.0 { 2.0 } else { input_bytes };
    let words = (spec.gpu.analytical.shared_memory_per_sm_bytes.unwrap_or(0) as f64 / input_bytes).floor() as u32;
    if words == 0 {
        return Err(AicError::MissingSystemFlops(
            "attention analytical model requires shared_memory_per_sm_bytes".into(),
        ));
    }
    let resident_width = head_dim.saturating_add(kv_storage_dim).saturating_add(value_head_dim).max(1);
    let capacity_bound = (words / resident_width).max(1);
    let automatic = floor_power_of_two(capacity_bound).min(128);
    let br = automatic;
    let bc = automatic;
    let history = kv_length - query_length;
    let mut score_elements_per_head = 0.0;
    let mut exact_scores_per_head = 0.0;
    let mut kv_tokens_loaded_per_head = 0.0;
    let mut row_tile_updates_per_head = 0.0;
    let mut active_kv_tiles_per_head = 0.0;

    let mut query_start = 0;
    while query_start < query_length {
        let rows = br.min(query_length - query_start);
        let exact_columns = if causal {
            kv_length.min(history.saturating_add(query_start).saturating_add(rows))
        } else {
            kv_length
        };
        if causal {
            let mut row = query_start;
            while row < query_start + rows {
                exact_scores_per_head += kv_length.min(history.saturating_add(row).saturating_add(1)) as f64;
                row += 1;
            }
        } else {
            exact_scores_per_head += (rows * kv_length) as f64;
        }
        let key_tiles = ceil_u32(exact_columns, bc);
        let loaded_columns = kv_length.min(key_tiles.saturating_mul(bc));
        score_elements_per_head += (rows * loaded_columns) as f64;
        kv_tokens_loaded_per_head += loaded_columns as f64;
        row_tile_updates_per_head += (rows * key_tiles) as f64;
        active_kv_tiles_per_head += key_tiles as f64;
        query_start += br;
    }
    let query_tiles = ceil_u32(query_length, br);
    let total_kv_tiles = ceil_u32(kv_length, bc);
    let max_by_work = (total_kv_tiles / 4).max(1);
    let split_limit = 128u32.min(total_kv_tiles).min(max_by_work).max(1);
    let base_ctas = batch as f64 * query_heads as f64 * query_tiles as f64;
    let splits_for_one_wave = ceil_f64(spec.gpu.analytical.sm_count.unwrap_or(1) as f64 / base_ctas.max(1.0));
    let kv_splits = if query_length > 16 || history == 0 {
        1.0
    } else {
        splits_for_one_wave.max(1.0).min(split_limit as f64)
    };
    let cta_count = base_ctas * kv_splits;
    let sm_count = spec.gpu.analytical.sm_count.unwrap_or(1) as f64;
    let parallel_efficiency = (cta_count / sm_count).min(1.0).max(1.0 / sm_count);
    let reduction_parallel_efficiency = (base_ctas / sm_count).min(1.0).max(1.0 / sm_count);

    let score_elements = batch as f64 * query_heads as f64 * score_elements_per_head;
    let exact_scores = batch as f64 * query_heads as f64 * exact_scores_per_head;
    let qk_flops = 2.0 * score_elements * head_dim as f64;
    let pv_flops = 2.0 * score_elements * value_head_dim as f64;
    let matrix_flops = qk_flops + pv_flops;
    let query_rows = batch as f64 * query_heads as f64 * query_length as f64;
    let row_updates = batch as f64 * query_heads as f64 * row_tile_updates_per_head;
    let exp_cost = spec.misc.exp_flop_equivalent;
    let q_scale_flops = kv_splits * query_rows * head_dim as f64;
    let score_softmax_flops = score_elements * (exp_cost + 3.0);
    let online_state_flops = row_updates * (exp_cost + 4.0 + value_head_dim as f64);
    let final_normalize_flops = kv_splits * query_rows * (exp_cost + 2.0 + value_head_dim as f64);
    let split_reduction_flops = if kv_splits > 1.0 {
        query_rows
            * (2.0 * (kv_splits - 1.0)
                + kv_splits * (exp_cost + 1.0)
                + exp_cost
                + 1.0
                + 2.0 * kv_splits * value_head_dim as f64)
    } else {
        0.0
    };
    let mainloop_vector_flops = score_softmax_flops + online_state_flops;
    let split_boundary_flops = q_scale_flops + final_normalize_flops;
    let vector_flops = mainloop_vector_flops + split_boundary_flops + split_reduction_flops;

    let query_elements = batch as f64 * query_heads as f64 * query_length as f64 * head_dim as f64;
    let output_elements = batch as f64 * query_heads as f64 * query_length as f64 * value_head_dim as f64;
    let kv_loaded = kv_tokens_loaded_per_head;
    let kv_hbm_loaded = if query_tile_l2_reuse { kv_length as f64 } else { kv_loaded };
    let kv_l2_tokens = batch as f64 * kv_heads as f64 * kv_loaded;
    let kv_hbm_tokens = batch as f64 * kv_heads as f64 * kv_hbm_loaded;
    let lse_bytes = 0.0;
    let q_bytes = kv_splits * query_elements * input_bytes;
    let cache_bytes = kv_cache_bytes_per_token.unwrap_or(kv_storage_dim as f64 * input_bytes);
    let kv_hbm_bytes = kv_hbm_tokens * cache_bytes;
    let output_hbm_bytes = output_elements * output_bytes;
    let partial_bytes = if kv_splits > 1.0 {
        kv_splits * query_rows * (value_head_dim as f64 + 1.0) * 4.0
    } else {
        0.0
    };
    let (mainloop_hbm_bytes, reduction_hbm_bytes, mainloop_l2_bytes, reduction_l2_bytes) = if kv_splits > 1.0 {
        (
            q_bytes + kv_hbm_bytes + partial_bytes,
            partial_bytes + output_hbm_bytes + lse_bytes,
            kv_splits * query_elements * input_bytes + kv_l2_tokens * cache_bytes + partial_bytes,
            partial_bytes + output_hbm_bytes + lse_bytes,
        )
    } else {
        (
            q_bytes + kv_hbm_bytes + output_hbm_bytes + lse_bytes,
            0.0,
            query_elements * input_bytes + kv_l2_tokens * cache_bytes + output_hbm_bytes + lse_bytes + lse_bytes,
            0.0,
        )
    };
    let live_kv_tokens = batch as f64 * kv_heads as f64 * query_length as f64;
    let live_update = if include_kv_cache_update { 2.0 * live_kv_tokens * cache_bytes } else { 0.0 };
    let hbm_bytes = mainloop_hbm_bytes + reduction_hbm_bytes;
    let l2_bytes = mainloop_l2_bytes + reduction_l2_bytes;
    let hbm_bw = positive(spec.gpu.mem_bw, "mem_bw")? * parallel_efficiency;
    let l2_bw = positive(spec.gpu.analytical.l2_bandwidth_bytes_s.unwrap_or(1.0), "l2_bandwidth_bytes_s")? * parallel_efficiency;
    let reduction_hbm_bw = positive(spec.gpu.mem_bw, "mem_bw")? * reduction_parallel_efficiency;
    let reduction_l2_bw = positive(spec.gpu.analytical.l2_bandwidth_bytes_s.unwrap_or(1.0), "l2_bandwidth_bytes_s")?
        * reduction_parallel_efficiency;
    let raw_hbm_s = mainloop_hbm_bytes / hbm_bw + reduction_hbm_bytes / reduction_hbm_bw + live_update / hbm_bw;
    let raw_l2_s = mainloop_l2_bytes / l2_bw + reduction_l2_bytes / reduction_l2_bw;
    let matrix_s = matrix_flops / (attention_peak(spec, dtype)? * parallel_efficiency);
    let vector_peak = positive(spec.gpu.analytical.vector_peak_flops.unwrap_or(1.0), "vector_peak_flops")?;
    let vector_s = (mainloop_vector_flops / (vector_peak * parallel_efficiency))
        + split_boundary_flops / (vector_peak * parallel_efficiency)
        + split_reduction_flops / (vector_peak * reduction_parallel_efficiency);
    let overlap = if config.attention_algorithm == "fa3" { 1.0 } else { 0.0 };
    let compute_s = matrix_s + vector_s - overlap * matrix_s.min(vector_s);
    let raw_bottleneck = if raw_hbm_s >= raw_l2_s && raw_hbm_s >= compute_s { 0 } else if raw_l2_s >= compute_s { 1 } else { 2 };
    let decode_service_active = history > 0 && query_length <= 16;
    let fractional_task_waves = if decode_service_active {
        batch as f64 * kv_heads as f64 * query_tiles as f64 * kv_splits / sm_count
    } else {
        0.0
    };
    let _ = (exact_scores, active_kv_tiles_per_head, hbm_bytes, l2_bytes, profiled);
    Ok(FaParts { raw_hbm_s, raw_l2_s, compute_s, raw_bottleneck, cta_count, fractional_task_waves })
}

fn ceil_f64(value: f64) -> f64 {
    value.ceil()
}

fn fa_reference(level: &str) -> (f64, f64, f64, f64, f64) {
    match level {
        "low" => (10.0, 0.92, 0.95, 0.80, 3500.0),
        "high" => (16.0, 0.50, 0.60, 0.35, 8000.0),
        _ => (12.5, 0.72, 0.80, 0.55, 6000.0),
    }
}

pub fn attention_model_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    batch: u32,
    query_length: u32,
    kv_length: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    value_head_dim: u32,
    kv_storage_dim: u32,
    dtype: &str,
    causal: bool,
    kv_cache_bytes_per_token: Option<f64>,
    include_kv_cache_update: bool,
) -> Result<f64, AicError> {
    let parts = fa_parts(
        spec, config, batch, query_length, kv_length, query_heads, kv_heads, head_dim, value_head_dim,
        kv_storage_dim, dtype, causal, kv_cache_bytes_per_token, include_kv_cache_update, true, true,
    )?;
    let (fixed_us, hbm_eta, l2_eta, compute_eta, task_cycles) = fa_reference(&config.level);
    let adjusted = [
        parts.raw_hbm_s / hbm_eta,
        parts.raw_l2_s / l2_eta,
        parts.compute_s / compute_eta,
    ];
    let resource_s = adjusted[parts.raw_bottleneck as usize].max(adjusted.iter().copied().fold(0.0, f64::max));
    let history = kv_length - query_length;
    let task_s = if history > 0 && query_length <= 16 {
        parts.fractional_task_waves * task_cycles / positive(spec.gpu.analytical.clock_hz.unwrap_or(1.0), "clock_hz")?
    } else {
        0.0
    };
    Ok((fixed_us * 1e-6 + resource_s + task_s) * 1000.0)
}

pub fn mla_model_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    phase: &str,
    batch: u32,
    query_length: u32,
    sequence_length: u32,
    local_heads: u32,
    dtype: &str,
) -> Result<f64, AicError> {
    let (q, kv_heads, head_dim, value_dim, storage_dim, include_update, rounded, cycles, efficiency) = match phase {
        "prefill" => (query_length.min(sequence_length), local_heads, 192, 128, 320, false, true, 0.0, match config.level.as_str() { "low" => 0.90, "high" => 0.50, _ => 0.70 }),
        "decode" => (1, 1, 576, 512, 576, true, false, match config.level.as_str() { "low" => 4500.0, "high" => 9000.0, _ => 7000.0 }, match config.level.as_str() { "low" => 0.95, "high" => 0.60, _ => 0.85 }),
        other => return Err(AicError::InvalidEngineConfig(format!("unsupported MLA phase {other:?}"))),
    };
    let parts = fa_parts(
        spec, config, batch, q, sequence_length, local_heads, kv_heads, head_dim, value_dim, storage_dim,
        dtype, true, None, include_update, false, true,
    )?;
    let raw_resource = parts.raw_hbm_s.max(parts.raw_l2_s).max(parts.compute_s);
    let waves = parts.cta_count / positive(spec.gpu.analytical.sm_count.unwrap_or(1) as f64, "sm_count")?;
    let wave_rounding = if rounded && waves > 1.0 { waves.ceil() / waves } else { 1.0 };
    let task_us = if phase == "decode" {
        parts.fractional_task_waves * cycles / positive(spec.gpu.analytical.clock_hz.unwrap_or(1.0), "clock_hz")? * 1e6
    } else { 0.0 };
    let fixed_us = match config.level.as_str() { "low" => 10.0, "high" => 16.0, _ => 12.5 };
    Ok((fixed_us + raw_resource * wave_rounding / efficiency * 1e6 + task_us) / 1000.0)
}

pub fn bmm_model_latency_ms(
    config: &AnalyticalConfig,
    num_tokens: u32,
    num_heads: u32,
    if_pre: bool,
    dtype: &str,
    peak_flops_s: f64,
    mem_bandwidth_bytes_s: f64,
) -> Result<f64, AicError> {
    if num_tokens == 0 || num_heads == 0 { return Err(AicError::InvalidEngineConfig("BMM shape must be positive".into())); }
    let (n, k) = if if_pre { (512.0, 128.0) } else { (128.0, 512.0) };
    let batch = num_heads as f64;
    let m = num_tokens as f64;
    let flops = 2.0 * batch * m * n * k;
    let (logical_bytes, floor_us, eta_mem) = match dtype.trim().to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => (2.0 * batch * m * (k + n) + 2.0 * batch * k * n, match config.level.as_str() { "low" => 3.0, "high" => 5.5, _ => 4.2 }, match config.level.as_str() { "low" => 0.80, "high" => 0.56, _ => 0.70 }),
        "fp8" | "float8" | "float8_e4m3fn" => {
            let sa = 4.0 * batch;
            let sb = 4.0 * batch * (n / 128.0).ceil() * (k / 128.0).ceil();
            (4.0 * batch * m * k + 2.0 * batch * m * n + 2.0 * sa + batch * k * n + sb, match config.level.as_str() { "low" => 5.5, "high" => 9.0, _ => 7.1 }, match config.level.as_str() { "low" => 0.28, "high" => 0.18, _ => 0.22 })
        }
        other => return Err(AicError::InvalidEngineConfig(format!("BMM dtype must be bf16 or fp8, got {other:?}"))),
    };
    let compute_eta = match config.level.as_str() { "low" => 0.81, "high" => 0.52, _ => 0.65 };
    let memory_s = logical_bytes / positive(mem_bandwidth_bytes_s, "mem_bandwidth_bytes_s")? / eta_mem;
    let compute_s = flops / positive(peak_flops_s, "peak_flops_s")? / compute_eta;
    Ok((floor_us * 1e-6 + memory_s.max(compute_s)) * 1000.0)
}

fn analytical_scale(level: &str, low: f64, standard: f64, high: f64) -> f64 {
    match level {
        "low" => low,
        "high" => high,
        _ => standard,
    }
}

pub fn index_mqa_latency_ms(
    layout: &str,
    dtype: &str,
    batch: u32,
    query_length: u32,
    context_length: u32,
    index_heads: u32,
    head_dim: u32,
    sm_count: u32,
    clock_hz: f64,
    fp8_peak_flops: Option<f64>,
    bf16_peak_flops: Option<f64>,
    mem_bw: f64,
    level: &str,
) -> Result<f64, AicError> {
    if batch == 0 || query_length == 0 || context_length == 0 || index_heads == 0 || head_dim == 0 || sm_count == 0 {
        return Err(AicError::InvalidEngineConfig("Index MQA shape and hardware values must be positive".into()));
    }
    if query_length > context_length {
        return Err(AicError::InvalidEngineConfig("Index MQA query_length must not exceed context_length".into()));
    }
    let layout = layout.trim().to_ascii_lowercase();
    if !matches!(layout.as_str(), "paged" | "ragged") {
        return Err(AicError::InvalidEngineConfig(format!("Index MQA layout must be paged or ragged, got {layout:?}")));
    }
    let clock = positive(clock_hz, "clock_hz")?;
    let bandwidth = positive(mem_bw, "mem_bw")?;
    let b = f64::from(batch);
    let q = f64::from(query_length);
    let ctx = f64::from(context_length);
    let heads = f64::from(index_heads);
    let dim = f64::from(head_dim);
    let sm = f64::from(sm_count);
    let (floor_us, cycles, service_atoms, chunks, valid_pairs, score_slots) = if layout == "paged" {
        let aligned = ((context_length + query_length - 1).div_ceil(64) * 64) as f64;
        (
            3.696_352_887_891_391_6,
            3.752_455_488_695_439,
            b * (ctx / 64.0).ceil(),
            1.0,
            b * (q * ctx + q * (q - 1.0) / 2.0),
            b * q * aligned,
        )
    } else {
        let rows = batch.saturating_mul(query_length);
        let total_k = batch.saturating_mul(context_length).max(1);
        let mut chunk_rows = (8_000_000u32 / total_k).max(1);
        chunk_rows = if rows <= 16 { rows } else { (chunk_rows / 16).max(1) * 16 };
        chunk_rows = chunk_rows.min(rows).max(1);
        let chunks = rows.div_ceil(chunk_rows) as f64;
        let mut critical = 0.0;
        let mut chunk_begin = 0;
        while chunk_begin < rows {
            let chunk_end = (chunk_begin + chunk_rows).min(rows);
            let mut blocks = Vec::new();
            let mut tile_begin = chunk_begin;
            while tile_begin < chunk_end {
                let tile_end = (tile_begin + 16).min(chunk_end);
                let mut min_start = u64::MAX;
                let mut max_end = 0u64;
                let mut row = tile_begin;
                while row < tile_end {
                    let request = row / query_length;
                    let local_query = row % query_length;
                    let start = u64::from(request) * u64::from(context_length);
                    let end = start + u64::from(context_length - query_length + 1 + local_query);
                    min_start = min_start.min(start);
                    max_end = max_end.max(end);
                    row += 1;
                }
                let span_blocks = ((max_end.saturating_sub(min_start)) as f64 / 64.0).ceil();
                let request_cap = (128.0 / heads).max(1.0);
                let one_request_blocks = (ctx / 64.0).ceil();
                blocks.push(span_blocks.min(one_request_blocks * request_cap));
                tile_begin += 16;
            }
            let max_blocks = blocks.iter().copied().fold(0.0, f64::max);
            let mean_blocks = blocks.iter().sum::<f64>() / sm;
            critical += max_blocks.max(mean_blocks);
            chunk_begin += chunk_rows;
        }
        (
            4.947_469_384_876_594_5,
            414.755_322_253_893_44,
            critical,
            chunks,
            b * (q * (ctx - q + 1.0) + q * (q - 1.0) / 2.0),
            b * q * b * ctx,
        )
    };
    let flops = valid_pairs * 2.0 * heads * dim;
    let fp8_bytes = b * q * heads * dim + b * q * heads * 4.0 + b * ctx * (dim + 4.0) + score_slots * 4.0;
    let dtype = dtype.trim().to_ascii_lowercase();
    let (modeled_bytes, resource_scale) = match dtype.as_str() {
        "fp8" | "float8" | "float8_e4m3fn" => {
            let peak = positive(
                fp8_peak_flops.ok_or_else(|| {
                    AicError::MissingSystemFlops("Index MQA FP8 mode requires fp8_tc_flops".into())
                })?,
                "fp8_tc_flops",
            )?;
            let reference_resource_us = (flops / peak).max(fp8_bytes / bandwidth) * 1e6;
            (fp8_bytes, 1.0_f64.max(reference_resource_us / reference_resource_us))
        }
        "bf16" | "bfloat16" => {
            let bf16_peak = positive(
                bf16_peak_flops.ok_or_else(|| {
                    AicError::MissingSystemFlops("Index MQA BF16 mode requires bfloat16_tc_flops".into())
                })?,
                "bfloat16_tc_flops",
            )?;
            let bf16_bytes = b * q * heads * dim * 2.0
                + b * q * heads * 4.0
                + b * ctx * dim * 2.0
                + score_slots * 4.0;
            let reference_peak = fp8_peak_flops.map(|v| positive(v, "fp8_tc_flops")).transpose()?.unwrap_or(2.0 * bf16_peak);
            let reference_resource_us = (flops / reference_peak).max(fp8_bytes / bandwidth) * 1e6;
            let bf16_resource_us = (flops / bf16_peak).max(bf16_bytes / bandwidth) * 1e6;
            (bf16_bytes, (bf16_resource_us / reference_resource_us).max(1.0))
        }
        other => {
            return Err(AicError::InvalidEngineConfig(format!(
                "Index MQA dtype must be fp8 or bf16, got {other:?}"
            )))
        }
    };
    let profile_scale = analytical_scale(level, 0.80, 1.0, 1.20);
    let floor = floor_us * chunks * profile_scale;
    let task = service_atoms * cycles / clock * 1e6 * profile_scale * resource_scale;
    let _ = modeled_bytes;
    Ok((floor + task) / 1000.0)
}

pub fn index_topk_latency_ms(
    layout: &str,
    batch: u32,
    query_length: u32,
    context_length: u32,
    topk: u32,
    mem_bw: f64,
    level: &str,
) -> Result<f64, AicError> {
    if batch == 0 || query_length == 0 || context_length == 0 || topk == 0 {
        return Err(AicError::InvalidEngineConfig("Index TopK shape values must be positive".into()));
    }
    let layout = layout.trim().to_ascii_lowercase();
    if !matches!(layout.as_str(), "paged" | "ragged") {
        return Err(AicError::InvalidEngineConfig(format!("Index TopK layout must be paged or ragged, got {layout:?}")));
    }
    let bandwidth = positive(mem_bw, "mem_bw")?;
    let rows = batch.saturating_mul(query_length);
    let short = context_length <= topk;
    let tile = if short { 256 } else { 128 };
    let waves = rows.div_ceil(tile);
    let tail_waves = if short { 0 } else { waves.saturating_sub(4) };
    let (floor_us, inverse_efficiency, tail_inverse_efficiency) = match (short, layout.as_str()) {
        (true, "paged") => (3.223_162_275_792_814, 2.017_749_615_822_097_4e-6, 0.0),
        (true, "ragged") => (2.184_964_173_546_487_3, 1.278_894_000_372_432_5, 2.620_612_297_666_048),
        (false, "paged") => (6.885_035_995_748_586, 2.404_003_100_179_183, 0.0),
        (false, "ragged") => (7.005_540_416_937_35, 2.960_147_901_425_779_4, 2.620_612_297_666_048),
        _ => unreachable!(),
    };
    let executed_bytes = f64::from(waves) * f64::from(tile) * f64::from(context_length) * 4.0;
    let tail_bytes = f64::from(tail_waves) * f64::from(tile) * f64::from(context_length) * 4.0;
    let scale = analytical_scale(level, 0.75, 1.0, 1.35);
    Ok((floor_us + executed_bytes / bandwidth * inverse_efficiency * 1e6
        + tail_bytes / bandwidth * tail_inverse_efficiency * 1e6) * scale / 1000.0)
}

pub fn dsv4_topk_latency_ms(
    variant: &str,
    batch: u32,
    fresh_tokens: u32,
    prefix_tokens: u32,
    topk: u32,
    compression_ratio: u32,
    sm_count: u32,
    level: &str,
) -> Result<f64, AicError> {
    if batch == 0 || fresh_tokens == 0 || topk == 0 || compression_ratio == 0 || sm_count == 0 {
        return Err(AicError::InvalidEngineConfig("DeepSeek-V4 TopK shape values must be positive".into()));
    }
    let variant = variant.trim().to_ascii_lowercase();
    let context = ((prefix_tokens + fresh_tokens) / compression_ratio).max(1);
    let latency_us = match variant.as_str() {
        "v1" => {
            let first_active = (compression_ratio * (topk + 1)).saturating_sub(prefix_tokens).max(1);
            let active_per_request = fresh_tokens.saturating_sub(first_active).saturating_add(1);
            if context <= topk { 0.0 } else {
                let (floor, scan_ns, row_us, active_row_us, log_us) = if topk == 1024 {
                    (2.073_230_505_106_282_7, 0.539_315_304_942_849_4, 0.365_997_411_639_613_75, 0.0, 0.0)
                } else {
                    (1.887_679_677_785_294_9, 0.631_594_796_240_780_2, 0.304_705_083_985_064_63, 0.054_648_730_770_234_88, 0.132_818_865_802_338_6)
                };
                let per_request_scan = if active_per_request == 0 { 0.0 } else {
                    let sum_floor = |last: i64| -> i64 {
                        if last < 0 { return 0; }
                        let groups = last / i64::from(compression_ratio);
                        let remainder = last % i64::from(compression_ratio);
                        i64::from(compression_ratio) * groups * (groups - 1) / 2 + groups * (remainder + 1)
                    };
                    let lower = i64::from(prefix_tokens + first_active - 1);
                    let upper = i64::from(prefix_tokens + fresh_tokens);
                    (sum_floor(upper) - sum_floor(lower)) as f64
                };
                let active_rows = f64::from(batch) * f64::from(active_per_request);
                let critical_scan = if active_per_request == 0 { 0.0 } else { f64::from(context).max(f64::from(batch) * per_request_scan / f64::from(sm_count)) };
                floor + scan_ns * critical_scan / 1000.0 + row_us * f64::from(batch * fresh_tokens) / f64::from(sm_count)
                    + active_row_us * active_rows / f64::from(sm_count) + log_us * (f64::from(context) / f64::from(topk)).log2()
            }
        }
        "v2" => {
            if fresh_tokens != 1 { return Err(AicError::InvalidEngineConfig("DeepSeek-V4 TopK v2 requires fresh_tokens=1".into())); }
            if context <= topk { 0.0 } else {
                let batch_log = f64::from(batch).log2();
                if context <= 16_384 {
                    2.303_645_789_924_076 + 0.291_032_689_348_660_2 * (f64::from(context) / f64::from(topk)).log2() + 0.052_747_428_779_569_026 * batch_log
                } else if context <= 32_768 {
                    7.437_705_388_636_25 + 2.035_150_913_808_507_8 * (f64::from(context) / 16_384.0).log2() + 0.013_487_009_687_934_43 * batch_log
                } else {
                    8.014_170_795_757_238 + 2.004_893_480_458_116 * (f64::from(context) / 32_768.0).log2()
                        + 0.160_935_640_951_813_37 * batch_log + (f64::from(batch) / 15.0 - 1.0).max(0.0)
                }
            }
        }
        other => return Err(AicError::InvalidEngineConfig(format!("DeepSeek-V4 TopK variant must be v1 or v2, got {other:?}"))),
    };
    Ok(latency_us * analytical_scale(level, 0.75, 1.0, 1.30) / 1000.0)
}

pub fn msa_index_latency_ms(
    phase: &str,
    batch: u32,
    query_length: u32,
    context_length: u32,
    index_heads: u32,
    head_dim: u32,
    sm_count: u32,
    mem_bw: f64,
    level: &str,
) -> Result<f64, AicError> {
    if batch == 0 || query_length == 0 || context_length == 0 || index_heads == 0 || head_dim == 0 || sm_count == 0 {
        return Err(AicError::InvalidEngineConfig("MSA Index shape and hardware values must be positive".into()));
    }
    let bandwidth = positive(mem_bw, "mem_bw")?;
    let phase = phase.trim().to_ascii_lowercase();
    let (floor_us, eta, modeled_bytes) = match phase.as_str() {
        "prefill" => {
            let tasks = batch.saturating_mul(index_heads).saturating_mul(query_length.div_ceil(128));
            let waves = tasks.div_ceil(sm_count);
            (2.829_717_731_734_192, 0.580_859_671_591_291_4, f64::from(waves) * f64::from(context_length) * f64::from(head_dim) * 2.0 * f64::from(sm_count))
        }
        "decode" => {
            if query_length != 1 { return Err(AicError::InvalidEngineConfig("decode MSA Index requires query_length=1".into())); }
            let executed_heads = index_heads.next_power_of_two().max(16);
            let blocks = context_length.div_ceil(128);
            (3.824_147_994_300_949_3, 0.923_894_707_616_108_8, f64::from(batch) * f64::from(context_length) * f64::from(head_dim) * 2.0 + f64::from(batch) * f64::from(executed_heads) * f64::from(blocks) * 4.0)
        }
        other => return Err(AicError::InvalidEngineConfig(format!("MSA Index phase must be prefill or decode, got {other:?}"))),
    };
    Ok((floor_us + modeled_bytes / bandwidth / eta * 1e6) * analytical_scale(level, 0.85, 1.0, 1.25) / 1000.0)
}

pub fn dsa_sparse_attention_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    batch: u32,
    query_length: u32,
    selected_pairs: u64,
    local_heads: u32,
    qk_latent_dim: u32,
    value_latent_dim: u32,
    qk_nope_dim: u32,
    output_value_dim: u32,
) -> Result<f64, AicError> {
    if batch == 0 || query_length == 0 || selected_pairs == 0 || local_heads == 0 || qk_latent_dim == 0 || value_latent_dim == 0 || qk_nope_dim == 0 || output_value_dim == 0 {
        return Err(AicError::InvalidEngineConfig("DSA sparse attention shape values must be positive".into()));
    }
    let quantum = config.sparse_attention_head_quantum.unwrap_or(1);
    let executed_heads = if local_heads % quantum == 0 { local_heads } else if quantum % local_heads == 0 { quantum } else {
        return Err(AicError::InvalidEngineConfig("sparse attention heads are incompatible with the configured head quantum".into()));
    };
    let tokens = f64::from(batch) * f64::from(query_length);
    let pairs = selected_pairs as f64;
    let heads = f64::from(executed_heads);
    let qk = f64::from(qk_latent_dim);
    let value = f64::from(value_latent_dim);
    let flops = 2.0 * pairs * heads * (qk + value)
        + 2.0 * tokens * heads * value * f64::from(qk_nope_dim + output_value_dim);
    let average_selected = pairs / tokens;
    let bytes = tokens * heads * qk * 2.0 + f64::from(batch) * average_selected * qk * 2.0
        + tokens * heads * value * 2.0 + heads * value * f64::from(qk_nope_dim + output_value_dim) * 2.0;
    let (_, hbm_eta, _, compute_eta, _) = fa_reference(&config.level);
    let peak = named_peak(spec.gpu.bfloat16_tc_flops, "bfloat16_tc_flops")?;
    Ok(fa_reference(&config.level).0 / 1000.0 + (flops / (peak * compute_eta)).max(bytes / (positive(spec.gpu.mem_bw, "mem_bw")? * hbm_eta)) * 1000.0)
}

pub fn moe_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    mapping: QuantMapping,
    tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    moe_tp: u32,
    moe_ep: u32,
    _is_gated: bool,
) -> Result<f64, AicError> {
    if tokens == 0
        || hidden_size == 0
        || inter_size == 0
        || topk == 0
        || num_experts == 0
        || moe_tp == 0
        || moe_ep == 0
    {
        return Err(AicError::InvalidEngineConfig(
            "analytical MoE shape values must be positive".into(),
        ));
    }
    if inter_size % moe_tp != 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "inter_size must be divisible by moe_tp_size: {inter_size} % {moe_tp} != 0"
        )));
    }
    if num_experts % moe_ep != 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "num_experts must be divisible by moe_ep_size: {num_experts} % {moe_ep} != 0"
        )));
    }

    // These recipe names and traffic equations are the direct Rust port of
    // kernelsim/moe/model.py::work_terms.  `is_gated` is intentionally not a
    // multiplier: the collector boundary always accounts for the fused gate
    // and up projection as GEMM1, then the down projection as GEMM2.
    let recipe = match mapping.name {
        "bfloat16" => "bf16_triton",
        "int8_wo" => "w8a16_int8wo_bf16_transfer",
        "fp8" | "fp8_block" => "fp8_block_triton",
        "nvfp4" => "nvfp4_cutedsl",
        "w4a16_mxfp4" | "w4a16_mxfp4_cutlass" => "w4a16_mxfp4_bf16_transfer",
        "w4a8_mxfp4_mxfp8" | "w4a8_mxfp4_mxfp8_trtllm" => "nvfp4_cutedsl",
        unsupported => {
            return Err(AicError::InvalidEngineConfig(format!(
                "ANALYTICAL MoE does not support quant mode '{unsupported}'"
            )))
        }
    };

    let t = f64::from(tokens);
    let h = f64::from(hidden_size);
    let j = f64::from(inter_size / moe_tp);
    let assignments = t * f64::from(topk) / f64::from(moe_ep);
    let local_experts = f64::from(num_experts / moe_ep);
    let active = local_experts.min(assignments);
    let flops = 6.0 * assignments * h * j;
    let mut bytes_routing_logits = 0.0;
    let mut bytes_routing_topk = 0.0;
    let mut bytes_combine = 0.0;
    let mut bytes_input_quant = 0.0;
    let mut bytes_gemm1_input = 0.0;
    let mut bytes_gemm1_weight_values = 0.0;
    let mut bytes_gemm1_weight_scale = 0.0;
    let mut bytes_gemm1_output = 0.0;
    let mut bytes_activation_quant = 0.0;
    let mut bytes_gemm2_input = 0.0;
    let mut bytes_gemm2_weight_values = 0.0;
    let mut bytes_gemm2_weight_scale = 0.0;
    let mut bytes_gemm2_output = 0.0;
    let mut bytes_control = 0.0;

    if recipe != "nvfp4_cutedsl" {
        bytes_routing_logits = 6.0 * t * f64::from(num_experts);
        bytes_routing_topk = 32.0 * assignments;
        bytes_combine = if topk == 1 {
            0.0
        } else {
            2.0 * assignments * h + 2.0 * t * h
        };
    }

    match recipe {
        "bf16_triton" => {
            bytes_gemm1_input = 2.0 * assignments * h;
            bytes_gemm1_weight_values = 4.0 * active * h * j;
            bytes_gemm1_output = 4.0 * assignments * j;
            bytes_activation_quant = 6.0 * assignments * j;
            bytes_gemm2_input = 2.0 * assignments * j;
            bytes_gemm2_weight_values = 2.0 * active * h * j;
            bytes_gemm2_output = 2.0 * assignments * h;
        }
        "w8a16_int8wo_bf16_transfer" => {
            bytes_gemm1_input = 2.0 * assignments * h;
            bytes_gemm1_weight_values = 2.0 * active * h * j;
            bytes_gemm1_weight_scale = 16.0 * active * j;
            bytes_gemm1_output = 4.0 * assignments * j;
            bytes_activation_quant = 6.0 * assignments * j;
            bytes_gemm2_input = 2.0 * assignments * j;
            bytes_gemm2_weight_values = active * h * j;
            bytes_gemm2_weight_scale = 4.0 * active * h;
            bytes_gemm2_output = 2.0 * assignments * h;
        }
        "w4a16_mxfp4_bf16_transfer" => {
            bytes_gemm1_input = 2.0 * assignments * h;
            bytes_gemm1_weight_values = active * h * j;
            bytes_gemm1_weight_scale = active * mxfp4_scale_bytes(2.0 * j, h);
            bytes_gemm1_output = 4.0 * assignments * j;
            bytes_activation_quant = 6.0 * assignments * j;
            bytes_gemm2_input = 2.0 * assignments * j;
            bytes_gemm2_weight_values = 0.5 * active * h * j;
            bytes_gemm2_weight_scale = active * mxfp4_scale_bytes(h, j);
            bytes_gemm2_output = 2.0 * assignments * h;
        }
        "fp8_block_triton" => {
            let h_blocks = (h / 128.0).ceil();
            let j_blocks = (j / 128.0).ceil();
            bytes_input_quant = 3.0 * t * h + 4.0 * t * h_blocks;
            bytes_gemm1_input = assignments * h + 4.0 * assignments * h_blocks;
            bytes_gemm1_weight_values = 2.0 * active * h * j;
            bytes_gemm1_weight_scale = 4.0 * active * (2.0 * j / 128.0).ceil() * h_blocks;
            bytes_gemm1_output = 4.0 * assignments * j;
            bytes_activation_quant = 9.0 * assignments * j + 4.0 * assignments * j_blocks;
            bytes_gemm2_input = assignments * j + 4.0 * assignments * j_blocks;
            bytes_gemm2_weight_values = active * h * j;
            bytes_gemm2_weight_scale = 4.0 * active * (h / 128.0).ceil() * j_blocks;
            bytes_gemm2_output = 2.0 * assignments * h;
        }
        "nvfp4_cutedsl" => {
            let h_scale_cols = (h / 16.0).ceil();
            let j_scale_cols = (j / 16.0).ceil();
            bytes_input_quant = 2.5 * assignments * h + assignments * h_scale_cols;
            bytes_gemm1_input = 0.5 * assignments * h + assignments * h_scale_cols;
            bytes_gemm1_weight_values = active * h * j;
            bytes_gemm1_weight_scale = active * nvfp4_scale_bytes(2.0 * j, h);
            bytes_gemm1_output = 4.0 * assignments * j;
            bytes_activation_quant = 4.5 * assignments * j + assignments * j_scale_cols;
            bytes_gemm2_input = 0.5 * assignments * j + assignments * j_scale_cols;
            bytes_gemm2_weight_values = 0.5 * active * h * j;
            bytes_gemm2_weight_scale = active * nvfp4_scale_bytes(h, j);
            bytes_gemm2_output = 2.0 * assignments * h;
            bytes_control = 16.0 * local_experts + 16.0 * active;
        }
        _ => unreachable!(),
    }

    let bytes = bytes_routing_logits
        + bytes_routing_topk
        + bytes_combine
        + bytes_input_quant
        + bytes_gemm1_input
        + bytes_gemm1_weight_values
        + bytes_gemm1_weight_scale
        + bytes_gemm1_output
        + bytes_activation_quant
        + bytes_gemm2_input
        + bytes_gemm2_weight_values
        + bytes_gemm2_weight_scale
        + bytes_gemm2_output
        + bytes_control;
    let (eta_compute, eta_memory, launch_us) = match (recipe, config.level.as_str()) {
        ("bf16_triton" | "w8a16_int8wo_bf16_transfer" | "w4a16_mxfp4_bf16_transfer", "low") => (1.0, 0.42, 30.0),
        ("bf16_triton" | "w8a16_int8wo_bf16_transfer" | "w4a16_mxfp4_bf16_transfer", "high") => (0.55, 0.17, 75.0),
        ("bf16_triton" | "w8a16_int8wo_bf16_transfer" | "w4a16_mxfp4_bf16_transfer", _) => (0.83, 0.27, 49.0),
        ("fp8_block_triton", "low") => (1.0, 1.0, 25.0),
        ("fp8_block_triton", "high") => (0.43, 0.47, 75.0),
        ("fp8_block_triton", _) => (0.70, 0.73, 46.0),
        ("nvfp4_cutedsl", "low") => (1.0, 0.90, 20.0),
        ("nvfp4_cutedsl", "high") => (0.45, 0.34, 75.0),
        ("nvfp4_cutedsl", _) => (0.76, 0.56, 42.5),
        _ => unreachable!(),
    };
    let peak = match recipe {
        "fp8_block_triton" => peak_for(
            spec,
            QuantMapping {
                memory: 1.0,
                compute: 2.0,
                name: "fp8_block",
                compute_dtype: Some(ComputeDtype::Fp8),
            },
        )?,
        "nvfp4_cutedsl" => named_peak(spec.gpu.fp4_tc_flops, "fp4_tc_flops")?,
        _ => named_peak(spec.gpu.bfloat16_tc_flops, "bfloat16_tc_flops")?,
    };
    let bandwidth = positive(spec.gpu.mem_bw, "mem_bw")?;
    let compute_ms = flops / peak / eta_compute * 1000.0;
    let memory_ms = bytes / bandwidth / eta_memory * 1000.0;
    Ok(launch_us / 1000.0 + compute_ms + memory_ms)
}

pub fn communication_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    quant: CommQuantMode,
    group_size: u32,
    elements: f64,
) -> Result<f64, AicError> {
    collective_latency_ms(spec, config, quant, group_size, elements, "all_reduce")
}

/// Table-free latency for an NCCL-style collective. `elements` is the logical
/// element count, matching the communication op APIs; dtype converts it to
/// wire bytes. Gather/scatter uses one ring transfer and all-reduce uses two.
pub fn collective_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    quant: CommQuantMode,
    group_size: u32,
    elements: f64,
    operation: &str,
) -> Result<f64, AicError> {
    if group_size <= 1 || elements <= 0.0 {
        return Ok(0.0);
    }
    let bw = spec.get_p2p_bandwidth(group_size).max(1.0);
    let element_bytes = match quant {
        CommQuantMode::Int8 | CommQuantMode::Fp8 => 1.0,
        CommQuantMode::Half => 2.0,
    };
    let wire_bytes = elements * element_bytes;
    let p = f64::from(group_size);
    let rounds = if operation == "all_reduce" { 2.0 } else { 1.0 };
    let launch_ms = config.profile().2 / 1000.0;
    Ok(rounds * wire_bytes * (p - 1.0) / p / bw * 1000.0
        + spec.node.p2p_latency * 1000.0
        + launch_ms)
}

/// Table-free point-to-point transfer latency. `bytes` is already a byte
/// count because P2P has no dtype axis in the SDK.
pub fn p2p_latency_ms(
    spec: &SystemSpec,
    config: &AnalyticalConfig,
    bytes: f64,
) -> Result<f64, AicError> {
    if bytes <= 0.0 {
        return Ok(0.0);
    }
    let bw = positive(spec.node.inter_node_bw, "inter_node_bw")?;
    Ok(bytes / bw * 1000.0 + spec.node.p2p_latency * 1000.0 + config.profile().2 / 1000.0)
}
