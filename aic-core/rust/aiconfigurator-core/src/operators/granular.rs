// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Table-free granular sparse-attention operators.
//!
//! These operators are the Rust wire/query counterparts of the Python
//! `dsa_granular.py` and DSV4 CP helper operations.  They intentionally use
//! only the system analytical model and runtime shape; they do not open
//! operator performance tables.

use crate::common::analytical;
use crate::common::enums::{CommQuantMode, DatabaseMode, FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::NcclOp;
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

fn analytical_source(db: &PerfDatabase) -> Source {
    if db.database_mode == DatabaseMode::Analytical {
        Source::Analytical
    } else {
        Source::Estimated
    }
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor.saturating_sub(1)) / divisor.max(1)
}

fn index_dtype(db: &PerfDatabase) -> &'static str {
    if db.system_spec.gpu.fp8_tc_flops.is_some() {
        "fp8"
    } else {
        "bf16"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaIndexScoreOp {
    pub name: String,
    pub scale_factor: f64,
    pub layout: String,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub cp_size: u32,
    pub context_stride: u32,
}

impl DsaIndexScoreOp {
    fn one_query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        query_length: u32,
        context_length: u32,
    ) -> Result<f64, AicError> {
        analytical::index_mqa_latency_ms(
            &self.layout,
            index_dtype(db),
            batch,
            query_length,
            context_length,
            self.index_heads,
            self.index_head_dim,
            db.system_spec.gpu.analytical.sm_count.unwrap_or(1),
            db.system_spec.gpu.analytical.clock_hz.unwrap_or(1.0),
            db.system_spec.gpu.fp8_tc_flops,
            db.system_spec.gpu.bfloat16_tc_flops,
            db.system_spec.gpu.mem_bw,
            &db.analytical_config.level,
        )
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let stride = self.context_stride.max(1);
        let full_context = if self.layout == "ragged" {
            prefix.saturating_add(sequence)
        } else {
            sequence
        };
        let indexed_context = (full_context / stride).max(1);
        if self.layout == "ragged" && indexed_context <= self.index_topk {
            return Ok(PerformanceResult::new(0.0, Source::Analytical));
        }
        let query_length = if self.layout == "ragged" {
            ceil_div(sequence, self.cp_size.max(1))
        } else {
            1
        };
        let latency = if self.layout == "ragged" && query_length > indexed_context {
            let chunks = query_length / indexed_context;
            let remainder = query_length % indexed_context;
            let mut total = f64::from(chunks) * self.one_query(db, batch, indexed_context, indexed_context)?;
            if remainder != 0 {
                total += self.one_query(db, batch, remainder, indexed_context)?;
            }
            total
        } else {
            self.one_query(db, batch, query_length, indexed_context)?
        };
        Ok(PerformanceResult::new(latency * self.scale_factor, analytical_source(db)))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaTopKSelectOp {
    pub name: String,
    pub scale_factor: f64,
    pub layout: String,
    pub index_topk: u32,
    pub cp_size: u32,
    pub context_stride: u32,
    pub kernel_recipe: String,
}

impl DsaTopKSelectOp {
    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let stride = self.context_stride.max(1);
        let full_context = if self.layout == "ragged" {
            prefix.saturating_add(sequence)
        } else {
            sequence
        };
        let indexed_context = (full_context / stride).max(1);
        if self.layout == "ragged" && indexed_context <= self.index_topk {
            return Ok(PerformanceResult::new(0.0, Source::Analytical));
        }
        let query_length = if self.layout == "ragged" {
            ceil_div(sequence, self.cp_size.max(1))
        } else {
            1
        };
        let latency = if self.kernel_recipe == "dsv4" {
            let (fresh, local_prefix, variant) = if self.layout == "ragged" {
                (query_length, prefix.saturating_add(sequence.saturating_sub(query_length)), "v1")
            } else {
                (1, sequence.saturating_sub(1), "v2")
            };
            analytical::dsv4_topk_latency_ms(
                variant,
                batch,
                fresh,
                local_prefix,
                self.index_topk,
                stride,
                db.system_spec.gpu.analytical.sm_count.unwrap_or(1),
                &db.analytical_config.level,
            )?
        } else {
            analytical::index_topk_latency_ms(
                &self.layout,
                batch,
                query_length,
                indexed_context,
                self.index_topk,
                db.system_spec.gpu.mem_bw,
                &db.analytical_config.level,
            )?
        };
        Ok(PerformanceResult::new(latency * self.scale_factor, analytical_source(db)))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaSparseAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub layout: String,
    pub local_heads: u32,
    pub index_topk: u32,
    pub qk_latent_dim: u32,
    pub value_latent_dim: u32,
    pub qk_nope_dim: u32,
    pub output_value_dim: u32,
    pub cp_size: u32,
}

impl DsaSparseAttentionOp {
    fn causal_pairs(batch: u32, query: u32, prefix: u32, limit: u32) -> u64 {
        let full = prefix.saturating_add(query);
        if prefix >= limit {
            return u64::from(batch) * u64::from(query) * u64::from(limit);
        }
        if full <= limit {
            return u64::from(batch)
                * u64::from(full * (full + 1) - prefix * (prefix + 1))
                / 2;
        }
        u64::from(batch) * u64::from(limit * (limit + 1) - prefix * (prefix + 1)) / 2
            + u64::from(batch) * u64::from(full - limit) * u64::from(limit)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        if batch == 0 || sequence == 0 {
            return Ok(PerformanceResult::new(0.0, Source::Analytical));
        }
        let (query, pairs) = if self.layout == "ragged" {
            let query = ceil_div(sequence, self.cp_size.max(1));
            let local_prefix = prefix.saturating_add(sequence.saturating_sub(query));
            (
                query,
                Self::causal_pairs(batch, query, local_prefix, self.index_topk),
            )
        } else {
            (1, u64::from(batch) * u64::from(sequence.min(self.index_topk)))
        };
        let latency = analytical::dsa_sparse_attention_latency_ms(
            &db.system_spec,
            &db.analytical_config,
            batch,
            query,
            pairs.max(1),
            self.local_heads,
            self.qk_latent_dim,
            self.value_latent_dim,
            self.qk_nope_dim,
            self.output_value_dim,
        )?;
        Ok(PerformanceResult::new(latency * self.scale_factor, analytical_source(db)))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4KvAllGatherOp {
    pub name: String,
    pub scale_factor: f64,
    pub kind: String,
    pub width: u32,
    pub cp_size: u32,
    pub window_size: u32,
    pub compress_ratio: u32,
}

impl Dsv4KvAllGatherOp {
    pub fn query(&self, db: &PerfDatabase, batch: u32, sequence: u32) -> Result<PerformanceResult, AicError> {
        let entries = match self.kind.as_str() {
            "window" => sequence.min(self.window_size),
            "compressed" => sequence / self.compress_ratio.max(1),
            "index" => sequence,
            other => return Err(AicError::InvalidEngineConfig(format!("unknown DSV4 KV all-gather kind {other:?}"))),
        };
        let elements = u64::from(batch) * u64::from(entries) * u64::from(self.width);
        let result = NcclOp {
            name: self.name.clone(),
            scale_factor: 1.0,
            hidden_size: elements as f64,
            num_gpus: self.cp_size,
            dtype: CommQuantMode::Half,
            operation: "all_gather".to_string(),
            seq_split: 1,
        }
        .query(db, 1)?;
        Ok(result.scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4SparseAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub layout: String,
    pub local_heads: u32,
    pub head_dim: u32,
    pub window_size: u32,
    pub compress_ratio: u32,
    pub index_topk: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub cp_size: u32,
}

impl Dsv4SparseAttentionOp {
    fn floor_sum(n: u32, divisor: u32) -> u64 {
        if n == 0 || divisor == 0 {
            return 0;
        }
        let groups = n / divisor;
        u64::from(divisor) * u64::from(groups) * u64::from(groups.saturating_sub(1)) / 2
            + u64::from(groups) * u64::from(n - groups * divisor + 1)
    }

    fn compressed_pairs(batch: u32, query: u32, prefix: u32, ratio: u32, limit: u32) -> u64 {
        let unclamped_queries = query.min(ratio.saturating_mul(limit).saturating_sub(1).saturating_sub(prefix));
        let unclamped = Self::floor_sum(prefix.saturating_add(unclamped_queries), ratio)
            .saturating_sub(Self::floor_sum(prefix, ratio));
        u64::from(batch)
            * (unclamped + u64::from(query.saturating_sub(unclamped_queries)) * u64::from(limit))
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        if batch == 0 || sequence == 0 {
            return Ok(PerformanceResult::new(0.0, Source::Analytical));
        }
        let (query, pairs) = if self.layout == "ragged" {
            let query = ceil_div(sequence, self.cp_size.max(1));
            let local_prefix = prefix.saturating_add(sequence.saturating_sub(query));
            let mut pairs = DsaSparseAttentionOp::causal_pairs(batch, query, local_prefix, self.window_size);
            if self.compress_ratio != 0 {
                let limit = if self.compress_ratio == 4 { self.index_topk } else { u32::MAX / 2 };
                pairs += Self::compressed_pairs(batch, query, local_prefix, self.compress_ratio, limit);
            }
            (query, pairs)
        } else {
            let mut pairs = u64::from(batch) * u64::from(sequence.min(self.window_size));
            if self.compress_ratio != 0 {
                let compressed = (sequence / self.compress_ratio).min(if self.compress_ratio == 4 { self.index_topk } else { u32::MAX });
                pairs += u64::from(batch) * u64::from(compressed);
            }
            (1, pairs)
        };
        let effective_kv = ceil_div((pairs / u64::from(batch.max(1) * query.max(1))) as u32, 1).max(1);
        let quantum = db.analytical_config.sparse_attention_head_quantum.unwrap_or(1);
        let executed_heads = if self.local_heads % quantum == 0 {
            self.local_heads
        } else if quantum % self.local_heads == 0 {
            quantum
        } else {
            return Err(AicError::InvalidEngineConfig("sparse attention heads are incompatible with the configured head quantum".into()));
        };
        let latency = analytical::attention_model_latency_ms(
            &db.system_spec,
            &db.analytical_config,
            batch,
            query,
            effective_kv,
            executed_heads,
            1,
            self.head_dim,
            self.head_dim,
            self.head_dim,
            "bf16",
            false,
            Some(584.0),
            false,
        )?;
        Ok(PerformanceResult::new(latency * self.scale_factor, analytical_source(db)))
    }
}
