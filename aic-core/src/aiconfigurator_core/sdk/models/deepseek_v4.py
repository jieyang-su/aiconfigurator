# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from typing import ClassVar

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor


def _dsv4_attention_granular_ops(
    *,
    phase: str,
    scale_factor: float,
    local_heads: int,
    hidden_size: int,
    q_lora_rank: int,
    o_lora_rank: int,
    head_dim: int,
    rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    window_size: int,
    compress_ratio: int,
    local_o_groups: int,
    kvcache_quant_mode,
    fmha_quant_mode,
    gemm_quant_mode,
    cp_size: int = 1,
) -> list:
    """Build the SGLang V4 attention boundary from existing analytical ops."""
    is_context = phase == "context"
    layout = "ragged" if is_context else "paged"
    seq_split = cp_size if is_context else 1
    ops_list = [
        ops.GEMM(
            f"{phase}_dsv4_q_a_proj_c{compress_ratio}",
            scale_factor,
            q_lora_rank,
            hidden_size,
            gemm_quant_mode,
            seq_split=seq_split,
        ),
        ops.GEMM(
            f"{phase}_dsv4_wkv_proj_c{compress_ratio}",
            scale_factor,
            head_dim,
            hidden_size,
            gemm_quant_mode,
            seq_split=seq_split,
        ),
        ops.ElementWise(
            f"{phase}_dsv4_q_kv_norm_c{compress_ratio}",
            scale_factor,
            q_lora_rank + head_dim,
            q_lora_rank + head_dim,
            0.8,
            seq_split=seq_split,
        ),
        ops.GEMM(
            f"{phase}_dsv4_q_b_proj_c{compress_ratio}",
            scale_factor,
            local_heads * head_dim,
            q_lora_rank,
            gemm_quant_mode,
            seq_split=seq_split,
        ),
    ]

    if compress_ratio:
        compressor_mult = 2 if compress_ratio == 4 else 1
        for kind in ("kv", "gate"):
            ops_list.append(
                ops.GEMM(
                    f"{phase}_dsv4_main_compressor_{kind}_c{compress_ratio}",
                    scale_factor,
                    compressor_mult * head_dim,
                    hidden_size,
                    gemm_quant_mode,
                    seq_split=seq_split,
                )
            )
        ops_list.append(
            ops.ElementWise(
                f"{phase}_dsv4_main_compress_store_c{compress_ratio}",
                scale_factor,
                compressor_mult * compress_ratio * head_dim,
                head_dim + common.deepseek_v4_indexer_cache_entry_bytes(head_dim),
                0.65,
                seq_split=seq_split,
            )
        )

    if compress_ratio == 4:
        # The V4 indexer has an independent overlap compressor and projection
        # path. Its score/TopK operate on the c4 cache, hence context_stride=4.
        ops_list.extend(
            [
                ops.GEMM(
                    f"{phase}_dsv4_index_q_proj",
                    scale_factor,
                    index_n_heads * index_head_dim,
                    q_lora_rank,
                    gemm_quant_mode,
                    seq_split=seq_split,
                ),
                ops.GEMM(
                    f"{phase}_dsv4_index_weight_proj",
                    scale_factor,
                    index_n_heads,
                    hidden_size,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=seq_split,
                ),
                ops.GEMM(
                    f"{phase}_dsv4_index_compressor_kv",
                    scale_factor,
                    2 * index_head_dim,
                    hidden_size,
                    gemm_quant_mode,
                    seq_split=seq_split,
                ),
                ops.GEMM(
                    f"{phase}_dsv4_index_compressor_gate",
                    scale_factor,
                    2 * index_head_dim,
                    hidden_size,
                    gemm_quant_mode,
                    seq_split=seq_split,
                ),
                ops.ElementWise(
                    f"{phase}_dsv4_index_norm_rope_quant",
                    scale_factor,
                    index_n_heads * index_head_dim + 2 * 4 * index_head_dim,
                    index_n_heads * index_head_dim + common.deepseek_v4_indexer_cache_entry_bytes(index_head_dim),
                    0.65,
                    seq_split=seq_split,
                ),
                ops.DSAIndexScore(
                    f"{phase}_dsv4_index_score",
                    scale_factor,
                    layout=layout,
                    index_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    index_topk=index_topk,
                    cp_size=cp_size,
                    context_stride=4,
                ),
                ops.DSATopKSelect(
                    f"{phase}_dsv4_topk",
                    scale_factor,
                    layout=layout,
                    index_topk=index_topk,
                    cp_size=cp_size,
                    context_stride=4,
                    kernel_recipe="dsv4",
                ),
            ]
        )

    if is_context and cp_size > 1:
        if compress_ratio == 4:
            ops_list.append(
                ops.DeepSeekV4KVAllGather(
                    f"{phase}_dsv4_index_k_all_gather",
                    scale_factor,
                    kind="index",
                    width=index_head_dim,
                    cp_size=cp_size,
                )
            )
        else:
            ops_list.append(
                ops.DeepSeekV4KVAllGather(
                    f"{phase}_dsv4_window_kv_all_gather_c{compress_ratio}",
                    scale_factor,
                    kind="window",
                    width=head_dim,
                    cp_size=cp_size,
                    window_size=window_size,
                )
            )
        if compress_ratio:
            ops_list.append(
                ops.DeepSeekV4KVAllGather(
                    f"{phase}_dsv4_compressed_kv_all_gather_c{compress_ratio}",
                    scale_factor,
                    kind="compressed",
                    width=head_dim,
                    cp_size=cp_size,
                    compress_ratio=compress_ratio,
                )
            )

    ops_list.extend(
        [
            ops.DeepSeekV4SparseAttention(
                f"{phase}_dsv4_attention_core_c{compress_ratio}",
                scale_factor,
                layout=layout,
                local_heads=local_heads,
                head_dim=head_dim,
                window_size=window_size,
                compress_ratio=compress_ratio,
                index_topk=index_topk,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                cp_size=cp_size,
            ),
            ops.ElementWise(
                f"{phase}_dsv4_output_rope_c{compress_ratio}",
                scale_factor,
                local_heads * rope_head_dim,
                local_heads * rope_head_dim,
                0.8,
                seq_split=seq_split,
            ),
            # wo_a is grouped einsum. Flattening its independent groups into
            # GEMM preserves FLOPs and weight bytes for the analytical model.
            ops.GEMM(
                f"{phase}_dsv4_wo_a_c{compress_ratio}",
                scale_factor,
                local_o_groups * o_lora_rank,
                local_heads * head_dim // local_o_groups,
                common.GEMMQuantMode.bfloat16,
                seq_split=seq_split,
            ),
            ops.GEMM(
                f"{phase}_dsv4_wo_b_c{compress_ratio}",
                scale_factor,
                hidden_size,
                local_o_groups * o_lora_rank,
                gemm_quant_mode,
                seq_split=seq_split,
            ),
        ]
    )
    return ops_list


def _dsv4_attention_with_granular(
    *,
    is_context: bool,
    name: str,
    scale_factor: float,
    num_heads: int,
    native_heads: int,
    tp_size: int,
    hidden_size: int,
    q_lora_rank: int,
    o_lora_rank: int,
    head_dim: int,
    rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    window_size: int,
    compress_ratio: int,
    o_groups: int,
    kvcache_quant_mode,
    fmha_quant_mode,
    gemm_quant_mode,
    cp_size: int = 1,
    silicon_compress_ratio: int | None = None,
):
    phase = "context" if is_context else "generation"
    op_cls = ops.ContextDeepSeekV4AttentionModule if is_context else ops.GenerationDeepSeekV4AttentionModule
    primary_kwargs = dict(
        num_heads=num_heads,
        native_heads=native_heads,
        tp_size=tp_size,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        window_size=window_size,
        compress_ratio=(compress_ratio if silicon_compress_ratio is None else silicon_compress_ratio),
        o_groups=o_groups,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        gemm_quant_mode=gemm_quant_mode,
    )
    if is_context:
        primary_kwargs["cp_size"] = cp_size
    primary = op_cls(name, scale_factor, **primary_kwargs)
    fallback = _dsv4_attention_granular_ops(
        phase=phase,
        scale_factor=scale_factor,
        local_heads=num_heads,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        window_size=window_size,
        compress_ratio=compress_ratio,
        local_o_groups=o_groups,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        gemm_quant_mode=gemm_quant_mode,
        cp_size=cp_size,
    )
    wrapper = ops.FallbackOp(
        name,
        primary=primary,
        fallback=fallback,
        primary_excluded_modes=(common.DatabaseMode.ANALYTICAL,),
    )
    # FallbackOp delegates scaling to children, but model/SDK callers inspect
    # this metadata to recover the number of represented layers.
    wrapper._scale_factor = scale_factor
    wrapper._gemm_quant_mode = gemm_quant_mode
    wrapper._compress_ratio = compress_ratio
    return wrapper


@register_model("DEEPSEEKV4")
class DeepSeekV4Model(BaseModel):
    """DeepSeek-V4 model with mHC plus SWA/CSA/HCA compressed attention."""

    _SUPPORTED_COMPRESS_RATIOS: ClassVar[set[int]] = {0, 4, 128}

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # DeepSeek-V4 CSA/HCA prefill CP: SGLang AllGather only. CP is modeled
        # INSIDE the engine's ContextDeepSeekV4AttentionModule operator
        # (operators/dsv4.rs: GLM-5-style mqa full/cp + topk full/cp deltas;
        # HCA adds a windowed-KV all-gather), NOT via the dense
        # _cp_attn_comm_ops / seq_split-only skeleton.
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            model_info["extra_params"],
            backend_name=backend_name,
        )

    @property
    def activation_hidden_size(self) -> int:
        # DSv4 attention expands Q/O internals, but resident MoE/residual activations use hidden_size.
        return self._hidden_size

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)
        self._backend_name = backend_name

        if not isinstance(self.extra_params, common.DeepSeekV4Config):
            raise TypeError("DeepSeekV4Model requires DeepSeekV4Config extra_params")
        deepseek_v4_cfg = self.extra_params
        self._compress_ratios = deepseek_v4_cfg.compress_ratios
        unknown_ratios = set(self._compress_ratios) - self._SUPPORTED_COMPRESS_RATIOS
        if unknown_ratios:
            raise ValueError(f"Unsupported DeepSeek-V4 compress_ratios: {sorted(unknown_ratios)}")

        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should be equal to "
            f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._power_law_alpha = 1.01

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size
        # Context parallelism (sglang AllGather, prefill-only):
        #  - attention modules: cp_size on the module spec -> the engine's CP
        #    path (GLM-5-style mqa/topk full/cp deltas + CSA/HCA all-gathers);
        #  - token-major context ops (Embedding/MHC/norm/GEMM): seq_split=cp;
        #  - context MoEDispatch: attn_cp_size=cp (AG_hidden+RS comm), MoE compute
        #    cp-invariant. Generation/decode is NOT CP'd.
        cp = self.config.cp_size
        moe_backend = self.config.moe_backend
        use_megamoe = moe_backend == "megamoe"
        if use_megamoe:
            if backend_name != common.BackendName.sglang.value:
                raise ValueError("DeepSeek-V4 MegaMoE modeling is only supported with the SGLang backend.")
            if moe_tp_size != 1:
                raise ValueError(f"DeepSeek-V4 MegaMoE requires moe_tp_size=1, got {moe_tp_size}.")
            if moe_ep_size <= 1:
                raise ValueError(f"DeepSeek-V4 MegaMoE requires moe_ep_size > 1, got {moe_ep_size}.")

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        local_heads = self._num_heads // tp_size
        local_o_groups = max(1, deepseek_v4_cfg.o_groups // tp_size)
        local_moe_inter_size = self._moe_inter_size // tp_size

        def _attention_ops(is_context: bool, scale_factor: float):
            ratio_counts = Counter(self._compress_ratios)
            name = "context_attention" if is_context else "generation_attention"
            return [
                _dsv4_attention_with_granular(
                    is_context=is_context,
                    name=name,
                    scale_factor=count * scale_factor,
                    num_heads=local_heads,
                    native_heads=self._num_heads,
                    tp_size=tp_size,
                    hidden_size=h,
                    q_lora_rank=deepseek_v4_cfg.q_lora_rank,
                    o_lora_rank=deepseek_v4_cfg.o_lora_rank,
                    head_dim=deepseek_v4_cfg.head_dim,
                    rope_head_dim=deepseek_v4_cfg.qk_rope_head_dim,
                    index_n_heads=deepseek_v4_cfg.index_n_heads,
                    index_head_dim=deepseek_v4_cfg.index_head_dim,
                    index_topk=deepseek_v4_cfg.index_topk,
                    window_size=deepseek_v4_cfg.sliding_window,
                    compress_ratio=ratio,
                    o_groups=local_o_groups,
                    kvcache_quant_mode=kvcache_quant_mode,
                    fmha_quant_mode=fmha_quant_mode,
                    gemm_quant_mode=gemm_quant_mode,
                    cp_size=(cp if is_context else 1),
                    # Keep the historical silicon approximation for pure SWA;
                    # ANALYTICAL always executes the true ratio-0 fallback.
                    silicon_compress_ratio=(128 if ratio == 0 else ratio),
                )
                for ratio, count in ratio_counts.items()
                if count > 0
            ]

        def _moe_ops(phase: str, num_layers: float, is_context: bool, attn_cp: int = 1):
            # attn_cp>1 (context under CP) makes MoEDispatch use the attn-CP+moe-TP
            # comm pattern (pre=all_gather, post=reduce_scatter) instead of all_reduce.
            # MoE expert compute is cp-invariant (A2A globalises tokens) -> no change.
            if use_megamoe:
                return [
                    ops.DeepSeekV4MegaMoEModule(
                        f"{phase}_megamoe",
                        num_layers,
                        h,
                        self._moe_inter_size,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        moe_quant_mode,
                        workload_distribution,
                        is_context=is_context,
                    )
                ]
            return [
                ops.MoEDispatch(
                    f"{phase}_moe_pre_dispatch",
                    num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    quant_mode=moe_quant_mode,
                    attn_cp_size=attn_cp,
                    backend=self._backend_name,
                ),
                ops.MoE(
                    f"{phase}_moe",
                    num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                ),
                ops.MoEDispatch(
                    f"{phase}_moe_post_dispatch",
                    num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                    quant_mode=moe_quant_mode,
                    attn_cp_size=attn_cp,
                    backend=self._backend_name,
                ),
            ]

        context_moe_ops = _moe_ops("context", self._num_layers, is_context=True, attn_cp=cp)
        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3, seq_split=cp),
                ops.DeepSeekV4MHCModule(
                    "context_mhc_pre",
                    self._num_layers,
                    "pre",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                    architecture=self.architecture,
                ),
                ops.ElementWise("context_attn_norm", self._num_layers, h, h, 0.8, seq_split=cp),
                *_attention_ops(is_context=True, scale_factor=1.0),
                ops.DeepSeekV4MHCModule(
                    "context_mhc_post",
                    self._num_layers,
                    "post",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                    architecture=self.architecture,
                ),
                ops.ElementWise("context_ffn_norm", self._num_layers, h, h, 0.8, seq_split=cp),
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * local_moe_inter_size,
                    h,
                    gemm_quant_mode,
                    seq_split=cp,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * local_moe_inter_size,
                    local_moe_inter_size,
                    0.8,
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm", self._num_layers, h, local_moe_inter_size, gemm_quant_mode, seq_split=cp
                ),
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                ),
                *context_moe_ops,
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                ),
            ]
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.DeepSeekV4MHCModule(
                    "generation_mhc_pre",
                    self._num_layers * self._mtp_scale_factor,
                    "pre",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                    architecture=self.architecture,
                ),
                ops.ElementWise("generation_attn_norm", self._num_layers * self._mtp_scale_factor, h, h, 0.8),
                *_attention_ops(is_context=False, scale_factor=self._mtp_scale_factor),
                ops.DeepSeekV4MHCModule(
                    "generation_mhc_post",
                    self._num_layers * self._mtp_scale_factor,
                    "post",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                    architecture=self.architecture,
                ),
                ops.ElementWise("generation_ffn_norm", self._num_layers * self._mtp_scale_factor, h, h, 0.8),
            ]
        )

        gen_shared_ops = [
            ops.GEMM(
                "generation_shared_gate_up_gemm",
                self._num_layers * self._mtp_scale_factor,
                2 * local_moe_inter_size,
                h,
                gemm_quant_mode,
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                self._num_layers * self._mtp_scale_factor,
                2 * local_moe_inter_size,
                local_moe_inter_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                self._num_layers * self._mtp_scale_factor,
                h,
                local_moe_inter_size,
                gemm_quant_mode,
            ),
        ]
        generation_moe_ops = _moe_ops(
            "generation",
            self._num_layers * self._mtp_scale_factor,
            is_context=False,
            attn_cp=cp,
        )
        gen_routed_ops = [
            ops.GEMM(
                "generation_router_gemm",
                self._num_layers * self._mtp_scale_factor,
                self._num_experts,
                h,
                common.GEMMQuantMode.bfloat16,
            ),
            *generation_moe_ops,
        ]
        self.generation_ops.append(
            ops.OverlapOp("generation_moe_overlap", group_a=gen_routed_ops, group_b=gen_shared_ops)
        )
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * self._mtp_scale_factor,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        deepseek_v4_cfg = self.extra_params
        seq_len = max(0, seq_len)
        total = 0.0
        cache_entry_bytes = deepseek_v4_cfg.head_dim * self.config.kvcache_quant_mode.value.memory
        for ratio in self._compress_ratios:
            total += min(seq_len, deepseek_v4_cfg.sliding_window) * cache_entry_bytes
            if ratio:
                compressed_entries = seq_len // ratio
                total += compressed_entries * cache_entry_bytes
                coff = 2 if ratio == 4 else 1
                # Compressor decode state keeps FP32 kv_state and score_state buffers.
                total += 2 * ratio * coff * deepseek_v4_cfg.head_dim * 4
                if ratio == 4:
                    total += compressed_entries * common.deepseek_v4_indexer_cache_entry_bytes(
                        deepseek_v4_cfg.index_head_dim
                    )
                    # CSA has a second FP4 indexer compressor with its own decode state.
                    total += 2 * ratio * 2 * deepseek_v4_cfg.index_head_dim * 4
        return total

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over the window-capped + compressed KV curve (non-linear)."""
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
