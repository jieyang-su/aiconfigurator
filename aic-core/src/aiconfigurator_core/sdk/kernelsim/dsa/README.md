# DSA Index KernelSim Models

This package contains provisional models for the sparse-attention Index MQA
logits and TopK/index-transform kernels. It is intentionally separate from the
FlashAttention model because Index MQA materializes FP32 scores, performs
index-head gating/reduction, and uses DeepGEMM-specific ragged/paged scheduling.

The calibration scope is limited to one H100 SXM, SGLang 0.5.12, DeepGEMM FP8
MQA, index heads 32/64, head dim 128, FP32 score, int32 TopK output, and the
temporary collector boundary in `.self/dsa-index-kernel-modeling`. The models
have not been validated on other GPUs, backends, dtypes, head dimensions or
sparse-attention implementations. They are a provisional, semi-mature fallback
for architecture studies, not a replacement for silicon data.

`standard`, `high`, and `low` are central, conservative high-latency, and
optimistic low-latency engineering scenarios. They are not statistical
confidence intervals. Every estimate emits `DsaIndexModelWarning` and records
the same limitation in its result metadata.
