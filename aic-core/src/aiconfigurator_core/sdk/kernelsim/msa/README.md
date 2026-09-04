# MiniMax MSA KernelSim

This package contains the provisional MiniMax-M3 MSA Triton index-score model.
It is calibrated from one H100 SXM using BF16, `head_dim=128`, and
`block_size=128`. The score kernel includes the fused block-max reduction.

The model is separate from the DSA FP8 DeepGEMM-style Index MQA model. It is
not validated across GPUs, SGLang versions, backends, dtypes, or MSA variants.
The three parameter levels are engineering envelopes, not confidence
intervals. TopK/page-table transform is intentionally represented by a small
launch-aware ElementWise approximation in the MSA analytical path; it is not a
dedicated calibrated model. The selected-block GQA attention currently uses
the FA model as a provisional proxy.
