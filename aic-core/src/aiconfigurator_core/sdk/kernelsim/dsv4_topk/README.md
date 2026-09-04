# DeepSeek V4 TopK KernelSim

This provisional model covers SGLang 0.5.14 DeepSeek-V4-Flash CSA TopK with
`topk=512/1024` and a c4 compressed candidate cache.

- v1: context/prefill radix TopK, modeled from causal row lengths and CTA waves.
- v2: normal decode planned TopK, split into register one-pass (`<=16K`),
  register two-pass (`<=32K`) and cluster (`>32K`) regimes.
- calibration: H100 SXM graph-replay `v1_top_last` and `v2_top_last` rows from
  `dsv4_csa_topk_calib_perf`.
- validation: H200 and B200/B300/GB200/GB300 tables.

Flash K=512 H100 shape-group OOF MAPE is 12.98% for v1 and 7.25% for v2. The v2 H100
profile transfers to the five validation platforms with 8.1-10.8% MAPE. V1
matches H200 but has 42-47% MAPE on Blackwell, so cross-generation prefill
estimates remain low confidence.

The repository tables contain only Flash K=512 rows. A targeted paired H100
CUDA-graph collection adds 23 Pro K=1024 v1 shapes from the launch boundary to
1M full context. Its dedicated v1 profile has 13.85% shape-group OOF MAPE. In
the paired v2 data, K=1024/K=512 has a 1.004 median latency ratio, so v2 shares
the K=512 parameters. Pro results have no H200/Blackwell validation and remain
provisional.

`standard` contains the fitted H100 parameters. `high=1.30x` and `low=0.75x`
are engineering envelopes, not confidence intervals. K=1024 emits a targeted
H100-only warning; other K values and score distributions are uncalibrated.
