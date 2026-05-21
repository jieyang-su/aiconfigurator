# Phase 0 Experiment Contract (must-match for both topologies)

- model: `sgl-project/DeepSeek-V4-Flash-FP8`
- backend: `sglang`
- backend-version: `0.5.10.post2`
- quant: `fp8_block`
- SLA: `ttft=10000`, `tpot=5000`
- token shape: `isl=2048`, `osl=500`
- total-gpus: `32`
- database-mode: `HYBRID` (optional later SILICON replay for stricter reproducibility)

Only `--system` differs:
- scale-up: `rtx_pro_6000_scaleup_32`
- scale-out: `rtx_pro_6000_scaleout_2x16`
