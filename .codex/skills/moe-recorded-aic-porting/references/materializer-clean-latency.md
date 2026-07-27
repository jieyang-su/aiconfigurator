# Materializer And Clean Latency

The materializer converts source rows into public compact AIC latency.

## Materializer Responsibilities

- Choose family/phase-specific logic.
- Use model semantics and source shape.
- Produce final `latency`.
- Preserve useful audit/source columns only when requested.
- Avoid hardware platform-name special cases.
- Never use real-machine truth as an input.

Model-specific logic is acceptable when it follows model structure:

- expert count;
- top-k;
- shared expert behavior;
- quant/kernel path;
- WideEP/DeepEP availability;
- context vs generation execution semantics.

Model-specific parameters are expected during porting, but they must remain
thin and semantic:

- read them from model config or backend source;
- document why they are model-specific;
- avoid pointwise truth fitting;
- keep the recorded source -> materializer -> final latency method intact.

## Clean Latency

Clean latency installs materialized final latency into top-level compact files:

```text
moe_perf.txt
moe_token_distribution_perf.txt
wideep_context_moe_perf.txt
wideep_generation_moe_perf.txt
```

WideEP files should exist only for models whose case matrix enables WideEP.

## Default Command Contract

The normal collector command should be enough for production data:

```bash
python3 collect.py --backend sglang --model-path <model> \
  --ops moe_token_distribution moe wideep_moe --keep-csv
```

For models without WideEP, omit `wideep_moe`.

Debug controls such as source retention may be environment variables, but they
must only control diagnostics on disk:

```text
KEEP_LATENCY_SOURCES=1 -> keep source/audit artifacts
CLEAN_LATENCY=1       -> run clean materialized output if not defaulted
```

The exact variable names can be model-specific.  The invariant is that final
latency is identical between no-keep and keep-source runs.
