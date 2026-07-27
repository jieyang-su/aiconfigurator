# Collector Source Contract

Collector source is raw measurement material.  It is not the final AIC latency.

## Source Types

Recorded MoE source can include:

- raw latency;
- rank-local replay latency;
- rank max/mean latency;
- rank spread;
- token distribution shape;
- expert/rank assignment shape;
- active experts;
- masked token counts;
- WideEP stage shape;
- backend/kernel metadata.

The exact fields are family-specific, but they must be explicit and stable.
They are also runtime-version-specific: a backend upgrade can rename columns,
change replay shape, or move a measured value from final output into a debug
source table.

## Source Contract Rules

- Source feature availability must not depend on whether debug/source files are
  kept on disk.
- no-keep and keep-source collector runs must produce the same final compact
  latency.
- Missing source columns should fail loudly or choose a documented fallback;
  do not silently switch semantics.
- Backend/SGLang version changes require a source schema audit before reusing
  an older materializer contract.
- Do not use server/profile truth paths as source input.
- Do not use path names such as `dense_refresh`, `parsed`, `profile`, or
  `rank_aggregate` as materializer inputs unless the code is a validation-only
  script.

## Source Health Guard

Source health guard is allowed when short kernels are unstable.  It can:

- repeat small-token source measurement;
- select the median complete source row;
- retry when spread is too high;
- record instability/audit metadata.

It must not:

- read truth;
- write platform-specific rules;
- directly set final latency to a truth value;
- select a partial row that lacks materializer inputs.

Default thresholds should live in model/backend defaults, not in a global rule
for all MoE models.

## Coverage Checks

After a run, check:

- expected compact files exist for enabled families;
- disabled families do not create fake empty outputs;
- `Total errors` is zero or every failure is understood;
- `moe_token_distribution_replay/manifest.csv` exists when replay is required;
- source rows cover all public case-matrix points.
