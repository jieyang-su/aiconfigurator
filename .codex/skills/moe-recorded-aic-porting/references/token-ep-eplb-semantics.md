# Token, EP, And EPLB Semantics

Recorded AIC points must use the same semantic keys across collector, compact
tables, truth, and gate:

```text
family
phase
token
moe_ep_size
eplb state
distribution
```

## Token

Define whether a token field is global request token count, rank-local token
count, or replay-local token count.  Do not rely on column names alone.

Rules:

- Public compact tables should expose the lookup token semantic expected by
  AIC query code.
- `moe_token_distribution` source may contain intermediate local-token fields.
- Materializer must convert source fields into the public compact semantic.
- Truth must be keyed to the same public semantic before gate.

When adding left-endpoint points such as 1/2/4/16, verify both AIC and truth
use the same semantic.  A mismatch can look like a calibration failure.

## EP

Recorded AIC defaults to single-card EP simulation:

- EP changes local expert partition and rank-local replay shape.
- It is not real multi-card dispatch/combine communication unless the collector
  explicitly runs a multi-rank backend path.
- Gate and docs must state the simulation path.

For truth:

- EP should match the runtime TP/EP launch semantics.
- If hardware lacks enough visible GPUs for an EP, mark that point unverified
  instead of inventing or copying truth.

## EPLB

EPLB is optional:

- Do not add EPLB for ordinary-only models that do not expose it.
- For WideEP/DeepEP models, generate `recorded_no_eplb` and `recorded_eplb`
  only when both paths are supported and measurable.
- EP1 + EPLB can be semantically meaningless for some models; skip or document
  according to the model/backend contract.

