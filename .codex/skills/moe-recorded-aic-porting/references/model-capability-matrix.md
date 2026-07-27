# Model Capability Matrix

Before changing collector code, build a model-specific MoE capability matrix.
Do not start from the DeepSeek-V3 four-family grid by default.

## Required Checks

Read model config and backend code to answer:

- Is the model MoE?
- Routed expert count and local expert partition rules.
- Top-k, routing groups, correction bias, and routing normalization.
- Shared expert presence and whether it is fused with routed experts.
- MoE layer range.
- Context/prefill path uses MoE: yes/no.
- Generation/decode path uses MoE: yes/no.
- EP support and EP size range.
- WideEP/DeepEP support: yes/no.
- EPLB support: yes/no/optional.
- CUDA graph behavior for generation.
- Whether AIC can use single-card EP simulation.
- Backend/runtime version and container image.
- Actual SGLang MoE class/function path used by this model.

## Family Rules

Recorded families are model capabilities, not fixed outputs:

```text
ordinary_context + ordinary_generation:
  include together when ordinary MoE recorded is needed

wideep_context + wideep_generation:
  include together when model/backend supports WideEP or DeepEP recorded

EPLB:
  optional dimension, usually tied to WideEP/DeepEP backend behavior
```

Only split context/generation within a family if the model has a real exception.
Record that exception in the case matrix.

## Case Matrix Template

```yaml
model:
backend:
backend_version:
runtime_image:
model_config:
runtime_path:
single_card_ep_simulation: true
families:
  ordinary:
    enabled:
    phases: [context, generation]
    ep_sizes:
    eplb: none
    context_tokens:
    generation_tokens:
  wideep:
    enabled:
    phases: [context, generation]
    ep_sizes:
    eplb: [off, on]
    context_tokens:
    generation_tokens:
source_health_guard:
  enabled:
  small_token_threshold:
truth_gate_required:
```

## DeepSeek-V3 Example

DeepSeek-V3 is a reference, not a universal template:

```text
ordinary MoE: context + generation
WideEP/DeepEP MoE: context + generation
EPLB: WideEP/DeepEP variants
single-card EP simulation: enabled
generation cuda graph: enabled in WideEP generation truth runs
```
