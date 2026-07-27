# Truth Strategy

Real-machine truth validates recorded AIC.  It is not a runtime dependency of
collector or materializer.

## Truth Manifest

Every frozen truth set should record:

```yaml
model:
backend:
sglang_version_or_commit:
sglang_package_version:
container_image:
runtime_image_digest:
hardware:
visible_gpus:
family:
phase:
ep_list:
eplb_list:
token_list:
model_loading:
layer_truncation:
truth_target_semantic:
sglang_instrumentation_files:
instrumentation_patch:
profiler:
profile_command:
parser:
parser_assumptions:
kernel_include_rules:
kernel_exclude_rules:
aggregation:
output_dir:
known_limitations:
```

## Family-Specific Guidance

Ordinary MoE:

- Align context/generation and token/EP semantics with public compact tables.
- Target routed expert compute.
- Exclude attention, shared expert, router/topk, dispatch/combine, and all-reduce
  when the truth target is compute-only.
- Torch profiler can be acceptable when its window reliably covers the compute
  range and cuda graph does not hide the target.

WideEP context:

- Confirm runtime really uses WideEP/DeepEP context path.
- Explicitly decide whether truth is routed compute only or a wider WideEP
  stage metric.
- If truth target is routed compute, dispatch/combine must be excluded.

WideEP generation:

- If cuda graph is enabled, prefer nsys semantic routed/compute truth.
- Do not use torch profiler windows that can include neighboring decode steps.
- Read `nsys-wideep-generation-truth.md` before selecting values.

## Freezing Truth

Freeze truth only after:

- the target semantic is written down;
- scripts and parser are archived;
- missing points are listed;
- stability/session behavior is checked;
- source directories and container paths are recorded.

Once frozen, use the same truth set for gate until a semantic bug or coverage
gap justifies a new frozen version.

## Runtime Version Changes

A SGLang/backend version change can invalidate the truth method even on the same
hardware:

- MoE Python path can move.
- WideEP/DeepEP dispatcher implementation can change.
- cuda graph capture/replay behavior can change.
- `/start_profile` request/step semantics can change.
- NVTX ranges can disappear from worker processes.
- kernel names or `graphNodeId` metadata can change.

When runtime version changes:

1. Record the old and new SGLang package version, commit, image, and Python path.
2. Re-locate MoE dispatch/compute/combine code.
3. Reapply or verify instrumentation.
4. Run a small truth smoke on representative ordinary and WideEP cases.
5. Verify parser evidence before reusing old frozen truth as a comparable gate.

Old truth may remain a baseline for the old runtime, but it does not prove the
new runtime has the same truth semantics.
