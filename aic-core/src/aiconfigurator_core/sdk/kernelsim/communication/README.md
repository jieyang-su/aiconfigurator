# Communication Analytical Proxy

This package provides the fixed communication model selected for table-free
Analytical simulation:

```text
latency_ms = 0.007 ms + sol_latency_ms / 0.75
```

The launch term and efficiency are engineering parameters informed primarily
by the packaged H100/H200 NCCL data. The underlying experiment and the more
detailed `(collective, rank)` candidate parameters remain under:

```text
.self/domestic-gpu/communication-startup-bandwidth-model/
```

The production model deliberately does not key parameters by hardware,
collective, rank, or dtype. The input SOL latency still captures:

- message bytes and communication dtype;
- the ring factor for each collective;
- the requested rank count;
- topology and placement-selected intra-node or inter-node bandwidth.

The fixed startup is not charged when SOL latency is zero, including a
collective group of size one.

## Selection

Use `database_mode=ANALYTICAL` together with:

```text
analytical_communication_mode=analytical
```

The other values retain their previous behavior:

- `empirical`: legacy formula-only `SOL / 0.8` path;
- `silicon`: measured communication table path where available.

Standalone `SOL`, `SOL_FULL`, `EMPIRICAL`, `HYBRID`, and `SILICON` database
modes are unchanged.

## Limitations

- This is not target-hardware Silicon data and must be reported with source
  `analytical`.
- The fixed parameters hide measured collective/rank variation. In the source
  data, rank-2 all-to-all and rank-8 all-reduce have materially different
  efficiencies.
- The calibration is dominated by Hopper intra-node measurements. Domestic
  collective runtimes, switch fabrics, launch stacks, and large-supernode
  algorithms remain unvalidated.
- Rank counts above the measured 2/4/8 range rely entirely on the existing SOL
  topology formula and the fixed efficiency; no hierarchy, congestion, or
  algorithm transition is fitted.
- Vendor-specific fused communication such as DeepEP and TRT-LLM all-to-all is
  represented only through its logical SOL byte volume when this mode is
  selected.
