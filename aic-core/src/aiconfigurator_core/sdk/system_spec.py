# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SystemSpec — hardware system spec loaded from a per-system YAML file.

Subclasses ``dict`` so existing code that does ``spec["gpu"]["mem_bw"]`` or
``isinstance(spec, dict)`` keeps working. ``get_p2p_bandwidth`` is the only
added method, replacing ``PerfDatabase._get_p2p_bandwidth``.
"""

from __future__ import annotations

from dataclasses import dataclass

_PLACEMENT_POLICIES = frozenset({"independent", "tp_first"})


@dataclass(frozen=True)
class ParallelLayout:
    """Small placement model for formula-based communication estimates.

    ``independent`` preserves the historical size-only decision. ``tp_first``
    lays out ranks with TP as the fastest-changing coordinate, followed by CP,
    attention-DP, and PP. It deliberately models locality only; NCCL
    algorithms, link contention, and hierarchical collectives remain outside
    this abstraction.
    """

    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    moe_tp: int = 1
    moe_ep: int = 1
    policy: str = "independent"

    def __post_init__(self) -> None:
        policy = self.policy.strip().lower()
        if policy not in _PLACEMENT_POLICIES:
            raise ValueError(f"communication placement must be one of {sorted(_PLACEMENT_POLICIES)}")
        object.__setattr__(self, "policy", policy)
        for name in ("tp", "pp", "dp", "cp", "moe_tp", "moe_ep"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"parallel dimension {name} must be positive, got {value}")
            object.__setattr__(self, name, value)

    @classmethod
    def from_model_config(cls, model_config, policy: str | None = None) -> ParallelLayout:
        return cls(
            tp=model_config.tp_size,
            pp=model_config.pp_size,
            dp=model_config.attention_dp_size,
            cp=model_config.cp_size,
            moe_tp=model_config.moe_tp_size or 1,
            moe_ep=model_config.moe_ep_size or 1,
            policy=policy or getattr(model_config, "communication_placement", "independent"),
        )

    @property
    def worker_gpus(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    @property
    def attention_width(self) -> int:
        return self.tp * self.dp * self.cp

    def _attention_rank(self, pp: int, dp: int, cp: int, tp: int) -> int:
        """Return a TP-first logical rank for one attention worker."""
        return (((pp * self.dp + dp) * self.cp + cp) * self.tp) + tp

    @staticmethod
    def _same_node(ranks: list[int], capacity: int) -> bool:
        return bool(ranks) and len({rank // capacity for rank in ranks}) == 1

    def group_is_local(self, group: str | None, capacity: int, group_size: int | None = None) -> bool:
        """Whether a logical communication group fits within one node."""
        if self.policy == "independent":
            if group_size is None:
                raise ValueError("group_size is required for independent placement")
            return group_size <= capacity
        if capacity < 1:
            raise ValueError(f"node capacity must be positive, got {capacity}")

        group = (group or "collective").lower()
        if group in {"tp", "attention_tp"}:
            ranks = [self._attention_rank(0, 0, 0, rank) for rank in range(self.tp)]
        elif group == "cp":
            ranks = [self._attention_rank(0, 0, rank, 0) for rank in range(self.cp)]
        elif group == "dp":
            ranks = [self._attention_rank(0, rank, 0, 0) for rank in range(self.dp)]
        elif group in {"attention", "tp_cp_dp", "tp_dp"}:
            ranks = [
                self._attention_rank(0, dp, cp, tp)
                for dp in range(self.dp)
                for cp in range(self.cp)
                for tp in range(self.tp)
            ]
        elif group == "moe_tp":
            # MoE TP is the local/fast dimension in the same logical worker
            # layout. Use the first expert-parallel slice as representative.
            ranks = list(range(self.moe_tp))
        elif group == "moe_ep":
            # EP advances after the MoE-TP block. This deliberately models
            # placement only; it does not emulate a backend's rank remapping.
            ranks = [ep * self.moe_tp for ep in range(self.moe_ep)]
        elif group in {"moe_tp_ep", "expert", "alltoall"}:
            # ModelConfig enforces moe_tp*moe_ep == attention width. The
            # complete MoE collective therefore spans the attention worker
            # group, whose TP coordinate is the fastest-changing one.
            ranks = [
                self._attention_rank(0, dp, cp, tp)
                for dp in range(self.dp)
                for cp in range(self.cp)
                for tp in range(self.tp)
            ]
        elif group == "pp":
            # One P2P operation connects exactly two adjacent stages. The
            # whole pipeline is not one collective. A pipeline estimate is on
            # the critical path, so use inter-node bandwidth if any adjacent
            # stage edge for any replicated TP/CP/DP lane crosses a node
            # boundary. This retains the two-participant P2P semantics while
            # avoiding a false local result from checking only lane zero.
            if self.pp <= 1:
                return True
            for stage in range(self.pp - 1):
                for dp in range(self.dp):
                    for cp in range(self.cp):
                        for tp in range(self.tp):
                            if not self._same_node(
                                [
                                    self._attention_rank(stage, dp, cp, tp),
                                    self._attention_rank(stage + 1, dp, cp, tp),
                                ],
                                capacity,
                            ):
                                return False
            return True
        else:
            if group_size is None:
                raise ValueError(f"unknown communication placement group {group!r}")
            ranks = list(range(group_size))
        return self._same_node(ranks, capacity)

    def bandwidth(self, system_spec: dict, group: str | None, group_size: int) -> float:
        """Select intra/inter bandwidth without changing the legacy policy."""
        node = system_spec["node"]
        capacity = int(node["num_gpus_per_node"])
        if self.policy == "independent":
            # Preserve the historical pipeline approximation: P2P used the
            # pipeline link directly (normally inter-node), even though the
            # communicating pair itself has only two participants. Other
            # collectives retain the historical size-only bandwidth helper.
            if (group or "").lower() == "pp":
                return get_pipeline_p2p_bandwidth(system_spec)
            return get_p2p_bandwidth(system_spec, group_size)
        if is_single_supernode(system_spec) and group_size > capacity:
            raise ValueError(
                f"Communication group of {group_size} GPUs exceeds single-supernode capacity {capacity}."
            )
        bandwidth = node["intra_node_bw"] if self.group_is_local(group, capacity, group_size) else node["inter_node_bw"]
        if bandwidth <= 0:
            raise ValueError(
                f"No positive bandwidth is configured for {group or 'collective'} communication "
                f"in {self.policy} placement (selected bandwidth={bandwidth})."
            )
        return bandwidth


def _gpu_spec(system_spec: dict) -> dict:
    return system_spec.get("gpu", {})


def architecture_family(system_spec: dict) -> str | None:
    """Return an explicitly declared hardware family, if any."""
    value = _gpu_spec(system_spec).get("architecture_family")
    return str(value).lower() if value is not None else None


def _capability(system_spec: dict, name: str, legacy_default: bool) -> bool:
    overrides = _gpu_spec(system_spec).get("capability_overrides", {}) or {}
    if name in overrides:
        return bool(overrides[name])
    return legacy_default


def is_blackwell_spec(system_spec: dict) -> bool:
    family = architecture_family(system_spec)
    if family is not None and family != "nvidia":
        return False
    return int(_gpu_spec(system_spec).get("sm_version", -1)) >= 100


def is_hopper_spec(system_spec: dict) -> bool:
    family = architecture_family(system_spec)
    if family is not None and family != "nvidia":
        return False
    return int(_gpu_spec(system_spec).get("sm_version", -1)) == 90


def is_sm100_spec(system_spec: dict) -> bool:
    family = architecture_family(system_spec)
    if family is not None and family != "nvidia":
        return False
    return int(_gpu_spec(system_spec).get("sm_version", -1)) == 100


def supports_fp8_mma(system_spec: dict) -> bool:
    sm_version = int(_gpu_spec(system_spec).get("sm_version", -1))
    return _capability(system_spec, "fp8_mma", sm_version >= 89)


def supports_fp4_mma(system_spec: dict) -> bool:
    sm_version = int(_gpu_spec(system_spec).get("sm_version", -1))
    return _capability(system_spec, "fp4_mma", sm_version >= 100)


def supports_mnnvl(system_spec: dict) -> bool:
    sm_version = int(_gpu_spec(system_spec).get("sm_version", -1))
    return _capability(system_spec, "mnnvl", sm_version >= 100)


def is_single_supernode(system_spec: dict) -> bool:
    return system_spec.get("node", {}).get("topology_scope") == "single_supernode"


def get_pipeline_p2p_bandwidth(system_spec: dict) -> float:
    node_spec = system_spec["node"]
    return node_spec["intra_node_bw"] if is_single_supernode(system_spec) else node_spec["inter_node_bw"]


def get_p2p_bandwidth(system_spec: dict, num_gpus: int) -> float:
    node_spec = system_spec["node"]
    num_gpus_per_node = node_spec["num_gpus_per_node"]
    if is_single_supernode(system_spec) and num_gpus > num_gpus_per_node:
        raise ValueError(
            f"Requested communication group of {num_gpus} GPUs exceeds single-supernode "
            f"capacity {num_gpus_per_node}."
        )
    if num_gpus <= num_gpus_per_node:
        return node_spec["intra_node_bw"]
    if num_gpus <= node_spec.get("num_gpus_per_rack", float("inf")):
        return node_spec["inter_node_bw"]
    return node_spec.get("inter_rack_bw", node_spec["inter_node_bw"])


def validate_parallelism(
    system_spec: dict,
    *,
    tp: int,
    pp: int,
    attention_dp: int,
    cp: int,
    moe_tp: int = 1,
    moe_ep: int = 1,
) -> None:
    """Validate one layout against an explicitly single-supernode spec."""
    node_spec = system_spec.get("node", {})
    if node_spec.get("topology_scope") != "single_supernode":
        return
    capacity = int(node_spec["num_gpus_per_node"])
    widths = {
        "worker GPUs (tp*pp*attention_dp*cp)": tp * pp * attention_dp * cp,
        "attention group (tp*attention_dp*cp)": tp * attention_dp * cp,
        "MoE group (moe_tp*moe_ep)": moe_tp * moe_ep,
    }
    exceeded = {name: value for name, value in widths.items() if value > capacity}
    if exceeded:
        details = ", ".join(f"{name}={value}" for name, value in exceeded.items())
        raise ValueError(
            f"Parallel configuration exceeds single-supernode capacity {capacity}: {details}. "
            "Cross-supernode communication is not modeled for this system."
        )


class SystemSpec(dict):
    """Hardware system spec backed by the YAML dict.

    The dict is the single source of truth — there are no parallel structured
    attributes. Construct directly with ``SystemSpec(yaml_dict)``.
    """

    @property
    def is_single_supernode(self) -> bool:
        return is_single_supernode(self)

    def validate_parallelism(
        self,
        *,
        tp: int,
        pp: int,
        attention_dp: int,
        cp: int,
        moe_tp: int = 1,
        moe_ep: int = 1,
    ) -> None:
        """Reject a parallel layout that escapes an explicitly single-supernode system."""
        validate_parallelism(
            self,
            tp=tp,
            pp=pp,
            attention_dp=attention_dp,
            cp=cp,
            moe_tp=moe_tp,
            moe_ep=moe_ep,
        )

    def get_pipeline_p2p_bandwidth(self) -> float:
        """Return PP P2P bandwidth while preserving legacy multi-node behavior."""
        return get_pipeline_p2p_bandwidth(self)

    def get_p2p_bandwidth(self, num_gpus: int) -> float:
        """Return point-to-point bandwidth (bytes/s) based on topology.

        Three-tier selection:

        - ``num_gpus <= num_gpus_per_node``: ``intra_node_bw`` (NVLink within node)
        - ``num_gpus <= num_gpus_per_rack``: ``inter_node_bw`` (NVSwitch within rack)
        - ``num_gpus > num_gpus_per_rack``: ``inter_rack_bw`` (InfiniBand between racks),
          falling back to ``inter_node_bw`` when ``inter_rack_bw`` is unset.

        Raises ``KeyError`` for misconfigured specs that lack required keys —
        same loud-failure behavior as the original ``_get_p2p_bandwidth``.
        """
        return get_p2p_bandwidth(self, num_gpus)
