"""Unified communication analytical proxy.

The proxy is calibrated from the H100/H200 NCCL measurements summarized in
``.self/domestic-gpu/communication-startup-bandwidth-model``.  Callers supply
the existing SOL latency, which already contains the collective byte volume,
ring factor, selected topology bandwidth, and communication dtype.  The
proxy deliberately adds only a fixed launch term and a calibrated efficiency
factor:

    latency = startup + sol_latency / eta

This is a topology- and backend-agnostic engineering proxy for systems that
have no collective silicon table.  It is not a replacement for a measured
NCCL, oneCCL, CustomAllReduce, or vendor-specific collective curve.
"""

from __future__ import annotations

from dataclasses import dataclass

COMMUNICATION_MODEL_VERSION = "2026-08-17.hopper-startup-eta-v1"


@dataclass(frozen=True)
class CommunicationParameters:
    """Fixed parameters for the general-purpose communication proxy."""

    startup_us: float = 7.0
    eta: float = 0.75

    def __post_init__(self) -> None:
        if self.startup_us < 0:
            raise ValueError("communication startup must be non-negative")
        if not 0 < self.eta <= 1:
            raise ValueError("communication eta must be in (0, 1]")


@dataclass(frozen=True)
class CommunicationEstimate:
    """Inspectable result of one communication proxy evaluation."""

    latency_ms: float
    startup_ms: float
    payload_ms: float
    sol_latency_ms: float
    eta: float


HOPPER_COMMUNICATION_PARAMS = CommunicationParameters()


def estimate_communication(
    sol_latency_ms: float,
    params: CommunicationParameters = HOPPER_COMMUNICATION_PARAMS,
) -> CommunicationEstimate:
    """Apply the calibrated startup-plus-efficiency proxy to an SOL latency.

    ``sol_latency_ms`` is the existing AIC SOL result for the exact requested
    message and topology.  Zero SOL is a no-op communication and must remain
    zero rather than paying a launch term for a group of size one.
    """
    sol_latency_ms = float(sol_latency_ms)
    if sol_latency_ms < 0:
        raise ValueError("SOL communication latency must be non-negative")
    if sol_latency_ms == 0:
        return CommunicationEstimate(0.0, 0.0, 0.0, 0.0, params.eta)
    startup_ms = params.startup_us / 1000.0
    payload_ms = sol_latency_ms / params.eta
    return CommunicationEstimate(
        latency_ms=startup_ms + payload_ms,
        startup_ms=startup_ms,
        payload_ms=payload_ms,
        sol_latency_ms=sol_latency_ms,
        eta=params.eta,
    )
