"""Table-free Kimi Delta Attention kernel model."""

from .dispatch import KdaAnalyticalEstimate, estimate_kernel

__all__ = ["KdaAnalyticalEstimate", "estimate_kernel"]
