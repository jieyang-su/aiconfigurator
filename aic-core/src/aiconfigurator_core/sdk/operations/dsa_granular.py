"""Rust-backed granular DSA operations used by analytical model graphs.

The constructors and query implementations live in the compiled engine.  This
module only keeps the historical Python import names for model builders.
"""

from __future__ import annotations

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit


class DSAIndexScore(_core.DSAIndexScore, OpShellKit):
    pass


class DSATopKSelect(_core.DSATopKSelect, OpShellKit):
    pass


class DSASparseAttention(_core.DSASparseAttention, OpShellKit):
    pass
