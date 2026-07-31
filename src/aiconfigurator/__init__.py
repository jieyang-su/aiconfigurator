# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aiconfigurator")
except importlib.metadata.PackageNotFoundError:
    # Source-tree/maturin --skip-install execution used by offline DynoSim.
    __version__ = "0.9.0"
