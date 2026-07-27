#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper for recorded MoE truth families.
#
# Examples:
#   PLATFORM=h20 FAMILY=ordinary_context ./run_truth_family.sh
#   PLATFORM=h100 FAMILY=wideep_generation PRESET=deepseek_v3 ./run_truth_family.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${FAMILY:?set FAMILY to ordinary_context, ordinary_generation, wideep_context, or wideep_generation}"
PLATFORM="${PLATFORM:?set PLATFORM, for example h20 or h100}"
PRESET="${PRESET:-deepseek_v3}"

case "${PRESET}" in
  deepseek_v3)
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/presets/deepseek_v3.sh"
    ;;
  none)
    ;;
  *)
    echo "unknown PRESET=${PRESET}" >&2
    exit 2
    ;;
esac

export FAMILY PLATFORM
exec "${SCRIPT_DIR}/run_truth_stage.sh" "$@"
