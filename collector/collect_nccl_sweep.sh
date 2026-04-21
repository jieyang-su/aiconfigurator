#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=$(pwd)
GPU_LIST=""
OPS="all_gather,alltoall,reduce_scatter,all_reduce"
DTYPES="half,int8"
TEST_RANGE="512,536870913,2"
CLEAN_OUTPUT=false
MEASURE_POWER=false
POWER_TEST_DURATION=1.0

if [[ -n "${NCCL_TEST_BIN_PATH:-}" ]]; then
    export PATH="${NCCL_TEST_BIN_PATH}:$PATH"
fi

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run NCCL collector sweeps and append results to nccl_perf.txt.

Options:
  --output-dir DIR           Output directory for nccl_perf.txt (default: current dir)
  --gpu-list LIST            Comma-separated GPU counts, e.g. 2,4,8
  --ops LIST                 Comma-separated ops: all_gather,alltoall,reduce_scatter,all_reduce
  --dtypes LIST              Comma-separated dtypes: half,int8
  --range SPEC               Message size sweep for collect_nccl.py (default: 512,536870913,2)
  --clean                    Remove existing nccl_perf.txt and lock file before running
  --measure_power            Enable NVML power monitoring
  --power_test_duration SEC  Forwarded to collect_nccl.py for consistency (default: 1.0)
  -h, --help                 Show this help message

Examples:
  $0
  $0 --gpu-list 2,4,8 --output-dir /tmp/b200_nccl
  $0 --ops all_gather,all_reduce --dtypes half --clean
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-list)
            GPU_LIST="$2"
            shift 2
            ;;
        --ops)
            OPS="$2"
            shift 2
            ;;
        --dtypes)
            DTYPES="$2"
            shift 2
            ;;
        --range)
            TEST_RANGE="$2"
            shift 2
            ;;
        --clean)
            CLEAN_OUTPUT=true
            shift
            ;;
        --measure_power)
            MEASURE_POWER=true
            shift
            ;;
        --power_test_duration|--power_test_duration_sec)
            POWER_TEST_DURATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found in PATH" >&2
    exit 1
fi

if ! command -v all_gather_perf >/dev/null 2>&1; then
    echo "all_gather_perf not found in PATH." >&2
    echo "Fix it by either:" >&2
    echo "  1. export NCCL_TEST_BIN_PATH=/path/to/nccl-tests/build" >&2
    echo "  2. export PATH=/path/to/nccl-tests/build:\$PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -z "$GPU_LIST" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found and --gpu-list not provided" >&2
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$GPU_COUNT" -ge 8 ]]; then
        GPU_LIST="2,4,8"
    elif [[ "$GPU_COUNT" -ge 4 ]]; then
        GPU_LIST="2,4"
    elif [[ "$GPU_COUNT" -ge 2 ]]; then
        GPU_LIST="2"
    else
        echo "Need at least 2 GPUs for NCCL sweep" >&2
        exit 1
    fi
fi

IFS=',' read -r -a GPU_COUNTS <<< "$GPU_LIST"
IFS=',' read -r -a OP_LIST <<< "$OPS"
IFS=',' read -r -a DTYPE_LIST <<< "$DTYPES"

if [[ "$CLEAN_OUTPUT" == "true" ]]; then
    rm -f "$OUTPUT_DIR/nccl_perf.txt" "$OUTPUT_DIR/nccl_perf.txt.lock"
fi

pushd "$OUTPUT_DIR" >/dev/null

for n in "${GPU_COUNTS[@]}"; do
    for op in "${OP_LIST[@]}"; do
        for dtype in "${DTYPE_LIST[@]}"; do
            echo "Running NCCL sweep: num_gpus=$n op=$op dtype=$dtype range=$TEST_RANGE"
            if [[ "$MEASURE_POWER" == "true" ]]; then
                python3 "$SCRIPT_DIR/collect_nccl.py" \
                    -n "$n" \
                    -NCCL "$op" \
                    --dtype "$dtype" \
                    --range "$TEST_RANGE" \
                    --measure_power \
                    --power_test_duration_sec "$POWER_TEST_DURATION"
            else
                python3 "$SCRIPT_DIR/collect_nccl.py" \
                    -n "$n" \
                    -NCCL "$op" \
                    --dtype "$dtype" \
                    --range "$TEST_RANGE"
            fi
        done
    done
done

popd >/dev/null

echo "NCCL sweep completed: $OUTPUT_DIR/nccl_perf.txt"