#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# DeepSeek-V3 recorded MoE truth defaults. Source this file before
# run_truth_stage.sh when reproducing the DSV3 truth workflow.

case "${FAMILY:-}" in
  wideep_context)
    export DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-64 128 512 640 1536 2048 2560 4096 5120 8192 10240 12288 14336 16384 18888}"
    export DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-}"
    export DEFAULT_EPS="${DEFAULT_EPS:-2 4 8}"
    ;;
  wideep_generation)
    export DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-}"
    export DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-8 32 40 64 128 288 512 896 1024 1280}"
    export DEFAULT_EPS="${DEFAULT_EPS:-2 4 8}"
    ;;
  ordinary_context)
    export DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-4 8 32 64 128 512 640 896 1536 2048 4096 5120 8192 12288 16384}"
    export DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-}"
    export DEFAULT_EPS="${DEFAULT_EPS:-1 2 4 8}"
    ;;
  ordinary_generation)
    export DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-}"
    export DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-4 8 32 40 64 128 288 512 896 1024 1280 4096 8192 16384}"
    export DEFAULT_EPS="${DEFAULT_EPS:-1 2 4 8}"
    ;;
  "")
    ;;
  *)
    echo "unknown DSV3 FAMILY preset: ${FAMILY}" >&2
    return 2
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-/model/DeepSeek-V3}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export PROFILE_LAYERS="${PROFILE_LAYERS:-3 4 5}"
export MODEL_OVERRIDE_ARGS="${MODEL_OVERRIDE_ARGS:-num_hidden_layers=6,first_k_dense_replace=3}"
