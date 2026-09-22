#!/usr/bin/env bash
# Usage (batch-size is per GPU; four GPUs give global batch-size 4):
# bash scripts/train_hyperseg_4gpu.sh --track A2 \
#   --config training/decoder_train/code/configs/hyperseg_h_raw.yaml \
#   --manifest /data/train/manifest.json --test-manifest /data/test/manifest.json \
#   --output runs/hyperseg_a2 --batch-size 1 --epochs 100
# Equivalent: torchrun --standalone --nnodes=1 --nproc-per-node=4 \
#   scripts/hyperseg_h.py train --precision bf16 [arguments above]
# --checkpoint runs/hyperseg_a2/latest.pt --resume resumes optimizer/epoch.
set -euo pipefail
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
exec torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    "$(dirname "$0")/hyperseg_h.py" train --device cuda --precision bf16 "$@"
