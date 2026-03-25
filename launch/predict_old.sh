#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO="${REPO:-$REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SEQUENCE_FILE="${SEQUENCE_FILE:-patient_sequences_2048_labeled.parquet}"

# The checkpoint saved by your training run lives here:
# $REPO/checkpoints/finetune_mortality_old_pipeline/best.ckpt  (or last.ckpt)
CHECKPOINT_PATH="$REPO/checkpoints/finetune_mortality_old_pipeline/best.ckpt"

python $REPO/training/predict.py \
  --model-type ehr_mamba \
  --config-dir $CONFIG_DIR \
  --checkpoint-path $CHECKPOINT_PATH \
  --is-decoder True \
  --label-name label_mortality_1month \
  --data-dir $REPO/data \
  --sequence-file $SEQUENCE_FILE \
  --id-file $ID_FILE \
  --vocab-dir $VOCAB_DIR \
  --valid_scheme few_shot \
  --num_finetune_patients 100 \
  --problem_type single_label_classification \
  --num_labels 2 \
  --batch-size 64 \
  --num-workers 4 \
  --target-recall 0.80 \
  --output-dir $REPO/checkpoints/finetune_mortality_old_pipeline/test_outputs