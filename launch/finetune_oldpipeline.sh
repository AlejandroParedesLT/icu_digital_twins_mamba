#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO="${REPO:-$REPO_ROOT}"

export WANDB_LOG_LEVEL=debug
export WANDB_CONSOLE=off  # reduces overhead
export WANDB_MODE=online
export WANDB_DIR=./logs
export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SEQUENCE_FILE="${SEQUENCE_FILE:-patient_sequences_2048_labeled.parquet}"

python $REPO/training/finetune.py \
  --model-type ehr_mamba \
  --exp-name finetune_mortality_old_pipeline \
  --pretrained-path $PRETRAINED_BACKBONE \
  --label-name label_mortality_1month \
  --config-dir $CONFIG_DIR \
  --is-decoder True \
  --data-dir $REPO/data \
  --sequence-file $SEQUENCE_FILE \
  --id-file $ID_FILE \
  --vocab-dir $VOCAB_DIR \
  --val-size 0.1 \
  --valid_scheme few_shot \
  --num_finetune_patients all \
  --problem_type single_label_classification \
  --num_labels 2 \
  --checkpoint-dir $REPO/checkpoints \
  --test_output_dir test_outputs \
  --wandb-project icu_digital_twins \
  --num_finetune_patients 100