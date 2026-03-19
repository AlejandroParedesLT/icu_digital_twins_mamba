#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${STAY_SEQUENCES_PATH:?Set STAY_SEQUENCES_PATH}"
: "${VOCAB_DIR:?Set VOCAB_DIR}"
: "${IMAGE_INDEX_PATH:?Set IMAGE_INDEX_PATH}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

MODEL_TYPE="${MODEL_TYPE:-late}"
MAX_LEN="${MAX_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_RATIO="${VAL_RATIO:-0.2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
FUSION_DIM="${FUSION_DIM:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_IMAGES_PER_STAY="${MAX_IMAGES_PER_STAY:-4}"
EHR_HIDDEN_SIZE="${EHR_HIDDEN_SIZE:-768}"
CDE_HIDDEN_SIZE="${CDE_HIDDEN_SIZE:-32}"
IMAGE_HIDDEN_SIZE="${IMAGE_HIDDEN_SIZE:-768}"

python "$REPO_ROOT/training/train_stay_fusion.py" \
  --stay-sequences-path "${STAY_SEQUENCES_PATH}" \
  --vocab-dir "${VOCAB_DIR}" \
  --image-index-path "${IMAGE_INDEX_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-type "${MODEL_TYPE}" \
  --max-len "${MAX_LEN}" \
  --image-size "${IMAGE_SIZE}" \
  --max-images-per-stay "${MAX_IMAGES_PER_STAY}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --val-ratio "${VAL_RATIO}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --fusion-dim "${FUSION_DIM}" \
  --ehr-hidden-size "${EHR_HIDDEN_SIZE}" \
  --cde-hidden-size "${CDE_HIDDEN_SIZE}" \
  --image-hidden-size "${IMAGE_HIDDEN_SIZE}" \
  ${IMAGE_ROOT:+--image-root "${IMAGE_ROOT}"} \
  ${EHR_CHECKPOINT:+--ehr-checkpoint "${EHR_CHECKPOINT}"} \
  ${EHR_CONFIG_DIR:+--ehr-config-dir "${EHR_CONFIG_DIR}"}
