#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

python3 "$REPO_ROOT/training/pretrain.py"  \
    --model_type ehr_mamba \
    --is_decoder True \
    --exp_name mamba_pretrain_with_embeddings \
    --config_dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/odyssey/models/configs/ \
    --data_dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/raw/train \
    --sequence_file /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048.parquet \
    --id_file /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_id_dict/dataset_2048_multi_v2.pkl \
    --vocab_dir  /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/vocab \
    --val_size 0.1 \
    --checkpoint_dir checkpoints
