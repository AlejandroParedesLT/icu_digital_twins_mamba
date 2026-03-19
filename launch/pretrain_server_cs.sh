#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# export CUBLAS_WORKSPACE_CONFIG=:4096:2
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export PYTHONPATH=/hpc/home/aparedeslatorre1/icu_digital_twins_mamba/

# cd /hpc/home/aparedeslatorre1/icu_digital_twins_mamba

# python3 "$REPO_ROOT/training/pretrain.py" \
#     --model_type ehr_mamba \
#     --is_decoder True \
#     --exp_name mamba_pretrain_with_embeddings \
#     --config_dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/odyssey/models/configs \
#     --data_dir data/raw/train \
#     --sequence_file /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048.parquet \
#     --id_file /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_id_dict/dataset_2048_multi_v2.pkl \
#     --vocab_dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/vocab \
#     --val_size 0.1 \
#     --checkpoint_dir checkpoints

export WANDB_LOG_LEVEL=debug
export WANDB_CONSOLE=off  # reduces overhead
export WANDB_MODE=online
export WANDB_DIR=/lstor/scratch/aparedeslatorre1/logs
export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES=

python3 "$REPO_ROOT/training/pretrain.py" \
    --model_type ehr_mamba \
    --is_decoder True \
    --exp_name mamba_pretrain_with_embeddings \
    --config_dir /odyssey/models/configs \
    --data_dir /data/ \
    --sequence_file /data/patient_sequences/patient_sequences_2048.parquet \
    --id_file /data/patient_id_dict/dataset_2048_multi_v2.pkl \
    --vocab_dir /data/vocab \
    --val_size 0.1 \
    --checkpoint_dir /checkpoints \
    --log_dir /logs
