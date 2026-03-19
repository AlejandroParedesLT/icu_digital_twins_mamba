#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/data_pipelines/create_dataset_splits_streaming.py" \
    --patient_sequences /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048.parquet \
    --output_dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_id_dict \
    --max_len 2048 \
    --test_size 0.15 \
    --batch_size 10000
