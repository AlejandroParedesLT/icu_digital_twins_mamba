#!/bin/bash

export REPO=/home/ap794/wspersonal/icu_digital_twins_mamba
export CONFIG_DIR=$REPO/odyssey/models/configs

# export PRE_MEDS=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/PRE_MEDS
# export MEDS_TRAIN=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/MEDS_COHORT/merge_to_MEDS_cohort/train

export PATIENT_SEQ=$REPO/data/patient_sequences/patient_sequences_2048_labeled.parquet
export ID_FILE=dataset_2048_multi_v2.pkl
export VOCAB_DIR=$REPO/data/vocab

export PRETRAINED_BACKBONE=/home/ap794/wspersonal/icu_digital_twins_mamba/checkpoints/mamba_pretrain_with_embeddings/best.ckpt