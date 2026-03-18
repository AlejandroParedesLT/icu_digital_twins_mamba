#!/bin/bash
#SBATCH --job-name=mamba_pretrain
#SBATCH --time=06:00:00
#SBATCH --account=kamaleswaranlab
#SBATCH --partition=gpu-common
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --output=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey/mamba_pretrain_a5000-%j.out
#SBATCH --error=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey/mamba_pretrain_a5000-%j.err
#SBATCH --no-requeue

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh

# Activate the shared environment
conda activate /hpc/group/kamaleswaranlab/Conda_Environments/alejandrop_conda/envs/mimic-pipeline/
pip install --upgrade transformers

echo "Python: $(which python)"
echo "Conda environment: $(conda info --envs | grep '*' )"


cd /hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 pretrain.py  \
                --model-type ehr_mamba \
                --is-decoder True \
                --exp-name mamba_pretrain_with_embeddings \
                --config-dir /hpc/home/aparedeslatorre1/icu_digital_twins_mamba/odyssey/models/configs \
                --data-dir /hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/MEDS_COHORT/data/train \
                --sequence-file data/patient_sequences/patient_sequences_2048.parquet \
                --id-file patient_id_dict/dataset_2048_multi_v2.pkl \
                --vocab-dir odyssey/data/vocab \
                --val-size 0.1 \
                --checkpoint-dir checkpoints