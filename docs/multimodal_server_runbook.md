## Multimodal Server Runbook

This runbook is organized around the actual modeling hierarchy:

1. **Old patient-level pipeline** = pretrain the strong EHR Mamba backbone
2. **Old patient-level finetuning** = the fastest baseline you can still run today
3. **Stay-level pipeline** = align EHR, images, and CDE at `stay_id`
4. **Fusion experiments** = build the final multimodal model on top of the backbone

If you are presenting in a few hours, the practical recommendation is:

1. run the **old finetuning pipeline first**
2. if time permits, run **EHR + image stay-level fusion** as the first multimodal demo

The old pipeline is still the best source of the pretrained backbone.  
The stay-level pipeline is the correct abstraction for multimodal ICU prediction.

### Model Hierarchy

Think about the stack like this:

- **Pretraining backbone**
  - old patient-level sequence pipeline
  - learns medication/event semantics and long-range EHR structure
- **Old finetuning baseline**
  - same patient-level pipeline
  - fastest path to a first mortality/sepsis result
- **Stay-level EHR encoder**
  - reuse the pretrained backbone
  - apply it to stay-specific sequences instead of patient-global ones
- **Fusion model**
  - combine stay-level EHR + image + CDE

So:

- the old pipeline is **not obsolete**
- it is the **foundation**
- the stay-level pipeline is the **alignment layer on top**

## What Exists in This Repo

### Old Patient-Level Pipeline

- [`training/pretrain.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/training/pretrain.py)
- [`training/finetune.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/training/finetune.py)
- [`data_pipelines/preprocess_dataset.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/preprocess_dataset.py)

Input unit:

- one **patient-level** sequence row

Typical file:

- `data/patient_sequences/patient_sequences_2048.parquet`

### New Stay-Level Multimodal Pipeline

- [`data_pipelines/build_master_stays.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_master_stays.py)
- [`data_pipelines/build_stay_image_index.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_stay_image_index.py)
- [`data_pipelines/preprocess_stay_images.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/preprocess_stay_images.py)
- [`data_pipelines/build_stay_sequences.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_stay_sequences.py)
- [`training/train_stay_fusion.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/training/train_stay_fusion.py)

Input unit:

- one **ICU stay** = one sample

Typical file:

- `data/stay_level/stay_sequences_2048.parquet`

## Assumed Server Paths

Replace these with your real server paths if needed.

```bash
export REPO=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey
export CONFIG_DIR=$REPO/odyssey/models/configs

export PRE_MEDS=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/PRE_MEDS
export MEDS_TRAIN=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/MEDS_COHORT/merge_to_MEDS_cohort/train

export PATIENT_SEQ=$REPO/data/patient_sequences/patient_sequences_2048_labeled.parquet
export ID_FILE=dataset_2048_multi_v2.pkl
export VOCAB_DIR=$REPO/data/vocab

export PRETRAINED_BACKBONE=/path/to/your/pretrained_backbone.ckpt
```

## Part A: From Zero To Backbone

### A1. Build the Patient-Level Sequence Parquet

Use the old preprocessing pipeline:

```bash
python $REPO/data_pipelines/preprocess_dataset.py \
  --meds_prep_dir $MEDS_TRAIN \
  --patients_path $PRE_MEDS/hosp/patients.parquet \
  --output_dir $REPO/data
```

This should produce:

- `data/patient_sequences/patient_sequences_2048.parquet`
- `data/patient_sequences/patient_sequences_2048_labeled.parquet`
- `data/patient_id_dict/dataset_2048_multi_v2.pkl`
- `data/vocab/event_vocab.json`

### A2. Optional: Repair Age Tokens If Needed

If integrity checks show age is wrong, repair in place:

```bash
python $REPO/data_pipelines/fix_age_tokens_parquet.py \
  $REPO/data/patient_sequences/patient_sequences_2048_labeled.parquet

python $REPO/data_pipelines/fix_age_tokens_parquet.py \
  $REPO/data/patient_sequences/patient_sequences_2048.parquet
```

### A3. Pretrain The Backbone

The old patient-level sequence pipeline is the correct place to pretrain the EHR backbone.

Example:

```bash
python $REPO/training/pretrain.py \
  --model_type ehr_mamba \
  --is_decoder True \
  --exp_name mamba_pretrain_with_embeddings \
  --config_dir $CONFIG_DIR \
  --data_dir $REPO/data \
  --sequence_file $REPO/data/patient_sequences/patient_sequences_2048.parquet \
  --id_file $REPO/data/patient_id_dict/dataset_2048_multi_v2.pkl \
  --vocab_dir $VOCAB_DIR \
  --val_size 0.1 \
  --checkpoint_dir $REPO/checkpoints/pretrain
```

Output:

- pretrained Mamba checkpoint

That checkpoint is the **backbone** used later by the stay-level fusion models.

### B1. Old-Pipeline Mortality Finetuning

This is the shortest path to a first result.

```bash
python $REPO/training/finetune.py \
  --model-type ehr_mamba \
  --exp-name finetune_mortality_old_pipeline \
  --pretrained-path $PRETRAINED_BACKBONE \
  --label-name label_mortality_1month \
  --config-dir $CONFIG_DIR \
  --is-decoder True \
  --data-dir $REPO/data \
  --sequence-file patient_sequences_2048_labeled.parquet \
  --id-file $ID_FILE \
  --vocab-dir $VOCAB_DIR \
  --val-size 0.1 \
  --valid_scheme few_shot \
  --num_finetune_patients all \
  --problem_type single_label_classification \
  --num_labels 2 \
  --checkpoint-dir $REPO/checkpoints \
  --test_output_dir test_outputs
```

Important notes:

- `--num_finetune_patients all` works because the current split pickle includes `finetune["few_shot"]["all"]`
- this is the old **patient-level** baseline
- this is not the stay-level baseline

### B3. Old-Pipeline Sepsis Finetuning

Only do this if your patient-level labeled parquet already contains `label_sepsis`.

Check first:

```bash
python - <<'PY'
import polars as pl
df = pl.read_parquet("/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey/data/patient_sequences/patient_sequences_2048_labeled.parquet")
print("label_sepsis" in df.columns)
if "label_sepsis" in df.columns:
    print(df["label_sepsis"].value_counts())
PY
```

If it exists, run:

```bash
python $REPO/training/finetune.py \
  --model-type ehr_mamba \
  --exp-name finetune_sepsis_old_pipeline \
  --pretrained-path $PRETRAINED_BACKBONE \
  --label-name label_sepsis \
  --config-dir $CONFIG_DIR \
  --is-decoder True \
  --data-dir $REPO/data \
  --sequence-file patient_sequences_2048_labeled.parquet \
  --id-file $ID_FILE \
  --vocab-dir $VOCAB_DIR \
  --val-size 0.1 \
  --valid_scheme few_shot \
  --num_finetune_patients all \
  --problem_type single_label_classification \
  --num_labels 2 \
  --checkpoint-dir $REPO/checkpoints \
  --test_output_dir test_outputs
```

### C1. Build Stay-Level Artifacts

```bash
export STAY_ROOT=$REPO/data/stay_level
export MASTER_STAYS=$STAY_ROOT/master_stays.parquet
export STAY_IMAGE_INDEX=$STAY_ROOT/stay_image_index.parquet
export STAY_IMAGE_INDEX_CACHED=$STAY_ROOT/stay_image_index_cached.parquet
export STAY_IMAGE_CACHE_DIR=$STAY_ROOT/image_cache
export STAY_SEQUENCES=$STAY_ROOT/stay_sequences_2048.parquet
export STAY_VOCAB_DIR=$STAY_ROOT/vocab
export CXR_CSV=$REPO/data/cxr_icu_multiple_studies_deduplicated_reports_preprocessed_with_jpg_paths.csv
```

Build master stays:

```bash
python $REPO/data_pipelines/build_master_stays.py \
  --icustays-path $PRE_MEDS/icu/icustays.csv.gz \
  --admissions-path $PRE_MEDS/hosp/admissions.csv.gz \
  --patients-path $PRE_MEDS/hosp/patients.parquet \
  --output-path $MASTER_STAYS
```

Build the stay-image index:

```bash
python $REPO/data_pipelines/build_stay_image_index.py \
  --master-stays-path $MASTER_STAYS \
  --cxr-csv-path $CXR_CSV \
  --output-path $STAY_IMAGE_INDEX \
  --keep-all-images
```

If the CXR CSV and JPG tree live on a different cluster, do not transfer the raw
JPG hierarchy. Instead preprocess them there into portable tensor packs first:

```bash
python $REPO/data_pipelines/precompute_cxr_tensor_pack.py \
  --cxr-csv-path /hpc/group/kamaleswaranlab/mimic_cxr/cxr_icu_multiple_studies_deduplicated_reports_preprocessed_with_jpg_paths.csv \
  --image-root /hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg \
  --output-dir /hpc/group/kamaleswaranlab/mimic_cxr/precomputed_cxr_tensor_pack_224 \
  --image-size 224 \
  --shard-size 1024
```

That writes:

- `cxr_tensor_manifest.parquet`
- `tensor_packs/cxr_tensor_pack_*.pt`

Copy that compact directory to the training server, then materialize only the
stay-matched studies into the local cache format:

```bash
python $REPO/data_pipelines/materialize_stay_image_tensors_from_pack.py \
  --stay-image-index-path $STAY_IMAGE_INDEX \
  --manifest-path /path/to/precomputed_cxr_tensor_pack_224/cxr_tensor_manifest.parquet \
  --pack-root /path/to/precomputed_cxr_tensor_pack_224 \
  --output-dir $STAY_IMAGE_CACHE_DIR \
  --output-index-path $STAY_IMAGE_INDEX_CACHED
```

Use this route when the images live on another server and you want to transfer a
portable preprocessed cache instead of millions of raw JPG files.

Cache the matched images:

```bash
python $REPO/data_pipelines/preprocess_stay_images.py \
  --image-index-path $STAY_IMAGE_INDEX \
  --output-index-path $STAY_IMAGE_INDEX_CACHED \
  --output-dir $STAY_IMAGE_CACHE_DIR \
  --image-size 224
```

Build the stay-level EHR sequences:

```bash
python $REPO/data_pipelines/build_stay_sequences.py \
  --meds-data-dir $MEDS_TRAIN \
  --master-stays-path $MASTER_STAYS \
  --output-path $STAY_SEQUENCES \
  --max-len 2048 \
  --context-hours 168 \
  --vocab-dir $STAY_VOCAB_DIR
```

### C2. Run EHR + Image Fusion

Use the pretrained patient-level backbone as the EHR encoder initialization:

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export IMAGE_INDEX_PATH=$STAY_IMAGE_INDEX_CACHED
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_image
export MODEL_TYPE=late
export IMAGE_SIZE=224
export MAX_IMAGES_PER_STAY=4
export EHR_CHECKPOINT=$PRETRAINED_BACKBONE
export EHR_CONFIG_DIR=$CONFIG_DIR

bash $REPO/launch/run_stay_ehr_image.sh
```

This is the fastest way to show “we added images” without rewriting the old baseline trainer.

## Part D: Full Stay-Level Multimodal Path

Once the baseline and the first image fusion run are done, continue with the full stack.

### D1. CDE Inputs

Set these to your classmate’s NCDE outputs:

```bash
export CDE_COEFFS_PATH=/path/to/cde/coeffs.pt
export CDE_META_PATH=/path/to/cde/meta.parquet
```

The `meta.parquet` file must contain `stay_id`.

### D2. Launch The Stay-Level Experiments

#### EHR-only stay baseline

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_only
export MODEL_TYPE=late
export EHR_CHECKPOINT=$PRETRAINED_BACKBONE
export EHR_CONFIG_DIR=$CONFIG_DIR

bash $REPO/launch/run_stay_ehr_only.sh
```

#### EHR + image

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export IMAGE_INDEX_PATH=$STAY_IMAGE_INDEX_CACHED
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_image
export MODEL_TYPE=late
export IMAGE_SIZE=224
export MAX_IMAGES_PER_STAY=4
export EHR_CHECKPOINT=$PRETRAINED_BACKBONE
export EHR_CONFIG_DIR=$CONFIG_DIR

bash $REPO/launch/run_stay_ehr_image.sh
```

#### EHR + CDE

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export CDE_COEFFS_PATH=$CDE_COEFFS_PATH
export CDE_META_PATH=$CDE_META_PATH
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_cde
export MODEL_TYPE=late
export EHR_CHECKPOINT=$PRETRAINED_BACKBONE
export EHR_CONFIG_DIR=$CONFIG_DIR

bash $REPO/launch/run_stay_ehr_cde.sh
```

#### EHR + CDE + image

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export CDE_COEFFS_PATH=$CDE_COEFFS_PATH
export CDE_META_PATH=$CDE_META_PATH
export IMAGE_INDEX_PATH=$STAY_IMAGE_INDEX_CACHED
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_cde_image
export MODEL_TYPE=late
export IMAGE_SIZE=224
export MAX_IMAGES_PER_STAY=4
export EHR_CHECKPOINT=$PRETRAINED_BACKBONE
export EHR_CONFIG_DIR=$CONFIG_DIR

bash $REPO/launch/run_stay_ehr_cde_image.sh
```

## Recommended Order Overall

For the next few hours:

1. old-pipeline mortality finetune
2. old-pipeline sepsis finetune if label exists
3. stay-level EHR + image

For the full project:

1. backbone pretraining
2. old-pipeline baseline finetuning
3. stay-level EHR-only
4. stay-level EHR + image
5. stay-level EHR + CDE
6. stay-level EHR + CDE + image

## Practical Notes

- The old patient-level baseline and the stay-level EHR-only baseline are **not the same experiment**.
- The old pipeline is still the correct place to obtain the pretrained backbone.
- The stay-level pipeline is the correct place to align images and CDE.
- For today, do not try to force images directly into the old finetune script unless you are ready to debug a new model path.
- The image cache step is worth doing. Raw JPG decoding in every training batch is much slower.
- If GPU memory is tight, reduce:
  - `BATCH_SIZE`
  - `MAX_IMAGES_PER_STAY`
- If you want a different fusion style later, change:
  - `export MODEL_TYPE=gated`
  - or `export MODEL_TYPE=cross`

## Output Files To Expect

Old finetune runs:

- Lightning checkpoints in the chosen checkpoint directory
- optional `test_outputs/*.pt`

Stay-level fusion runs:

- `best_model.pt`
- `last_model.pt`

## Bottom Line

For presentation today:

- use the **old pipeline** to show the backbone-based baseline
- use **stay-level EHR + image** as the first multimodal add-on

That matches the real architecture logic and avoids wasting time trying to graft images into the old patient-level finetune path right before a deadline.
