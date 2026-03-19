## Multimodal Server Runbook

This runbook builds the stay-level artifacts needed for:

- EHR-only stay baseline
- EHR + image fusion
- EHR + CDE fusion
- EHR + CDE + image fusion

It uses the scripts already in this repo plus the new image-cache step.

### What Each Script Does

- [`build_master_stays.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_master_stays.py)
  Builds one ICU-stay table with `stay_id`, timing, and mortality labels.
- [`build_stay_image_index.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_stay_image_index.py)
  Maps the CXR metadata CSV to ICU stays.
- [`preprocess_stay_images.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/preprocess_stay_images.py)
  Reads the stay-image index, resizes matched JPGs to a fixed size, saves `.pt` tensors, and writes back an updated index with `tensor_path`.
- [`build_stay_sequences.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/data_pipelines/build_stay_sequences.py)
  Builds one token sequence per stay from MEDS events and writes `label_in_hosp_mortality`, `label_mortality_28d`, and `label_sepsis`.
- [`train_stay_fusion.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/training/train_stay_fusion.py)
  Trains the stay-level EHR/CDE/image fusion models.
- [`run_stay_ehr_only.sh`](/home/ap794/wspersonal/icu_digital_twins_mamba/launch/run_stay_ehr_only.sh)
  Launches the EHR-only stay model.
- [`run_stay_ehr_image.sh`](/home/ap794/wspersonal/icu_digital_twins_mamba/launch/run_stay_ehr_image.sh)
  Launches EHR + image fusion.
- [`run_stay_ehr_cde.sh`](/home/ap794/wspersonal/icu_digital_twins_mamba/launch/run_stay_ehr_cde.sh)
  Launches EHR + CDE fusion.
- [`run_stay_ehr_cde_image.sh`](/home/ap794/wspersonal/icu_digital_twins_mamba/launch/run_stay_ehr_cde_image.sh)
  Launches EHR + CDE + image fusion.

### Assumed Paths

Replace these with your real server paths if needed.

```bash
export REPO=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey
export PRE_MEDS=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/PRE_MEDS
export MEDS_TRAIN=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/MEDS_COHORT/merge_to_MEDS_cohort/train
export CXR_CSV=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey/data/cxr_icu_multiple_studies_deduplicated_reports_preprocessed_with_jpg_paths.csv

export STAY_ROOT=/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/odyssey/data/stay_level
export MASTER_STAYS=$STAY_ROOT/master_stays.parquet
export STAY_IMAGE_INDEX=$STAY_ROOT/stay_image_index.parquet
export STAY_IMAGE_INDEX_CACHED=$STAY_ROOT/stay_image_index_cached.parquet
export STAY_IMAGE_CACHE_DIR=$STAY_ROOT/image_cache
export STAY_SEQUENCES=$STAY_ROOT/stay_sequences_2048.parquet
export STAY_VOCAB_DIR=$STAY_ROOT/vocab
```

### Step 1: Build the ICU Stay Master Table

```bash
python $REPO/data_pipelines/build_master_stays.py \
  --icustays-path $PRE_MEDS/icu/icustays.csv.gz \
  --admissions-path $PRE_MEDS/hosp/admissions.csv.gz \
  --patients-path $PRE_MEDS/hosp/patients.parquet \
  --output-path $MASTER_STAYS
```

Expected output:

- `master_stays.parquet`
- stay-level mortality labels

### Step 2: Build the Stay-Level Image Index From the CXR CSV

Recommended first pass:

- keep all matched images
- optionally later try `--first-hours-only 24`

```bash
python $REPO/data_pipelines/build_stay_image_index.py \
  --master-stays-path $MASTER_STAYS \
  --cxr-csv-path $CXR_CSV \
  --output-path $STAY_IMAGE_INDEX \
  --keep-all-images
```

Expected output:

- `stay_image_index.parquet`
- one or more rows per stay, one row per matched study

### Step 3: Preprocess the Matched Images Into Tensor Cache

This is strongly recommended. It avoids decoding JPGs inside every training batch.

```bash
python $REPO/data_pipelines/preprocess_stay_images.py \
  --image-index-path $STAY_IMAGE_INDEX \
  --output-index-path $STAY_IMAGE_INDEX_CACHED \
  --output-dir $STAY_IMAGE_CACHE_DIR \
  --image-size 224
```

Expected output:

- `$STAY_IMAGE_CACHE_DIR/tensors_224/*.pt`
- `stay_image_index_cached.parquet` with:
  - `tensor_path`
  - `tensor_exists`

### Step 4: Build Stay-Level Sequences From MEDS

This creates the EHR tower input.

```bash
python $REPO/data_pipelines/build_stay_sequences.py \
  --meds-data-dir $MEDS_TRAIN \
  --master-stays-path $MASTER_STAYS \
  --output-path $STAY_SEQUENCES \
  --max-len 2048 \
  --context-hours 168 \
  --vocab-dir $STAY_VOCAB_DIR
```

What gets written:

- `event_tokens_2048`
- `type_tokens_2048`
- `age_tokens_2048`
- `position_tokens_2048`
- `elapsed_tokens_2048`
- `visit_tokens_2048`
- `label_in_hosp_mortality`
- `label_mortality_28d`
- `label_sepsis`

### Step 5: Point to Your CDE Outputs

Set these to the NCDE outputs your classmate pipeline already created.

```bash
export CDE_COEFFS_PATH=/path/to/cde/coeffs.pt
export CDE_META_PATH=/path/to/cde/meta.parquet
```

The `meta.parquet` file must contain `stay_id` so the CDE rows line up with the stay-level sequences.

### Step 6: Optional EHR Backbone Checkpoint

If you want the fusion experiments to start from your pretrained Mamba:

```bash
export EHR_CHECKPOINT=/path/to/mamba_pretrain.ckpt
export EHR_CONFIG_DIR=$REPO/odyssey/models/configs
```

If you do not set `EHR_CHECKPOINT`, the trainer falls back to a lightweight learned embedding encoder.

### Step 7: Launch the Four Experiments

#### 7A. EHR-only stay baseline

This uses the stay-level EHR encoder plus task heads, with the other modalities masked out.

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_only
export MODEL_TYPE=late

bash $REPO/launch/run_stay_ehr_only.sh
```

#### 7B. EHR + image fusion

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export IMAGE_INDEX_PATH=$STAY_IMAGE_INDEX_CACHED
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_image
export MODEL_TYPE=late
export IMAGE_SIZE=224
export MAX_IMAGES_PER_STAY=4

bash $REPO/launch/run_stay_ehr_image.sh
```

#### 7C. EHR + CDE fusion

```bash
export STAY_SEQUENCES_PATH=$STAY_SEQUENCES
export VOCAB_DIR=$STAY_VOCAB_DIR
export CDE_COEFFS_PATH=$CDE_COEFFS_PATH
export CDE_META_PATH=$CDE_META_PATH
export OUTPUT_DIR=$REPO/checkpoints/stay_ehr_cde
export MODEL_TYPE=late

bash $REPO/launch/run_stay_ehr_cde.sh
```

#### 7D. EHR + CDE + image fusion

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

bash $REPO/launch/run_stay_ehr_cde_image.sh
```

### Recommended Training Order

Run in this order:

1. EHR-only
2. EHR + image
3. EHR + CDE
4. EHR + CDE + image

That lets you measure the marginal value of each added modality.

### Recommended First Hyperparameters

These are safe first settings on the server:

```bash
export BATCH_SIZE=8
export EPOCHS=10
export NUM_WORKERS=4
export VAL_RATIO=0.2
export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-2
export FUSION_DIM=256
export MAX_LEN=2048
export EHR_HIDDEN_SIZE=768
export CDE_HIDDEN_SIZE=32
export IMAGE_HIDDEN_SIZE=768
```

If GPU memory is tight:

- reduce `BATCH_SIZE`
- reduce `MAX_IMAGES_PER_STAY`
- keep `IMAGE_SIZE=224`

### If You Want Cross-Attention Instead of Late Fusion

Change:

```bash
export MODEL_TYPE=cross
```

Or for a middle-ground baseline:

```bash
export MODEL_TYPE=gated
```

### Output Files to Expect

Each experiment writes into its own output directory:

- `best_model.pt`
- `last_model.pt`

### Practical Notes

- The EHR-only baseline here uses the same stay-level training stack as the multimodal models, so the comparison is fairer than mixing one patient-level trainer with three stay-level trainers.
- If your CDE `meta.parquet` uses a different stay id type than the stay-level EHR table, fix that first or the CDE tower will look missing for most rows.
- The image cache step is worth doing. Raw JPG loading inside the training loop will be much slower.
- If you want to restrict images to first-24h ICU studies, rebuild the stay-image index with `--first-hours-only 24` before the cache step.

### Legacy Sequence-Only Fine-Tuning

If you still want to run the original patient-sequence baseline with the old finetune script, use [`finetune.py`](/home/ap794/wspersonal/icu_digital_twins_mamba/training/finetune.py). That is a separate pipeline from the stay-level fusion experiments above.
