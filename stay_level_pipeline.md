## Stay-Level Build Pipeline

This repository now includes three stay-level data builders:

- `build_master_stays.py`
- `build_stay_image_index.py`
- `build_stay_sequences.py`

### 1. Build the ICU stay master table

```bash
python build_master_stays.py \
  --icustays-path /path/to/PRE_MEDS/icu/icustays.csv.gz \
  --admissions-path /path/to/PRE_MEDS/hosp/admissions.csv.gz \
  --patients-path /path/to/PRE_MEDS/hosp/patients.parquet \
  --output-path data/stay_level/master_stays.parquet
```

### 2. Build the stay-aligned image index

```bash
python build_stay_image_index.py \
  --master-stays-path data/stay_level/master_stays.parquet \
  --cxr-csv-path data/cxr_icu_multiple_studies_deduplicated_reports_preprocessed_with_jpg_paths.csv \
  --output-path data/stay_level/stay_image_index.parquet \
  --keep-all-images
```

To keep only early ICU images, add `--first-hours-only 24`.

### 3. Build stay-level token sequences from MEDS events

```bash
python build_stay_sequences.py \
  --meds-data-dir /path/to/MEDS_COHORT/data/train \
  --master-stays-path data/stay_level/master_stays.parquet \
  --output-path data/stay_level/stay_sequences_2048.parquet \
  --max-len 2048 \
  --context-hours 168 \
  --vocab-dir data/stay_level/vocab
```

`context-hours 168` keeps the 7 days before ICU admission plus the stay itself.

### Outputs

- `master_stays.parquet`: stay-level timing and labels
- `stay_image_index.parquet`: one or more image studies per stay
- `stay_sequences_2048.parquet`: one token sequence per stay
- `data/stay_level/vocab/event_vocab.json`: event tokenizer vocabulary for stay sequences

### 4. Train the first multimodal fusion baseline

```bash
python train_stay_fusion.py \
  --stay-sequences-path data/stay_level/stay_sequences_2048.parquet \
  --vocab-dir data/stay_level/vocab \
  --cde-coeffs-path /path/to/cde/coeffs.pt \
  --cde-meta-path /path/to/cde/meta.parquet \
  --image-index-path data/stay_level/stay_image_index.parquet \
  --output-dir checkpoints/stay_fusion_late \
  --model-type late
```

Switch `--model-type` to `gated` or `cross` for the stronger fusion variants.
