"""
Script to create tokenized dataset from preprocessed MEDS data.

This script:
1. Loads preprocessed MEDS sequences
2. Creates vocabularies for different token types
3. Tokenizes patient sequences
4. Creates train/test/finetune splits
5. Saves patient_sequences_2048.parquet and dataset_2048_multi_v2.pkl

Usage:
    python preprocess_dataset.py \
        --meds_prep_dir /path/to/MEDS_COHORT \
        --output_dir /path/to/output \
        --max_len 2048 \
        --test_size 0.15
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.utils.utils import save_object_to_disk


TYPE_MAP = {
    "special": 0,
    "diagnosis": 1,
    "lab": 2,
    "medication": 3,
    "procedure": 4,
    "transfer": 5,
    "icu": 6,
    "infusion": 7,
    "other": 8,
}

DEATH_TOKEN = "MEDS_DEATH"
SEPSIS_DIAGNOSIS_REGEX = r"^DIAGNOSIS(?:038|A40|A41)"


def load_meds_data(meds_prep_dir: str) -> pl.LazyFrame:
    """
    Load preprocessed MEDS data from parquet files.
    
    Parameters
    ----------
    meds_prep_dir : str
        Directory containing preprocessed MEDS parquet files
        
    Returns
    -------
    pl.LazyFrame
        Lazy MEDS event frame
    """
    print(f"Loading MEDS data from {meds_prep_dir}...")
    
    # Look for parquet files in the data directory
    parquet_pattern = os.path.join(meds_prep_dir, "**", "*.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {meds_prep_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Keep the heavy path lazy so we do not materialize 392M rows in pandas.
    lazy_df = pl.scan_parquet(parquet_files)
    print("Initialized lazy MEDS scan")
    return lazy_df


def load_patient_metadata(patients_path: str) -> pl.DataFrame:
    """Load patient-level metadata used to enrich MEDS events."""
    patients_fp = Path(patients_path)
    if patients_fp.suffix == ".parquet":
        patients = pl.read_parquet(patients_fp)
    else:
        patients = pl.read_csv(patients_fp)

    rename_map = {column: column.lower() for column in patients.columns}
    patients = patients.rename(rename_map)
    if "subject_id" not in patients.columns:
        raise ValueError("patients file must contain a subject_id column")

    keep_columns = [
        column
        for column in ["subject_id", "anchor_age", "anchor_year", "dod", "year_of_birth"]
        if column in patients.columns
    ]
    if len(keep_columns) == 1:
        raise ValueError(
            "patients file does not contain anchor_age/anchor_year or year_of_birth for age derivation"
        )

    patients = (
        patients.select(keep_columns)
        .with_columns(pl.col("subject_id").cast(pl.Int64, strict=False))
        .unique(subset=["subject_id"], keep="first")
    )
    return patients


def enrich_with_patient_metadata(df: pl.LazyFrame, patients_path: str) -> pl.LazyFrame:
    """Merge patient metadata into MEDS events for age derivation."""
    print(f"Loading patient metadata from {patients_path}...")
    patients = load_patient_metadata(patients_path)
    enriched = df.join(patients.lazy(), on="subject_id", how="left")

    if "year_of_birth" in patients.columns and "anchor_year" not in patients.columns:
        enriched = enriched.with_columns(
            [
                pl.col("year_of_birth").cast(pl.Int64, strict=False).alias("anchor_year"),
                (
                    pl.col("time").cast(pl.Datetime, strict=False).dt.year()
                    - pl.col("year_of_birth").cast(pl.Int64, strict=False)
                )
                .clip(lower_bound=0)
                .alias("anchor_age"),
            ]
        )

    schema_names = set(enriched.collect_schema().names())
    found_anchor_age = "anchor_age" in schema_names
    found_anchor_year = "anchor_year" in schema_names
    print(
        "Patient metadata merge complete:",
        {
            "anchor_age_present": bool(found_anchor_age),
            "anchor_year_present": bool(found_anchor_year),
        },
    )
    return enriched


def ensure_patient_id_column(df: pl.DataFrame) -> pl.DataFrame:
    """Mirror subject_id into patient_id for downstream Odyssey utilities."""
    if "patient_id" not in df.columns and "subject_id" in df.columns:
        df = df.with_columns(pl.col("subject_id").alias("patient_id"))
    return df


def extract_sequences_from_meds(df: pl.LazyFrame, max_len: int = 2048) -> pl.DataFrame:
    """
    Extract and structure sequences from MEDS format.
    
    This assumes the MEDS data has been preprocessed with the generate_sequence step
    and contains columns for event codes, timestamps, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw MEDS dataframe
    max_len : int
        Maximum sequence length
        
    Returns
    -------
    pl.DataFrame
        Structured dataframe with sequences
    """
    print("Extracting sequences from MEDS format...")

    schema_names = set(df.collect_schema().names())

    time_expr = pl.col("time").cast(pl.Datetime, strict=False)
    clean_code_expr = pl.col("code").cast(pl.Utf8).str.replace_all(r"[^A-Za-z0-9_]", "")
    raw_code_expr = pl.col("code").cast(pl.Utf8)

    type_expr = (
        pl.when(
            clean_code_expr.is_in(
                ["MEDS_BIRTH", "MEDS_DEATH", "[CLS]", "[PAD]", "[BOS]", "[EOS]"]
            )
        )
        .then(pl.lit(TYPE_MAP["special"]))
        .when(clean_code_expr.str.starts_with("DIAGNOSIS"))
        .then(pl.lit(TYPE_MAP["diagnosis"]))
        .when(clean_code_expr.str.starts_with("LAB"))
        .then(pl.lit(TYPE_MAP["lab"]))
        .when(clean_code_expr.str.starts_with("MEDICATION"))
        .then(pl.lit(TYPE_MAP["medication"]))
        .when(clean_code_expr.str.starts_with("PROCEDURE"))
        .then(pl.lit(TYPE_MAP["procedure"]))
        .when(clean_code_expr.str.starts_with("TRANSFER_TO"))
        .then(pl.lit(TYPE_MAP["transfer"]))
        .when(clean_code_expr.str.starts_with("ICU"))
        .then(pl.lit(TYPE_MAP["icu"]))
        .when(clean_code_expr.str.starts_with("INFUSION"))
        .then(pl.lit(TYPE_MAP["infusion"]))
        .otherwise(pl.lit(TYPE_MAP["other"]))
    )

    if "age" in schema_names:
        age_expr = (
            pl.col("age")
            .cast(pl.Int64, strict=False)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )
    elif {"anchor_age", "anchor_year"}.issubset(schema_names):
        age_expr = (
            pl.col("anchor_age")
            .cast(pl.Int64, strict=False)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            + (
                time_expr.dt.year()
                - pl.col("anchor_year")
                .cast(pl.Int64, strict=False)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
            )
        ).clip(lower_bound=0)
    else:
        age_expr = pl.lit(25, dtype=pl.Int64)

    num_patients = (
        df.select(pl.col("subject_id").n_unique().alias("n_patients"))
        .collect()
        .item()
    )
    print(f"Preparing to aggregate {num_patients} unique patients with Polars")

    sorted_df = df.sort(["subject_id", "time"])
    agg_exprs = [
        clean_code_expr.slice(0, max_len).alias(f"event_tokens_{max_len}"),
        pl.len().alias("event_count"),
        (
            pl.when(raw_code_expr == "[VS]")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .sum()
        ).alias("_num_visit_markers"),
        type_expr.cast(pl.Int64).slice(0, max_len).alias(f"type_tokens_{max_len}"),
        age_expr.cast(pl.Int64).slice(0, max_len).alias(f"age_tokens_{max_len}"),
        (
            (time_expr - time_expr.first()).dt.total_seconds() / 3600.0
        ).cast(pl.Float64).slice(0, max_len).alias(f"elapsed_tokens_{max_len}"),
        (
            pl.when(raw_code_expr == "[VS]")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cum_sum()
        ).cast(pl.Int64).slice(0, max_len).alias(f"visit_tokens_{max_len}"),
    ]
    if "time_token" in schema_names:
        agg_exprs.append(
            pl.col("time_token").slice(0, max_len).alias(f"time_tokens_{max_len}")
        )

    sequences_pl = (
        sorted_df.group_by("subject_id", maintain_order=True)
        .agg(agg_exprs)
        .with_columns(
            [
                pl.int_ranges(
                    pl.lit(0, dtype=pl.Int64),
                    pl.col(f"event_tokens_{max_len}").list.len(),
                ).alias(f"position_tokens_{max_len}"),
                pl.when(pl.col("_num_visit_markers") > 0)
                .then(pl.col("_num_visit_markers"))
                .otherwise(pl.lit(1))
                .alias("num_visits"),
            ]
        )
        .drop(["_num_visit_markers", "event_count"], strict=False)
    )

    sequences_df = sequences_pl.collect(streaming=True)
    print(f"Created sequences for {sequences_df.height} patients")
    
    return sequences_df


def create_vocabularies(df: pl.DataFrame, output_dir: str, max_len: int = 2048) -> None:
    """
    Create vocabulary files from sequences.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with patient sequences
    output_dir : str
        Directory to save vocabulary files
    max_len : int
        Maximum sequence length
    """
    print("Creating vocabularies...")
    
    vocab_dir = os.path.join(output_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)
    
    # Create vocabulary for event tokens
    event_sequences = df.get_column(f"event_tokens_{max_len}").to_list()
    ConceptTokenizer.create_vocab_from_sequences(
        sequences=event_sequences,
        save_path=os.path.join(vocab_dir, "event_vocab.json")
    )
    print(f"Created event vocabulary")
    
    # Numeric side channels are consumed directly by the model and should not
    # be added to the tokenizer vocabulary.


def add_task_labels(df: pl.DataFrame, max_len: int = 2048) -> pl.DataFrame:
    """
    Add task-specific labels for mortality, readmission, LOS, and conditions.
    
    This is a simplified version - you may need to adjust based on your actual data.
    
    Parameters
    ----------
    df : pl.DataFrame
        Dataframe with patient sequences
    max_len : int
        Maximum sequence length
        
    Returns
    -------
    pl.DataFrame
        Dataframe with added labels
    """
    print("Adding task labels...")
    return (
        df.with_columns(
            [
                pl.col(f"event_tokens_{max_len}")
                .list.eval(
                    pl.element().cast(pl.Utf8).str.contains(rf"^{DEATH_TOKEN}$|DEATH|deceased")
                )
                .list.any()
                .alias("_has_death"),
                pl.col(f"event_tokens_{max_len}")
                .list.eval(
                    pl.element()
                    .cast(pl.Utf8)
                    .str.to_uppercase()
                    .str.contains(SEPSIS_DIAGNOSIS_REGEX)
                )
                .list.any()
                .alias("_has_sepsis"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_has_death"))
                .then(pl.lit(0))
                .otherwise(pl.lit(-1))
                .alias("death_after_start"),
                pl.when(pl.col("_has_death"))
                .then(pl.lit(30))
                .otherwise(pl.lit(-1))
                .alias("death_after_end"),
                pl.when(pl.col("_has_death"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("label_mortality_2weeks"),
                pl.when(pl.col("_has_death"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("label_mortality_1month"),
                pl.when(pl.col("_has_sepsis"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("label_sepsis"),
                pl.lit([], dtype=pl.List(pl.Int64)).alias("common_conditions"),
                pl.lit([], dtype=pl.List(pl.Int64)).alias("rare_conditions"),
            ]
        )
        .drop(["_has_death", "_has_sepsis"])
    )


def create_train_test_splits(
    df: pl.DataFrame,
    output_dir: str,
    test_size: float = 0.15,
    max_len: int = 2048,
) -> Dict:
    """
    Create train/test/finetune splits.
    
    Parameters
    ----------
    df : pl.DataFrame
        Dataframe with patient sequences and labels
    output_dir : str
        Directory to save split files
    test_size : float
        Proportion of data for test set
    max_len : int
        Maximum sequence length
        
    Returns
    -------
    Dict
        Dictionary with patient ID splits
    """
    print("Creating train/test splits...")

    patient_ids = df.get_column("patient_id").to_list()
    rng = random.Random(23)
    patient_ids = list(patient_ids)
    rng.shuffle(patient_ids)
    test_count = int(test_size * len(patient_ids))
    test_ids = patient_ids[:test_count]
    pretrain_ids = patient_ids[test_count:]

    patient_ids_dict = {
        'pretrain': pretrain_ids,
        'test': test_ids,
        'finetune': {
            'few_shot': {'all': pretrain_ids}
        }
    }

    save_object_to_disk(
        patient_ids_dict,
        os.path.join(output_dir, f'dataset_{max_len}_multi_v2.pkl')
    )
    return patient_ids_dict


def main():
    print('Hello world')
    parser = argparse.ArgumentParser(
        description='Create tokenized dataset from preprocessed MEDS data'
    )
    parser.add_argument(
        '--meds_prep_dir',
        type=str,
        required=True,
        help='Directory containing preprocessed MEDS data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save tokenized dataset'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Proportion of data for test set (default: 0.15)'
    )
    parser.add_argument(
        '--patients_path',
        type=str,
        default=None,
        help='Optional path to MIMIC patients table for deriving event-level age tokens',
    )
    print('Processing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    sequence_dir = os.path.join(args.output_dir, "patient_sequences")
    os.makedirs(sequence_dir, exist_ok=True)
    sequence_file = os.path.join(sequence_dir, f"patient_sequences_{args.max_len}.parquet")
    labeled_sequence_file = os.path.join(
        sequence_dir,
        f"patient_sequences_{args.max_len}_labeled.parquet",
    )

    vocab_dir = os.path.join(args.output_dir, "vocab")
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    patient_id_dir = os.path.join(args.output_dir, "patient_id_dict")
    patient_id_file = os.path.join(
        patient_id_dir,
        f"dataset_{args.max_len}_multi_v2.pkl",
    )

    if os.path.exists(labeled_sequence_file):
        print(f"Loading labeled sequences from {labeled_sequence_file}")
        sequences_df = pl.read_parquet(labeled_sequence_file)
        required_label_columns = {
            "label_mortality_1month",
            "label_sepsis",
            "death_after_start",
            "death_after_end",
        }
        if not required_label_columns.issubset(set(sequences_df.columns)):
            print("Refreshing missing label columns in cached labeled parquet...")
            sequences_df = add_task_labels(
                ensure_patient_id_column(sequences_df),
                max_len=args.max_len,
            )
            sequences_df.write_parquet(labeled_sequence_file)
            print(f"Rewrote labeled sequences to {labeled_sequence_file}")
    else:
        if os.path.exists(sequence_file):
            print(f"Loading existing sequences from {sequence_file}")
            sequences_df = pl.read_parquet(sequence_file)
        else:
            # Step 1: Load MEDS data
            df = load_meds_data(args.meds_prep_dir)
            print('Loaded meds data')
            if args.patients_path:
                df = enrich_with_patient_metadata(df, args.patients_path)

            # Step 2: Extract sequences
            sequences_df = extract_sequences_from_meds(df, max_len=args.max_len)
            sequences_df.write_parquet(sequence_file)
            print(f"Saved intermediate sequences to {sequence_file}")

        sequences_df = ensure_patient_id_column(sequences_df)

        # Step 3: Add task labels (if applicable)
        sequences_df = add_task_labels(sequences_df, max_len=args.max_len)
        sequences_df.write_parquet(labeled_sequence_file)
        print(f"Saved labeled sequences to {labeled_sequence_file}")

    sequences_df = ensure_patient_id_column(sequences_df)

    # Step 4: Create vocabularies
    event_vocab_file = os.path.join(vocab_dir, "event_vocab.json")
    if os.path.exists(event_vocab_file):
        print(f"Using existing vocabulary at {event_vocab_file}")
    else:
        print('Creating vocabularies')
        create_vocabularies(sequences_df, args.output_dir, max_len=args.max_len)

    # Step 5: Initialize tokenizer
    if os.path.exists(tokenizer_dir):
        print(f"Tokenizer directory already exists at {tokenizer_dir}")
    else:
        print("Initializing tokenizer...")
        tokenizer = ConceptTokenizer(data_dir=vocab_dir)
        tokenizer.fit_on_vocab(with_tasks=True)
        tokenizer.save(tokenizer_dir)
        print(f"Saved tokenizer to {tokenizer_dir}")

    # Step 6: Create train/test splits
    os.makedirs(patient_id_dir, exist_ok=True)
    if os.path.exists(patient_id_file):
        print(f"Using existing patient id dict at {patient_id_file}")
        with open(patient_id_file, "rb") as file:
            patient_ids_dict = pickle.load(file)
    else:
        patient_ids_dict = create_train_test_splits(
            df=sequences_df,
            output_dir=patient_id_dir,
            test_size=args.test_size,
            max_len=args.max_len,
        )

    # Preserve the original expected output path for compatibility.
    if not os.path.exists(sequence_file):
        sequences_df.write_parquet(sequence_file)
        print(f"Saved patient sequences to {sequence_file}")

    print("\n" + "="*60)
    print("Dataset creation completed!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Patient sequences: {sequence_file}")
    print(f"Patient ID dict: {os.path.join(args.output_dir, 'patient_id_dict', f'dataset_{args.max_len}_multi_v2.pkl')}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"\nTotal patients: {sequences_df.height}")
    print(f"Train patients: {len(patient_ids_dict['pretrain'])}")
    print(f"Test patients: {len(patient_ids_dict['test'])}")


if __name__ == "__main__":
    main()
    
# python 
