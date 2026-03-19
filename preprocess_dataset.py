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
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.data.processor import (
    get_pretrain_test_split,
    get_finetune_split,
    process_condition_dataset,
    process_mortality_dataset,
    process_readmission_dataset,
    process_length_of_stay_dataset,
    process_multi_dataset,
)
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


def infer_token_type(token: Any) -> int:
    """Map event tokens to compact categorical type ids."""
    if not isinstance(token, str):
        return TYPE_MAP["other"]

    if token in {"MEDS_BIRTH", "MEDS_DEATH", "[CLS]", "[PAD]", "[BOS]", "[EOS]"}:
        return TYPE_MAP["special"]
    if token.startswith("DIAGNOSIS"):
        return TYPE_MAP["diagnosis"]
    if token.startswith("LAB"):
        return TYPE_MAP["lab"]
    if token.startswith("MEDICATION"):
        return TYPE_MAP["medication"]
    if token.startswith("PROCEDURE"):
        return TYPE_MAP["procedure"]
    if token.startswith("TRANSFER_TO"):
        return TYPE_MAP["transfer"]
    if token.startswith("ICU"):
        return TYPE_MAP["icu"]
    if token.startswith("INFUSION"):
        return TYPE_MAP["infusion"]
    return TYPE_MAP["other"]


def build_age_tokens(
    group: pd.DataFrame,
    event_tokens: List[str],
    max_len: int,
) -> List[int]:
    """Build per-token age values using true age when available."""
    token_count = min(len(event_tokens), max_len)

    if "age" in group.columns:
        age_values = pd.to_numeric(group["age"], errors="coerce").ffill().bfill()
        if not age_values.empty and age_values.notna().any():
            return age_values.fillna(age_values.iloc[0]).astype(int).tolist()[:max_len]

    if {"anchor_age", "anchor_year", "time"}.issubset(group.columns):
        anchor_age = pd.to_numeric(group["anchor_age"], errors="coerce").ffill().bfill()
        anchor_year = pd.to_numeric(group["anchor_year"], errors="coerce").ffill().bfill()
        event_year = pd.to_datetime(group["time"], errors="coerce").dt.year

        if (
            not anchor_age.empty
            and not anchor_year.empty
            and anchor_age.notna().any()
            and anchor_year.notna().any()
            and event_year.notna().any()
        ):
            age_values = (
                anchor_age.fillna(anchor_age.iloc[0])
                + (event_year - anchor_year.fillna(anchor_year.iloc[0]))
            )
            age_values = age_values.fillna(method="ffill").fillna(method="bfill")
            return age_values.clip(lower=0).astype(int).tolist()[:max_len]

    if "time" not in group.columns:
        return [25] * token_count

    times = pd.to_datetime(group["time"], errors="coerce")
    valid_times = times.dropna()
    if valid_times.empty:
        return [25] * token_count

    start_time = valid_times.min()
    elapsed_hours = [
        (timestamp - start_time).total_seconds() / 3600 if pd.notna(timestamp) else 0.0
        for timestamp in times
    ]

    birth_index = next(
        (i for i, token in enumerate(event_tokens) if token == "MEDS_BIRTH"),
        None,
    )
    if birth_index is not None and birth_index < len(elapsed_hours):
        birth_elapsed = elapsed_hours[birth_index]
        return [
            int(max((elapsed - birth_elapsed) / 24 / 365.25, 0))
            for elapsed in elapsed_hours[:max_len]
        ]

    return [25] * token_count


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


def ensure_patient_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror subject_id into patient_id for downstream Odyssey utilities."""
    if "patient_id" not in df.columns and "subject_id" in df.columns:
        df = df.copy()
        df["patient_id"] = df["subject_id"]
    return df


def extract_sequences_from_meds(df: pl.LazyFrame, max_len: int = 2048) -> pd.DataFrame:
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
    pd.DataFrame
        Structured dataframe with sequences
    """
    print("Extracting sequences from MEDS format...")

    schema_names = set(df.collect_schema().names())

    time_expr = pl.col("time").cast(pl.Datetime, strict=False)
    clean_code_expr = pl.col("code").cast(pl.Utf8).str.replace_all(r"[^A-Za-z0-9_]", "")
    raw_code_expr = pl.col("code").cast(pl.Utf8)

    type_expr = (
        pl.when(clean_code_expr.is_in(["MEDS_BIRTH", "MEDS_DEATH", "CLS", "PAD", "BOS", "EOS"]))
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
        age_expr = pl.col("age").cast(pl.Int64, strict=False).fill_null(strategy="forward").fill_null(strategy="backward")
    elif {"anchor_age", "anchor_year"}.issubset(schema_names):
        age_expr = (
            pl.col("anchor_age").cast(pl.Int64, strict=False).fill_null(strategy="forward").fill_null(strategy="backward")
            + (
                time_expr.dt.year()
                - pl.col("anchor_year").cast(pl.Int64, strict=False).fill_null(strategy="forward").fill_null(strategy="backward")
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
        pl.int_ranges(pl.lit(0), pl.len()).slice(0, max_len).alias(f"position_tokens_{max_len}"),
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
            pl.when(pl.col("_num_visit_markers") > 0)
            .then(pl.col("_num_visit_markers"))
            .otherwise(pl.lit(1))
            .alias("num_visits")
        )
        .drop(["_num_visit_markers", "event_count"], strict=False)
    )

    sequences_df = sequences_pl.collect(streaming=True).to_pandas()
    print(f"Created sequences for {len(sequences_df)} patients")
    
    return sequences_df


def create_vocabularies(df: pd.DataFrame, output_dir: str, max_len: int = 2048) -> None:
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
    event_sequences = df[f'event_tokens_{max_len}']
    ConceptTokenizer.create_vocab_from_sequences(
        sequences=event_sequences,
        save_path=os.path.join(vocab_dir, "event_vocab.json")
    )
    print(f"Created event vocabulary")
    
    # Numeric side channels are consumed directly by the model and should not
    # be added to the tokenizer vocabulary.


def add_task_labels(df: pd.DataFrame, max_len: int = 2048) -> pd.DataFrame:
    """
    Add task-specific labels for mortality, readmission, LOS, and conditions.
    
    This is a simplified version - you may need to adjust based on your actual data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with patient sequences
    max_len : int
        Maximum sequence length
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added labels
    """
    print("Adding task labels...")
    
    # Initialize label columns
    df['death_after_start'] = -1
    df['death_after_end'] = -1
    df['common_conditions'] = [np.array([], dtype=np.int64) for _ in range(len(df))]
    df['rare_conditions'] = [np.array([], dtype=np.int64) for _ in range(len(df))]
    
    # Process mortality labels
    # Note: This assumes you have death information in your MEDS data
    # You'll need to adjust based on your actual data structure
    for idx, row in df.iterrows():
        event_tokens = row[f'event_tokens_{max_len}']
        
        # Check for death indicator (adjust based on your coding)
        if 'DEATH' in str(event_tokens) or any('deceased' in str(t).lower() for t in event_tokens):
            # Simple heuristic - you'll need to calculate actual time differences
            df.at[idx, 'death_after_start'] = 0
            df.at[idx, 'death_after_end'] = 30  # Placeholder
    
    return df


def create_train_test_splits(
    df: pd.DataFrame,
    output_dir: str,
    test_size: float = 0.15,
    max_len: int = 2048,
) -> Dict:
    """
    Create train/test/finetune splits.
    
    Parameters
    ----------
    df : pd.DataFrame
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
    
    # Create pretrain/test split
    pretrain_ids, test_ids = get_pretrain_test_split(
        dataset=df,
        stratify_target=None,  # or specify a column to stratify on
        test_size=test_size,
    )
    
    patient_ids_dict = {
        'pretrain': pretrain_ids,
        'test': test_ids,
        'finetune': {
            'few_shot': {'all': pretrain_ids}
        }
    }
    
    # If you have task-specific datasets, create finetune splits
    # Example for mortality task
    try:
        # Process task datasets
        df_mortality = process_mortality_dataset(df.copy())
        df_readmission = process_readmission_dataset(df.copy(), max_len=max_len)
        df_los = process_length_of_stay_dataset(df.copy(), threshold=7, max_len=max_len)
        df_condition = process_condition_dataset(df.copy())
        
        # Create multi-task dataset
        datasets = {
            'original': df,
            'mortality': df_mortality,
            'readmission': df_readmission,
            'los': df_los,
            'condition': df_condition,
        }
        
        multi_dataset = process_multi_dataset(datasets, max_len=max_len)
        
        # Define task configurations for finetune splits
        task_config = {
            'mortality_1month': {
                'dataset': multi_dataset,
                'label_col': 'label_mortality_1month',
                'finetune_size': [100, 500, 1000],
                'save_path': os.path.join(output_dir, f'dataset_{max_len}_multi_v2.pkl'),
                'split_mode': 'single_label_stratified',
            }
        }
        
        # Create finetune splits
        patient_ids_dict = get_finetune_split(
            task_config=task_config,
            task='mortality_1month',
            patient_ids_dict=patient_ids_dict,
        )
        
    except Exception as e:
        print(f"Warning: Could not create task-specific splits: {e}")
        print("Saving basic splits only...")
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
        sequences_df = pd.read_parquet(labeled_sequence_file)
    else:
        if os.path.exists(sequence_file):
            print(f"Loading existing sequences from {sequence_file}")
            sequences_df = pd.read_parquet(sequence_file)
        else:
            # Step 1: Load MEDS data
            df = load_meds_data(args.meds_prep_dir)
            print('Loaded meds data')
            if args.patients_path:
                df = enrich_with_patient_metadata(df, args.patients_path)

            # Step 2: Extract sequences
            sequences_df = extract_sequences_from_meds(df, max_len=args.max_len)
            sequences_df.to_parquet(sequence_file, index=False)
            print(f"Saved intermediate sequences to {sequence_file}")

        sequences_df = ensure_patient_id_column(sequences_df)

        # Step 3: Add task labels (if applicable)
        sequences_df = add_task_labels(sequences_df, max_len=args.max_len)
        sequences_df.to_parquet(labeled_sequence_file, index=False)
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
        sequences_df.to_parquet(sequence_file, index=False)
        print(f"Saved patient sequences to {sequence_file}")

    print("\n" + "="*60)
    print("Dataset creation completed!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Patient sequences: {sequence_file}")
    print(f"Patient ID dict: {os.path.join(args.output_dir, 'patient_id_dict', f'dataset_{args.max_len}_multi_v2.pkl')}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"\nTotal patients: {len(sequences_df)}")
    print(f"Train patients: {len(patient_ids_dict['pretrain'])}")
    print(f"Test patients: {len(patient_ids_dict['test'])}")


if __name__ == "__main__":
    main()
    
# python 
