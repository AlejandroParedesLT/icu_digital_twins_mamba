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
from pathlib import Path
from typing import Dict, List

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


def load_meds_data(meds_prep_dir: str) -> pd.DataFrame:
    """
    Load preprocessed MEDS data from parquet files.
    
    Parameters
    ----------
    meds_prep_dir : str
        Directory containing preprocessed MEDS parquet files
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all patient sequences
    """
    print(f"Loading MEDS data from {meds_prep_dir}...")
    
    # Look for parquet files in the data directory
    parquet_pattern = os.path.join(meds_prep_dir, "**", "*.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {meds_prep_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load all parquet files using polars for efficiency
    dfs = []
    for file in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pl.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    # Concatenate all dataframes
    combined_df = pl.concat(dfs)
    
    # Convert to pandas for compatibility with existing code
    df = combined_df.to_pandas()
    
    print(f"Loaded {len(df)} patient records")
    return df


def extract_sequences_from_meds(df: pd.DataFrame, max_len: int = 2048) -> pd.DataFrame:
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
    
    # Group by patient_id and aggregate sequences
    # Note: Adjust these column names based on your actual MEDS output
    patient_sequences = []
    import re
    for i, (patient_id, group) in enumerate(tqdm(df.groupby('subject_id'), desc="Processing patients")):
        # Sort by timestamp
        group = group.sort_values('time')
        
        # Extract different token types
        event_tokens = [re.sub(r'[^A-Za-z0-9_]', '', str(t)) for t in group['code'].tolist()]
        
        # Create a record for this patient
        record = {
            'subject_id': patient_id,
            f'event_tokens_{max_len}': event_tokens[:max_len],
            'num_visits': len(group[group['code'] == '[VS]']) if '[VS]' in event_tokens else 1,
        }
        
        # Add other token types if available
        if 'code_type' in group.columns:
            record[f'type_tokens_{max_len}'] = group['code_type'].tolist()[:max_len]
        
        if 'age' in group.columns:
            record[f'age_tokens_{max_len}'] = group['age'].tolist()[:max_len]
            
        if 'time_token' in group.columns:
            record[f'time_tokens_{max_len}'] = group['time_token'].tolist()[:max_len]
        
        # Position and visit tokens (if not present, create them)
        record[f'position_tokens_{max_len}'] = list(range(len(event_tokens[:max_len])))
        
        # Calculate elapsed time tokens (hours since start)
        if 'time' in group.columns:
            times = pd.to_datetime(group['time'])
            start_time = times.min()
            elapsed_hours = [(t - start_time).total_seconds() / 3600 for t in times]
            record[f'elapsed_tokens_{max_len}'] = elapsed_hours[:max_len]
        
        # Visit segments (simplified - assign visit number to each event)
        visit_segments = []
        current_visit = 0
        for token in event_tokens[:max_len]:
            if token == '[VS]':
                current_visit += 1
            visit_segments.append(current_visit)
        record[f'visit_tokens_{max_len}'] = visit_segments
        
        patient_sequences.append(record)

        if i==100:
            break
    
    sequences_df = pd.DataFrame(patient_sequences)
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
    
    # Create vocabularies for other token types if present
    for token_type in ['type', 'age', 'time']:
        col_name = f'{token_type}_tokens_{max_len}'
        if col_name in df.columns:
            ConceptTokenizer.create_vocab_from_sequences(
                sequences=df[col_name],
                save_path=os.path.join(vocab_dir, f"{token_type}_vocab.json")
            )
            print(f"Created {token_type} vocabulary")


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
            'few_shot': {}
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
    print('Processing')
    
    args = parser.parse_args()
    
    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    
    # # Step 1: Load MEDS data
    # df = load_meds_data(args.meds_prep_dir)
    # print('LOaded meds data')
    # # Step 2: Extract sequences
    # sequences_df = extract_sequences_from_meds(df, max_len=args.max_len)
    # print('Vocabularies')
    # # Step 3: Create vocabularies
    # vocab_dir = os.path.join(args.output_dir, "vocab")
    # create_vocabularies(sequences_df, args.output_dir, max_len=args.max_len)
    
    # # Step 4: Initialize tokenizer
    # print("Initializing tokenizer...")
    # tokenizer = ConceptTokenizer(data_dir=vocab_dir)
    # tokenizer.fit_on_vocab(with_tasks=True)
    
    # # Save tokenizer
    # tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    # tokenizer.save(tokenizer_dir)
    # print(f"Saved tokenizer to {tokenizer_dir}")
    
    # # Step 5: Add task labels (if applicable)
    # sequences_df = add_task_labels(sequences_df, max_len=args.max_len)
    
    # Step 6: Save patient sequences as parquet
    sequence_dir = os.path.join(args.output_dir, "patient_sequences")
    os.makedirs(sequence_dir, exist_ok=True)
    sequence_file = os.path.join(sequence_dir, f"patient_sequences_{args.max_len}.parquet")
    sequences_df.to_parquet(sequence_file, index=False)
    print(f"Saved patient sequences to {sequence_file}")
    
    # Step 7: Create train/test splits
    patient_ids_dict = create_train_test_splits(
        df=sequences_df,
        output_dir=os.path.join(args.output_dir, "patient_id_dict"),
        test_size=args.test_size,
        max_len=args.max_len,
    )
    
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