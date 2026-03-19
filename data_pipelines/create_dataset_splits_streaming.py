"""
Memory-efficient script to create dataset splits from large parquet files.

Uses streaming/chunked processing to avoid loading entire file into RAM.

Usage:
    python create_dataset_splits_streaming.py \
        --patient_sequences /path/to/patient_sequences_2048.parquet \
        --output_dir /path/to/output \
        --max_len 2048 \
        --test_size 0.15
"""

import argparse
import os
import pickle
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Tuple, Dict, Set
from sklearn.model_selection import train_test_split


def get_all_patient_ids_streaming(parquet_file: str, batch_size: int = 10000) -> List:
    """
    Get all unique patient IDs from parquet file without loading entire file.
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file
    batch_size : int
        Number of rows to read at once
        
    Returns
    -------
    List
        Unique patient IDs
    """
    print(f"Scanning parquet file for patient IDs: {parquet_file}")
    print(f"Reading in batches of {batch_size} rows...")
    
    # Open parquet file
    parquet_file_obj = pq.ParquetFile(parquet_file)
    
    # Collect unique patient IDs
    patient_ids = set()
    total_rows = 0
    
    # Iterate through row groups
    for batch_idx, batch in enumerate(parquet_file_obj.iter_batches(batch_size=batch_size)):
        # Convert to pandas for easier processing
        df_batch = batch.to_pandas()
        
        # Extract patient IDs
        batch_ids = df_batch['subject_id'].unique()
        patient_ids.update(batch_ids)
        
        total_rows += len(df_batch)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {total_rows:,} rows, found {len(patient_ids):,} unique patients")
    
    patient_ids_list = sorted(list(patient_ids))
    
    print(f"\n✓ Scan complete:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique patients: {len(patient_ids_list):,}")
    
    return patient_ids_list


def get_patient_labels_streaming(
    parquet_file: str,
    label_col: str,
    patient_ids: Set,
    batch_size: int = 10000
) -> Dict:
    """
    Get labels for specific patients from parquet file.
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file
    label_col : str
        Label column name
    patient_ids : Set
        Set of patient IDs to get labels for
    batch_size : int
        Batch size for reading
        
    Returns
    -------
    Dict
        Dictionary mapping patient_id -> label
    """
    print(f"  Extracting {label_col} labels for {len(patient_ids):,} patients...")
    
    parquet_file_obj = pq.ParquetFile(parquet_file)
    patient_labels = {}
    
    for batch in parquet_file_obj.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        
        # Filter to relevant patients
        mask = df_batch['subject_id'].isin(patient_ids)
        relevant = df_batch[mask]
        
        if len(relevant) == 0:
            continue
        
        # Check if label column exists
        if label_col not in relevant.columns:
            continue
        
        # Get labels (take first label per patient if multiple rows)
        for pid, group in relevant.groupby('subject_id'):
            if pid not in patient_labels:
                label = group[label_col].iloc[0]
                # Only store if valid label
                if pd.notna(label) and label != -1:
                    patient_labels[pid] = label
    
    print(f"    Found {len(patient_labels):,} patients with valid {label_col} labels")
    return patient_labels


def get_pretrain_test_split(
    patient_ids: List,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[List, List]:
    """
    Split patient IDs into pretrain and test sets.
    
    Parameters
    ----------
    patient_ids : List
        All patient IDs
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[List, List]
        pretrain_ids, test_ids
    """
    pretrain_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Pretrain: {len(pretrain_ids):,} patients ({100*(1-test_size):.1f}%)")
    print(f"  Test: {len(test_ids):,} patients ({100*test_size:.1f}%)")
    
    return pretrain_ids, test_ids


def get_finetune_split_streaming(
    parquet_file: str,
    pretrain_ids: List,
    label_col: str,
    finetune_sizes: List[int] = [100, 500, 1000],
    batch_size: int = 10000,
    random_state: int = 42
) -> Dict:
    """
    Create few-shot finetune splits using streaming.
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file
    pretrain_ids : List
        Patient IDs from pretrain set
    label_col : str
        Label column for the task
    finetune_sizes : List[int]
        Sizes for few-shot splits
    batch_size : int
        Batch size for reading parquet
    random_state : int
        Random seed
        
    Returns
    -------
    Dict
        Dictionary with few-shot splits
    """
    # Get labels for pretrain patients
    pretrain_ids_set = set(pretrain_ids)
    patient_labels = get_patient_labels_streaming(
        parquet_file, label_col, pretrain_ids_set, batch_size
    )
    
    if len(patient_labels) == 0:
        print(f"  Warning: No patients with valid {label_col} labels found")
        return {}
    
    # Convert to lists for sklearn
    valid_patient_ids = list(patient_labels.keys())
    labels = [patient_labels[pid] for pid in valid_patient_ids]
    
    few_shot_splits = {}
    
    for size in finetune_sizes:
        if size > len(valid_patient_ids):
            print(f"  Warning: Requested size {size} exceeds available patients {len(valid_patient_ids)}")
            continue
        
        try:
            # Stratified split for train/val
            train_ids, val_ids = train_test_split(
                valid_patient_ids,
                train_size=size,
                random_state=random_state,
                stratify=labels
            )
            
            few_shot_splits[size] = {
                'train': train_ids,
                'val': val_ids
            }
            
            print(f"    {size}-shot: {len(train_ids)} train, {len(val_ids)} val")
        except Exception as e:
            print(f"  Warning: Could not create {size}-shot split: {e}")
            continue
    
    return few_shot_splits


def create_dataset_splits_streaming(
    parquet_file: str,
    output_dir: str,
    max_len: int = 2048,
    test_size: float = 0.15,
    tasks: List[str] = None,
    batch_size: int = 10000
) -> Dict:
    """
    Create complete dataset splits dictionary using streaming.
    
    Parameters
    ----------
    parquet_file : str
        Path to patient sequences parquet file
    output_dir : str
        Output directory
    max_len : int
        Maximum sequence length
    test_size : float
        Test set proportion
    tasks : List[str], optional
        List of task label columns to create finetune splits for
    batch_size : int
        Batch size for reading parquet file
        
    Returns
    -------
    Dict
        Complete patient_ids_dict
    """
    print("\n" + "="*70)
    print("CREATING DATASET SPLITS (STREAMING MODE)")
    print("="*70)
    
    # Step 1: Get all patient IDs
    patient_ids = get_all_patient_ids_streaming(parquet_file, batch_size)
    
    # Step 2: Create pretrain/test split
    pretrain_ids, test_ids = get_pretrain_test_split(
        patient_ids=patient_ids,
        test_size=test_size
    )
    
    # Initialize patient_ids_dict
    patient_ids_dict = {
        'pretrain': pretrain_ids,
        'test': test_ids,
        'finetune': {
            'few_shot': {}
        }
    }
    
    # Step 3: If tasks are specified, create finetune splits
    if tasks:
        print("\n" + "="*70)
        print("Creating finetune splits for tasks:")
        print("="*70)
        
        for task in tasks:
            print(f"\nTask: {task}")
            few_shot_splits = get_finetune_split_streaming(
                parquet_file=parquet_file,
                pretrain_ids=pretrain_ids,
                label_col=task,
                finetune_sizes=[100, 500, 1000],
                batch_size=batch_size
            )
            
            if few_shot_splits:
                patient_ids_dict['finetune']['few_shot'][task] = few_shot_splits
    
    # Step 4: Save the dictionary
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'dataset_{max_len}_multi_v2.pkl')
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(patient_ids_dict, f)
    
    print(f"✓ Saved dataset splits to {output_file}")
    
    return patient_ids_dict


def print_summary(patient_ids_dict: Dict):
    """Print summary of the splits."""
    print("\n" + "="*70)
    print("DATASET SPLITS SUMMARY")
    print("="*70)
    
    print(f"\nPretrain patients: {len(patient_ids_dict['pretrain']):,}")
    print(f"Test patients: {len(patient_ids_dict['test']):,}")
    print(f"Total patients: {len(patient_ids_dict['pretrain']) + len(patient_ids_dict['test']):,}")
    
    if patient_ids_dict['finetune']['few_shot']:
        print("\nFinetune splits:")
        for task, splits in patient_ids_dict['finetune']['few_shot'].items():
            print(f"\n  {task}:")
            for size, split in sorted(splits.items()):
                print(f"    {size}-shot: {len(split['train'])} train, {len(split['val'])} val")
    else:
        print("\nNo finetune splits created")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset splits pickle file (streaming mode for large files)'
    )
    parser.add_argument(
        '--patient_sequences',
        type=str,
        required=True,
        help='Path to patient_sequences parquet file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the pickle file'
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
        '--batch_size',
        type=int,
        default=10000,
        help='Number of rows to read at once (default: 10000)'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Task label columns for finetune splits (e.g., label_mortality_1month)'
    )
    
    args = parser.parse_args()
    
    # Verify file exists
    if not os.path.exists(args.patient_sequences):
        raise FileNotFoundError(f"File not found: {args.patient_sequences}")
    
    # Create splits
    patient_ids_dict = create_dataset_splits_streaming(
        parquet_file=args.patient_sequences,
        output_dir=args.output_dir,
        max_len=args.max_len,
        test_size=args.test_size,
        tasks=args.tasks,
        batch_size=args.batch_size
    )
    
    # Print summary
    print_summary(patient_ids_dict)
    
    print("\n✓ Dataset splits creation completed!")


if __name__ == "__main__":
    main()