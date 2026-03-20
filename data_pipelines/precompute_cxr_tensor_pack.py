#!/usr/bin/env python3
"""Precompute portable CXR tensor packs from a metadata CSV and JPG tree."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def load_cxr_metadata(csv_path: Path) -> pd.DataFrame:
    """Load and normalize the source CXR metadata CSV."""
    df = pd.read_csv(csv_path)

    unnamed_columns = [column for column in df.columns if column.startswith("Unnamed") or column == ""]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if "study_id" in df.columns:
        df["study_id"] = pd.to_numeric(df["study_id"], errors="coerce").astype("Int64")
    if "subject_id" in df.columns:
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    if "hadm_id" in df.columns:
        df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    if "StudyDateTime" in df.columns:
        df["StudyDateTime"] = pd.to_datetime(df["StudyDateTime"], errors="coerce")

    if "jpg_path_exists" in df.columns:
        truthy = {"1", "true", "t", "yes", "y"}
        df["jpg_path_exists"] = (
            df["jpg_path_exists"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(truthy)
        )
    else:
        df["jpg_path_exists"] = df["jpg_path"].notna()

    return df


def resolve_image_path(path_value: str, image_root: Path | None) -> Path:
    """Resolve a CSV-provided image path to an absolute filesystem path."""
    path = Path(path_value)
    if image_root is not None and not path.is_absolute():
        path = image_root / path
    return path


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    """Load one JPG and return a resized CHW uint8 tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def safe_scalar(value: Any) -> Any:
    """Convert pandas NA-like values into plain Python scalars."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_shard(
    output_dir: Path,
    shard_idx: int,
    images: list[torch.Tensor],
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write one tensor shard and return its manifest rows."""
    shard_dir = output_dir / "tensor_packs"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_name = f"cxr_tensor_pack_{shard_idx:05d}.pt"
    shard_path = shard_dir / shard_name

    payload = {
        "images": torch.stack(images, dim=0),
        "metadata": metadata_rows,
    }
    torch.save(payload, shard_path)

    manifest_rows = []
    for item_idx, metadata in enumerate(metadata_rows):
        manifest_rows.append(
            {
                **metadata,
                "tensor_pack_path": str(Path("tensor_packs") / shard_name),
                "tensor_pack_index": item_idx,
            }
        )
    return manifest_rows


def append_manifest_rows(rows: list[dict[str, Any]], manifest_csv_path: Path) -> None:
    """Append shard manifest rows to a CSV so progress is durable."""
    if not rows:
        return
    manifest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    shard_df = pd.DataFrame(rows)
    write_header = not manifest_csv_path.exists()
    shard_df.to_csv(manifest_csv_path, mode="a", header=write_header, index=False)


def load_existing_manifest(manifest_csv_path: Path) -> pd.DataFrame:
    """Load an existing incremental manifest if present."""
    if not manifest_csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_csv_path)


def write_summary(
    summary_path: Path,
    requested_rows: int,
    cached_rows: int,
    missing_count: int,
    failed_count: int,
    image_size: int,
    shard_size: int,
    shard_count: int,
) -> None:
    """Write a human-readable progress summary."""
    summary_path.write_text(
        "\n".join(
            [
                f"requested_rows={requested_rows}",
                f"cached_rows={cached_rows:,}",
                f"missing_source_image={missing_count:,}",
                f"failed_preprocess={failed_count:,}",
                f"image_size={image_size}",
                f"shard_size={shard_size}",
                f"written_shards={shard_count}",
            ]
        )
    )


def build_tensor_packs(
    df: pd.DataFrame,
    output_dir: Path,
    image_root: Path | None,
    image_size: int,
    shard_size: int,
    limit: int | None,
    resume: bool,
) -> pd.DataFrame:
    """Create sharded image tensor packs and return a manifest dataframe."""
    keep = df["jpg_path"].notna() & df["jpg_path_exists"]
    df = df.loc[keep].copy()
    if limit is not None:
        df = df.head(limit).copy()

    manifest_csv_path = output_dir / "cxr_tensor_manifest_partial.csv"
    summary_path = output_dir / "summary.txt"

    existing_manifest = load_existing_manifest(manifest_csv_path) if resume else pd.DataFrame()
    shard_idx = 0
    if not existing_manifest.empty:
        processed_paths = set(existing_manifest["jpg_path"].dropna().astype(str).tolist())
        df = df.loc[~df["jpg_path"].astype(str).isin(processed_paths)].copy()
        if "tensor_pack_path" in existing_manifest.columns:
            shard_idx = existing_manifest["tensor_pack_path"].dropna().astype(str).nunique()

    iterable = df.itertuples(index=False)
    total = len(df)
    if tqdm is not None:
        iterable = tqdm(iterable, total=total, desc="Precomputing CXR tensors")

    images: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    missing_count = 0
    failed_count = 0
    cached_rows = int(len(existing_manifest))

    for row in iterable:
        jpg_path = getattr(row, "jpg_path", None)
        if not isinstance(jpg_path, str) or not jpg_path:
            missing_count += 1
            continue

        resolved = resolve_image_path(jpg_path, image_root)
        if not resolved.exists():
            missing_count += 1
            continue

        try:
            tensor = preprocess_image(resolved, image_size=image_size)
        except Exception:
            failed_count += 1
            continue

        images.append(tensor)
        metadata_rows.append(
            {
                "study_id": safe_scalar(getattr(row, "study_id", None)),
                "dicom_id": safe_scalar(getattr(row, "dicom_id", None)),
                "subject_id": safe_scalar(getattr(row, "subject_id", None)),
                "hadm_id": safe_scalar(getattr(row, "hadm_id", None)),
                "StudyDateTime": safe_scalar(getattr(row, "StudyDateTime", None)),
                "jpg_path": str(jpg_path),
            }
        )

        if len(images) >= shard_size:
            shard_rows = write_shard(output_dir, shard_idx, images, metadata_rows)
            append_manifest_rows(shard_rows, manifest_csv_path)
            cached_rows += len(shard_rows)
            shard_idx += 1
            images = []
            metadata_rows = []
            write_summary(
                summary_path=summary_path,
                requested_rows=total,
                cached_rows=cached_rows,
                missing_count=missing_count,
                failed_count=failed_count,
                image_size=image_size,
                shard_size=shard_size,
                shard_count=shard_idx,
            )

    if images:
        shard_rows = write_shard(output_dir, shard_idx, images, metadata_rows)
        append_manifest_rows(shard_rows, manifest_csv_path)
        cached_rows += len(shard_rows)
        shard_idx += 1

    write_summary(
        summary_path=summary_path,
        requested_rows=total,
        cached_rows=cached_rows,
        missing_count=missing_count,
        failed_count=failed_count,
        image_size=image_size,
        shard_size=shard_size,
        shard_count=shard_idx,
    )

    manifest = load_existing_manifest(manifest_csv_path)
    if not manifest.empty:
        manifest["image_size"] = image_size
        manifest["tensor_dtype"] = "uint8"
        manifest = manifest.drop_duplicates(
            subset=["jpg_path", "tensor_pack_path", "tensor_pack_index"],
            keep="last",
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-csv-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_cxr_metadata(args.cxr_csv_path)
    manifest = build_tensor_packs(
        df=metadata,
        output_dir=args.output_dir,
        image_root=args.image_root,
        image_size=args.image_size,
        shard_size=args.shard_size,
        limit=args.limit,
        resume=args.resume,
    )

    manifest_csv_path = args.output_dir / "cxr_tensor_manifest_partial.csv"
    manifest_path = args.output_dir / "cxr_tensor_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    print(f"Saved manifest to {manifest_path}")
    print(f"Incremental manifest csv: {manifest_csv_path}")
    print(f"Cached studies: {len(manifest):,}")
    print(f"Tensor pack dir: {args.output_dir / 'tensor_packs'}")


if __name__ == "__main__":
    main()
