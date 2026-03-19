"""Precompute resized image tensors for stay-level fusion training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_image_index(image_index_path: Path) -> pd.DataFrame:
    """Read a stay-image index from parquet or csv."""
    if image_index_path.suffix == ".csv":
        return pd.read_csv(image_index_path)
    return pd.read_parquet(image_index_path)


def resolve_path(path_value: str, image_root: Path | None) -> Path:
    """Resolve a relative or absolute image path."""
    path = Path(path_value)
    if image_root is not None and not path.is_absolute():
        path = image_root / path
    return path


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    """Load one JPG and convert it into a fixed-size CHW uint8 tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def build_tensor_cache(
    image_index: pd.DataFrame,
    output_dir: Path,
    image_root: Path | None,
    image_size: int,
    overwrite: bool,
) -> pd.DataFrame:
    """Write cached `.pt` tensors and add tensor_path / tensor_exists columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / f"tensors_{image_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    index = image_index.copy()
    tensor_paths: list[str | None] = []
    tensor_exists: list[bool] = []

    for row in index.itertuples(index=False):
        jpg_path = getattr(row, "jpg_path", None)
        study_id = getattr(row, "study_id", None)
        dicom_id = getattr(row, "dicom_id", None)
        if not isinstance(jpg_path, str) or not jpg_path:
            tensor_paths.append(None)
            tensor_exists.append(False)
            continue

        resolved_jpg = resolve_path(jpg_path, image_root)
        if not resolved_jpg.exists():
            tensor_paths.append(None)
            tensor_exists.append(False)
            continue

        stem = str(study_id) if pd.notna(study_id) else str(dicom_id)
        tensor_path = cache_dir / f"{stem}.pt"

        if overwrite or not tensor_path.exists():
            tensor = preprocess_image(resolved_jpg, image_size=image_size)
            torch.save(tensor, tensor_path)

        tensor_paths.append(str(tensor_path))
        tensor_exists.append(True)

    index["tensor_path"] = tensor_paths
    index["tensor_exists"] = tensor_exists
    return index


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index-path", type=Path, required=True)
    parser.add_argument("--output-index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_index = load_image_index(args.image_index_path)
    cached_index = build_tensor_cache(
        image_index=image_index,
        output_dir=args.output_dir,
        image_root=args.image_root,
        image_size=args.image_size,
        overwrite=args.overwrite,
    )

    args.output_index_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_index_path.suffix == ".csv":
        cached_index.to_csv(args.output_index_path, index=False)
    else:
        cached_index.to_parquet(args.output_index_path, index=False)

    print(f"Saved cached image index to {args.output_index_path}")
    print(f"Tensor rows: {int(cached_index['tensor_exists'].sum()):,}")


if __name__ == "__main__":
    main()
