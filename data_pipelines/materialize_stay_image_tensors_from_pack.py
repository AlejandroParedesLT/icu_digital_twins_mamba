#!/usr/bin/env python3
"""Materialize stay-matched image tensors from portable tensor packs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def load_table(path: Path) -> pd.DataFrame:
    """Load parquet or csv."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save parquet or csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def materialize_tensor_cache(
    stay_image_index_path: Path,
    manifest_path: Path,
    pack_root: Path,
    output_dir: Path,
    output_index_path: Path,
) -> None:
    """Create per-study `.pt` tensors for only the studies used by the stay index."""
    stay_index = load_table(stay_image_index_path).copy()
    manifest = load_table(manifest_path).copy()

    if "study_id" in stay_index.columns:
        stay_index["study_id"] = pd.to_numeric(stay_index["study_id"], errors="coerce").astype("Int64")
    if "study_id" in manifest.columns:
        manifest["study_id"] = pd.to_numeric(manifest["study_id"], errors="coerce").astype("Int64")

    merged = stay_index.merge(
        manifest[["study_id", "tensor_pack_path", "tensor_pack_index"]],
        on="study_id",
        how="left",
    )

    tensor_dir = output_dir / "tensors_from_pack"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    loaded_packs: dict[Path, dict] = {}
    tensor_paths: list[str | None] = []
    tensor_exists: list[bool] = []

    for row_idx, row in enumerate(merged.itertuples(index=False)):
        pack_rel = getattr(row, "tensor_pack_path", None)
        pack_idx = getattr(row, "tensor_pack_index", None)
        study_id = getattr(row, "study_id", None)

        if not isinstance(pack_rel, str) or pd.isna(pack_idx):
            tensor_paths.append(None)
            tensor_exists.append(False)
            continue

        pack_path = pack_root / pack_rel
        if pack_path not in loaded_packs:
            loaded_packs[pack_path] = torch.load(pack_path, map_location="cpu")

        payload = loaded_packs[pack_path]
        tensor = payload["images"][int(pack_idx)]

        if pd.notna(study_id):
            study_stem = str(int(study_id))
        else:
            study_stem = f"row_{row_idx:08d}"

        tensor_path = tensor_dir / f"{study_stem}.pt"
        if not tensor_path.exists():
            torch.save(tensor, tensor_path)

        tensor_paths.append(str(tensor_path))
        tensor_exists.append(True)

    merged["tensor_path"] = tensor_paths
    merged["tensor_exists"] = tensor_exists
    save_table(merged, output_index_path)

    print(f"Saved cached stay image index to {output_index_path}")
    print(f"Tensor rows materialized: {int(pd.Series(tensor_exists).sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stay-image-index-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-index-path", type=Path, required=True)
    args = parser.parse_args()

    materialize_tensor_cache(
        stay_image_index_path=args.stay_image_index_path,
        manifest_path=args.manifest_path,
        pack_root=args.pack_root,
        output_dir=args.output_dir,
        output_index_path=args.output_index_path,
    )


if __name__ == "__main__":
    main()
