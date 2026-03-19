#!/usr/bin/env python3
"""Repair age_tokens_2048 in a saved patient sequence parquet."""

import argparse
from pathlib import Path

import polars as pl


HOURS_PER_YEAR = 24.0 * 365.25


def repair_age_tokens(parquet_path: Path) -> None:
    df = pl.read_parquet(parquet_path)

    required_columns = {"elapsed_tokens_2048", "event_tokens_2048"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    repaired = df.with_columns(
        pl.col("elapsed_tokens_2048")
        .list.eval((pl.element().clip(lower_bound=0.0) / HOURS_PER_YEAR).floor().cast(pl.Int64))
        .alias("age_tokens_2048")
    )

    repaired.write_parquet(parquet_path)
    print(f"Overwrote parquet with repaired age tokens: {parquet_path}")
    print(f"Rows: {repaired.height}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair age_tokens_2048 in-place.")
    parser.add_argument("parquet_path", type=Path, help="Path to the parquet file to rewrite")
    args = parser.parse_args()
    repair_age_tokens(args.parquet_path)


if __name__ == "__main__":
    main()
