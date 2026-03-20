#!/usr/bin/env python3
"""Read a patient-sequence parquet, repair nested/derived fields, and write a new parquet."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import polars as pl


HOURS_PER_YEAR = 24.0 * 365.25
DEATH_TOKEN = "MEDS_DEATH"
SEPSIS_PATTERN = re.compile(r"^DIAGNOSIS(?:038|A40|A41)", re.IGNORECASE)

TOKEN_COLUMNS = [
    "event_tokens_2048",
    "type_tokens_2048",
    "age_tokens_2048",
    "position_tokens_2048",
    "elapsed_tokens_2048",
    "visit_tokens_2048",
]


def flatten_singleton_nested(value: Any) -> Any:
    """Flatten `[[...]]` into `[...]` for malformed list columns."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def normalize_events(events: Any) -> list[Any]:
    """Convert Polars/Python list-like values into a plain Python list."""
    if events is None:
        return []
    if isinstance(events, pl.Series):
        return events.to_list()
    if isinstance(events, list):
        return events
    return [events]


def has_death(events: Any) -> bool:
    """Return whether a token list contains a death indicator."""
    for token in normalize_events(events):
        token_str = str(token)
        if token_str == DEATH_TOKEN or "DEATH" in token_str or "deceased" in token_str.lower():
            return True
    return False


def has_sepsis(events: Any) -> bool:
    """Return whether a token list contains a sepsis diagnosis code."""
    for token in normalize_events(events):
        if isinstance(token, str) and SEPSIS_PATTERN.match(token):
            return True
    return False


def repair_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Return a repaired copy of the patient-sequence dataframe."""
    required_columns = {"event_tokens_2048", "elapsed_tokens_2048"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in TOKEN_COLUMNS:
        if column in df.columns:
            repaired_values = [
                flatten_singleton_nested(value)
                for value in df.get_column(column).to_list()
            ]
            df = df.with_columns(pl.Series(column, repaired_values))

    if "patient_id" not in df.columns and "subject_id" in df.columns:
        df = df.with_columns(pl.col("subject_id").alias("patient_id"))

    df = df.with_columns(
        pl.col("elapsed_tokens_2048")
        .list.eval((pl.element().clip(lower_bound=0.0) / HOURS_PER_YEAR).floor().cast(pl.Int64))
        .alias("age_tokens_2048")
    )

    df = df.with_columns(
        [
            pl.col("event_tokens_2048")
            .map_elements(has_death, return_dtype=pl.Boolean)
            .alias("_has_death"),
            pl.col("event_tokens_2048")
            .map_elements(has_sepsis, return_dtype=pl.Boolean)
            .alias("_has_sepsis"),
        ]
    )

    return df.with_columns(
        [
            pl.when(pl.col("_has_death")).then(pl.lit(0)).otherwise(pl.lit(-1)).alias("death_after_start"),
            pl.when(pl.col("_has_death")).then(pl.lit(30)).otherwise(pl.lit(-1)).alias("death_after_end"),
            pl.when(pl.col("_has_death")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("label_mortality_2weeks"),
            pl.when(pl.col("_has_death")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("label_mortality_1month"),
            pl.when(pl.col("_has_sepsis")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("label_sepsis"),
        ]
    ).drop(["_has_death", "_has_sepsis"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair an existing patient sequence parquet and write a new parquet."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input parquet file")
    parser.add_argument("--output-path", type=Path, required=True, help="Output parquet file")
    args = parser.parse_args()

    df = pl.read_parquet(args.input_path)
    repaired = repair_dataframe(df)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write_parquet(args.output_path)

    print(f"Read: {args.input_path}")
    print(f"Wrote repaired parquet: {args.output_path}")
    print(f"Rows: {repaired.height}")


if __name__ == "__main__":
    main()
