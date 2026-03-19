"""Utilities for building stay-level multimodal datasets."""

from __future__ import annotations

from typing import Any, Iterable, List

import pandas as pd


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


def normalize_bool_series(values: pd.Series) -> pd.Series:
    """Normalize common string and numeric boolean representations."""
    if values.dtype == bool:
        return values.fillna(False)

    string_values = values.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "yes", "y"}
    return string_values.isin(true_values)


def build_age_tokens(
    ages: Iterable[Any],
    timestamps: Iterable[pd.Timestamp],
    event_tokens: List[str],
    max_len: int,
    default_age: int = 25,
) -> List[int]:
    """Build per-token age values using true age when available."""
    token_count = min(len(event_tokens), max_len)
    age_series = pd.to_numeric(pd.Series(list(ages)), errors="coerce").ffill().bfill()
    if not age_series.empty and age_series.notna().any():
        return age_series.fillna(age_series.iloc[0]).astype(int).tolist()[:max_len]

    time_series = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    valid_times = time_series.dropna()
    if valid_times.empty:
        return [default_age] * token_count

    start_time = valid_times.min()
    elapsed_hours = [
        (timestamp - start_time).total_seconds() / 3600 if pd.notna(timestamp) else 0.0
        for timestamp in time_series
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

    return [default_age] * token_count


def sanitize_event_token(token: Any) -> str:
    """Convert MEDS event codes to tokenizer-friendly strings."""
    token = "" if token is None else str(token)
    return "".join(char for char in token if char.isalnum() or char == "_")
