"""Build stay-level token sequences from MEDS event shards."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odyssey.data.stay_level import (
    TYPE_MAP,
    build_age_tokens,
    infer_token_type,
    sanitize_event_token,
)
from odyssey.data.tokenizer import ConceptTokenizer


SEPSIS_PATTERN = re.compile(r"^(DIAGNOSIS//)?(038|A40|A41)", re.IGNORECASE)


def load_master_stays(master_stays_path: Path) -> pd.DataFrame:
    """Load the stay-level master table."""
    if master_stays_path.suffix == ".csv":
        master = pd.read_csv(master_stays_path)
    else:
        master = pd.read_parquet(master_stays_path)

    for column in ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod", "death_time"]:
        if column in master.columns:
            master[column] = pd.to_datetime(master[column], errors="coerce")

    return master


def read_meds_shard(shard_path: Path) -> pd.DataFrame:
    """Read one MEDS shard and keep only the columns needed for sequence building."""
    frame = pd.read_parquet(shard_path)
    keep_columns = [
        column
        for column in ["subject_id", "hadm_id", "time", "code", "code_type", "age"]
        if column in frame.columns
    ]
    frame = frame[keep_columns].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["subject_id", "time", "code"])
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce").astype("Int64")
    if "hadm_id" in frame.columns:
        frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce").astype("Int64")
    return frame


def match_events_to_stays(
    events: pd.DataFrame,
    master_stays: pd.DataFrame,
    context_hours: float,
) -> list[dict[str, Any]]:
    """Assign MEDS events to ICU stays using subject_id, hadm_id, and time windows."""
    matched_events: list[dict[str, Any]] = []
    grouped_stays = {
        int(subject_id): group.sort_values(["intime", "outtime", "stay_id"]).copy()
        for subject_id, group in master_stays.groupby("subject_id")
    }

    for subject_id, patient_events in events.groupby("subject_id"):
        if pd.isna(subject_id):
            continue
        subject_id_int = int(subject_id)
        stays = grouped_stays.get(subject_id_int)
        if stays is None:
            continue

        patient_events = patient_events.sort_values("time").copy()
        for stay in stays.itertuples(index=False):
            window_start = stay.intime - pd.Timedelta(hours=context_hours)
            mask = (patient_events["time"] >= window_start) & (
                pd.isna(stay.outtime) or patient_events["time"] <= stay.outtime
            )
            if "hadm_id" in patient_events.columns and pd.notna(stay.hadm_id):
                hadm_mask = patient_events["hadm_id"].isna() | (
                    patient_events["hadm_id"] == int(stay.hadm_id)
                )
                mask &= hadm_mask

            stay_events = patient_events.loc[mask]
            if stay_events.empty:
                continue

            for event in stay_events.itertuples(index=False):
                matched_events.append(
                    {
                        "stay_id": int(stay.stay_id),
                        "subject_id": int(stay.subject_id),
                        "hadm_id": int(stay.hadm_id) if pd.notna(stay.hadm_id) else None,
                        "intime": stay.intime,
                        "outtime": stay.outtime,
                        "in_hosp_mortality": int(stay.in_hosp_mortality),
                        "mortality_28d": int(stay.mortality_28d),
                        "event_time": event.time,
                        "code": event.code,
                        "code_type": getattr(event, "code_type", None),
                        "age": getattr(event, "age", None),
                    }
                )

    return matched_events


def build_sequence_record(stay_events: pd.DataFrame, max_len: int) -> dict[str, Any]:
    """Convert all events for a stay into one sequence record."""
    stay_events = stay_events.sort_values("event_time").reset_index(drop=True)
    tokens = [sanitize_event_token(code) for code in stay_events["code"].tolist()]
    tokens = tokens[:max_len]

    intime = stay_events["intime"].iloc[0]
    elapsed_hours = (
        (stay_events["event_time"] - intime).dt.total_seconds() / 3600.0
    ).fillna(0.0)
    elapsed_hours = elapsed_hours.tolist()[:max_len]

    if "code_type" in stay_events.columns:
        code_type = pd.to_numeric(stay_events["code_type"], errors="coerce")
        if code_type.notna().any():
            type_tokens = code_type.fillna(TYPE_MAP["other"]).astype(int).tolist()[:max_len]
        else:
            type_tokens = [infer_token_type(token) for token in tokens]
    else:
        type_tokens = [infer_token_type(token) for token in tokens]

    age_tokens = build_age_tokens(
        ages=stay_events["age"].tolist() if "age" in stay_events.columns else [],
        timestamps=stay_events["event_time"].tolist(),
        event_tokens=tokens,
        max_len=max_len,
    )

    visit_tokens = [
        0 if event_time < intime else 1
        for event_time in stay_events["event_time"].tolist()[:max_len]
    ]

    sepsis_label = int(
        stay_events["code"].astype(str).str.match(SEPSIS_PATTERN).fillna(False).any()
    )

    return {
        "stay_id": int(stay_events["stay_id"].iloc[0]),
        "subject_id": int(stay_events["subject_id"].iloc[0]),
        "hadm_id": stay_events["hadm_id"].iloc[0],
        "intime": stay_events["intime"].iloc[0],
        "outtime": stay_events["outtime"].iloc[0],
        f"event_tokens_{max_len}": tokens,
        f"type_tokens_{max_len}": type_tokens,
        f"age_tokens_{max_len}": age_tokens,
        f"position_tokens_{max_len}": list(range(len(tokens))),
        f"elapsed_tokens_{max_len}": elapsed_hours,
        f"visit_tokens_{max_len}": visit_tokens,
        "num_visits": 1,
        "label_in_hosp_mortality": int(stay_events["in_hosp_mortality"].max()),
        "label_mortality_28d": int(stay_events["mortality_28d"].max()),
        "label_sepsis": sepsis_label,
    }


def build_stay_sequences(
    meds_data_dir: Path,
    master_stays: pd.DataFrame,
    max_len: int,
    context_hours: float,
) -> pd.DataFrame:
    """Build one token sequence per ICU stay."""
    shards = sorted(meds_data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {meds_data_dir}")

    matched_rows: list[dict[str, Any]] = []
    for index, shard_path in enumerate(shards, start=1):
        shard_events = read_meds_shard(shard_path)
        matched_rows.extend(
            match_events_to_stays(
                events=shard_events,
                master_stays=master_stays,
                context_hours=context_hours,
            )
        )
        if index % 10 == 0:
            print(f"Processed {index}/{len(shards)} MEDS shards")

    if not matched_rows:
        raise RuntimeError("No stay-aligned events were found. Check join keys and time windows.")

    matched_df = pd.DataFrame(matched_rows)
    sequence_records = [
        build_sequence_record(stay_events, max_len=max_len)
        for _, stay_events in matched_df.groupby("stay_id")
    ]

    sequences = pd.DataFrame(sequence_records)
    sequences = sequences.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    return sequences


def save_event_vocab(sequences: pd.DataFrame, vocab_dir: Path, max_len: int) -> None:
    """Create and save the event vocabulary for stay-level sequences."""
    vocab_dir.mkdir(parents=True, exist_ok=True)
    ConceptTokenizer.create_vocab_from_sequences(
        sequences=sequences[f"event_tokens_{max_len}"],
        save_path=str(vocab_dir / "event_vocab.json"),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meds-data-dir", type=Path, required=True)
    parser.add_argument("--master-stays-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument(
        "--context-hours",
        type=float,
        default=24.0 * 7.0,
        help="How many hours before ICU admission to include in the stay sequence.",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=None,
        help="Optional directory where an event vocab should be written.",
    )
    args = parser.parse_args()

    master_stays = load_master_stays(args.master_stays_path)
    sequences = build_stay_sequences(
        meds_data_dir=args.meds_data_dir,
        master_stays=master_stays,
        max_len=args.max_len,
        context_hours=args.context_hours,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == ".csv":
        sequences.to_csv(args.output_path, index=False)
    else:
        sequences.to_parquet(args.output_path, index=False)

    if args.vocab_dir is not None:
        save_event_vocab(sequences, args.vocab_dir, max_len=args.max_len)

    print(f"Saved {len(sequences):,} stay sequences to {args.output_path}")
    print(f"Unique stays: {sequences['stay_id'].nunique():,}")


if __name__ == "__main__":
    main()
