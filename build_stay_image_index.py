"""Map chest X-ray studies to ICU stays."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from odyssey.data.stay_level import normalize_bool_series


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


def load_cxr_metadata(cxr_csv_path: Path) -> pd.DataFrame:
    """Load and clean the study-level CXR metadata CSV."""
    studies = pd.read_csv(cxr_csv_path)

    unnamed_columns = [column for column in studies.columns if column.startswith("Unnamed") or column == ""]
    if unnamed_columns:
        studies = studies.drop(columns=unnamed_columns)

    studies["subject_id"] = pd.to_numeric(studies["subject_id"], errors="coerce").astype("Int64")
    if "hadm_id" in studies.columns:
        studies["hadm_id"] = pd.to_numeric(studies["hadm_id"], errors="coerce").astype("Int64")
    studies["study_id"] = pd.to_numeric(studies["study_id"], errors="coerce").astype("Int64")
    studies["StudyDateTime"] = pd.to_datetime(studies["StudyDateTime"], errors="coerce")

    if "jpg_path_exists" in studies.columns:
        studies["jpg_path_exists"] = normalize_bool_series(studies["jpg_path_exists"])
    else:
        studies["jpg_path_exists"] = studies["jpg_path"].notna()

    if "study_in_icu_csn" in studies.columns:
        studies["study_in_icu_csn"] = normalize_bool_series(studies["study_in_icu_csn"])

    return studies


def build_stay_image_index(
    master_stays: pd.DataFrame,
    studies: pd.DataFrame,
    keep_all_images: bool,
    first_hours_only: float | None,
) -> pd.DataFrame:
    """Attach each CXR study to the ICU stay it belongs to."""
    studies = studies.dropna(subset=["subject_id", "study_id", "StudyDateTime"]).copy()
    studies = studies[studies["jpg_path_exists"]].copy()

    merge_columns = ["subject_id"]
    if "hadm_id" in studies.columns and "hadm_id" in master_stays.columns:
        merge_columns.append("hadm_id")

    merged = studies.merge(
        master_stays[
            [
                "stay_id",
                "subject_id",
                "hadm_id",
                "intime",
                "outtime",
                "in_hosp_mortality",
                "mortality_28d",
            ]
        ],
        on=merge_columns,
        how="inner",
    )

    merged = merged[
        (merged["StudyDateTime"] >= merged["intime"])
        & (
            merged["outtime"].isna()
            | (merged["StudyDateTime"] <= merged["outtime"])
        )
    ].copy()
    if merged.empty:
        return merged

    merged["hours_from_icu_intime"] = (
        (merged["StudyDateTime"] - merged["intime"]).dt.total_seconds() / 3600.0
    )

    if first_hours_only is not None:
        merged = merged[merged["hours_from_icu_intime"] <= first_hours_only].copy()

    merged = merged.sort_values(["stay_id", "StudyDateTime", "study_id"]).reset_index(drop=True)
    merged["image_rank_in_stay"] = merged.groupby("stay_id").cumcount()
    merged["is_first_image"] = merged["image_rank_in_stay"].eq(0)
    merged["is_first_24h_image"] = merged["hours_from_icu_intime"].le(24.0)

    if not keep_all_images:
        merged = merged[merged["is_first_image"]].copy()

    return merged[
        [
            "stay_id",
            "subject_id",
            "hadm_id",
            "study_id",
            "dicom_id",
            "StudyDateTime",
            "hours_from_icu_intime",
            "image_rank_in_stay",
            "is_first_image",
            "is_first_24h_image",
            "jpg_path",
            "jpg_path_exists",
            "path",
            "delta_t",
            "delta_t_days",
            "study_in_icu_csn",
            "full_report",
            "findings",
            "impressions",
            "header",
            "in_hosp_mortality",
            "mortality_28d",
        ]
    ].copy()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-stays-path", type=Path, required=True)
    parser.add_argument("--cxr-csv-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--first-hours-only",
        type=float,
        default=None,
        help="Only keep images captured within this many hours after ICU admission.",
    )
    parser.add_argument(
        "--keep-all-images",
        action="store_true",
        help="Keep every matched study instead of only the first image per stay.",
    )
    args = parser.parse_args()

    master_stays = load_master_stays(args.master_stays_path)
    studies = load_cxr_metadata(args.cxr_csv_path)
    stay_image_index = build_stay_image_index(
        master_stays=master_stays,
        studies=studies,
        keep_all_images=args.keep_all_images,
        first_hours_only=args.first_hours_only,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == ".csv":
        stay_image_index.to_csv(args.output_path, index=False)
    else:
        stay_image_index.to_parquet(args.output_path, index=False)

    print(f"Saved {len(stay_image_index):,} stay-image rows to {args.output_path}")
    print(f"Matched stays: {stay_image_index['stay_id'].nunique():,}")


if __name__ == "__main__":
    main()
