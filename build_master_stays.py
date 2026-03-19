"""Build a stay-level master index for multimodal ICU modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_patients(patients_path: Path) -> pd.DataFrame:
    """Read the patients table from parquet or csv.gz."""
    if patients_path.suffix == ".parquet":
        patients = pd.read_parquet(patients_path)
    else:
        patients = pd.read_csv(
            patients_path,
            compression="gzip",
            usecols=["subject_id", "dod"],
        )

    patients.columns = patients.columns.str.lower()
    patients = patients[["subject_id", "dod"]].copy()
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    return patients.drop_duplicates(subset=["subject_id"])


def build_master_stays(
    icustays_path: Path,
    admissions_path: Path,
    patients_path: Path,
) -> pd.DataFrame:
    """Construct a stay-level table with labels and timing metadata."""
    icu = pd.read_csv(
        icustays_path,
        compression="gzip",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime"]).copy()

    admissions = pd.read_csv(
        admissions_path,
        compression="gzip",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")
    admissions["hospital_expire_flag"] = (
        admissions["hospital_expire_flag"].fillna(0).astype(int)
    )

    patients = read_patients(patients_path)

    master = icu.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    master = master.merge(patients, on="subject_id", how="left")

    master["death_time"] = master["deathtime"]
    missing_death_time = master["death_time"].isna()
    master.loc[missing_death_time, "death_time"] = master.loc[missing_death_time, "dod"]

    master["in_hosp_mortality"] = master["hospital_expire_flag"].astype(int)
    master["mortality_28d"] = (
        master["death_time"].notna()
        & (master["death_time"] >= master["intime"])
        & (master["death_time"] <= master["intime"] + pd.Timedelta(days=28))
    ).astype(int)

    master = master.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    return master[
        [
            "stay_id",
            "subject_id",
            "hadm_id",
            "intime",
            "outtime",
            "admittime",
            "dischtime",
            "deathtime",
            "dod",
            "death_time",
            "in_hosp_mortality",
            "mortality_28d",
            "hospital_expire_flag",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ]
    ].copy()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icustays-path", type=Path, required=True)
    parser.add_argument("--admissions-path", type=Path, required=True)
    parser.add_argument("--patients-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    master = build_master_stays(
        icustays_path=args.icustays_path,
        admissions_path=args.admissions_path,
        patients_path=args.patients_path,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == ".csv":
        master.to_csv(args.output_path, index=False)
    else:
        master.to_parquet(args.output_path, index=False)

    print(f"Saved {len(master):,} ICU stays to {args.output_path}")
    print(
        "Positive labels:",
        {
            "in_hosp_mortality": int(master["in_hosp_mortality"].sum()),
            "mortality_28d": int(master["mortality_28d"].sum()),
        },
    )


if __name__ == "__main__":
    main()
