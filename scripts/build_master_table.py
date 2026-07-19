from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

KEYS = ["id", "study_interval", "day_in_study"]


def load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"Loading {filename}...")
    return pd.read_csv(path, low_memory=False)


def assert_unique_daily_keys(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    duplicate_count = int(
        dataframe.duplicated(subset=KEYS, keep=False).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{table_name} still contains {duplicate_count:,} rows "
            "belonging to duplicate participant-study-day keys."
        )


def build_labels() -> pd.DataFrame:
    labels = load_csv("hormones_and_selfreport.csv")

    required = KEYS + ["phase", "is_weekend"]
    missing = [
        column for column in required
        if column not in labels.columns
    ]

    if missing:
        raise KeyError(
            f"Labels table is missing columns: {missing}"
        )

    labels = labels[required].copy()

    missing_phase_count = int(labels["phase"].isna().sum())

    labels = labels.dropna(
        subset=KEYS + ["phase"]
    ).copy()

    labels["phase"] = labels["phase"].astype(str)

    assert_unique_daily_keys(labels, "labels")

    print(
        f"Labels: removed {missing_phase_count:,} row(s) "
        "with a missing phase."
    )

    return labels


def build_resting_heart_rate() -> pd.DataFrame:
    dataframe = load_csv("resting_heart_rate.csv")

    dataframe = dataframe.drop_duplicates().copy()

    dataframe["value"] = pd.to_numeric(
        dataframe["value"],
        errors="coerce",
    )

    dataframe["error"] = pd.to_numeric(
        dataframe["error"],
        errors="coerce",
    )

    zero_or_negative = int(
        (dataframe["value"] <= 0).sum()
    )

    dataframe = dataframe[
        dataframe["value"] > 0
    ].copy()

    daily = (
        dataframe
        .dropna(subset=KEYS)
        .groupby(KEYS, as_index=False)
        .agg(
            rhr_bpm=("value", "median"),
            rhr_error_median=("error", "median"),
            rhr_estimate_count=("value", "size"),
            rhr_within_day_std=("value", "std"),
        )
    )

    assert_unique_daily_keys(
        daily,
        "resting heart rate",
    )

    print(
        "Resting heart rate: excluded "
        f"{zero_or_negative:,} zero or negative record(s)."
    )

    return daily


def build_temperature() -> pd.DataFrame:
    dataframe = load_csv("computed_temperature.csv")

    dataframe = dataframe.rename(
        columns={
            "sleep_end_day_in_study": "day_in_study"
        }
    )

    required = KEYS + [
        "nightly_temperature",
        "temperature_samples",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]

    missing = [
        column for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Temperature table is missing columns: {missing}"
        )

    dataframe = dataframe.drop_duplicates().copy()

    numeric_columns = [
        "nightly_temperature",
        "temperature_samples",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    daily = (
        dataframe
        .dropna(subset=KEYS)
        .groupby(KEYS, as_index=False)
        .agg(
            skin_temperature_c=(
                "nightly_temperature",
                "median",
            ),
            temperature_samples_max=(
                "temperature_samples",
                "max",
            ),
            temperature_record_count=(
                "nightly_temperature",
                "size",
            ),
            temperature_within_day_std=(
                "nightly_temperature",
                "std",
            ),
            temperature_nightly_sd=(
                "baseline_relative_nightly_standard_deviation",
                "median",
            ),
            temperature_sample_sd=(
                "baseline_relative_sample_standard_deviation",
                "median",
            ),
        )
    )

    assert_unique_daily_keys(daily, "temperature")

    return daily


def parse_boolean(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def build_sleep() -> pd.DataFrame:
    dataframe = load_csv("sleep.csv")

    dataframe = dataframe.rename(
        columns={
            "sleep_end_day_in_study": "day_in_study"
        }
    )

    required = KEYS + [
        "duration",
        "minutesasleep",
        "minutesawake",
        "minutesafterwakeup",
        "timeinbed",
        "minutestofallasleep",
        "efficiency",
        "mainsleep",
        "type",
        "sleep_end_timestamp",
    ]

    missing = [
        column for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Sleep table is missing columns: {missing}"
        )

    dataframe = dataframe.drop_duplicates().copy()

    numeric_columns = [
        "duration",
        "minutesasleep",
        "minutesawake",
        "minutesafterwakeup",
        "timeinbed",
        "minutestofallasleep",
        "efficiency",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["_is_main_sleep"] = parse_boolean(
        dataframe["mainsleep"]
    )

    dataframe["_is_stages"] = (
        dataframe["type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("stages")
    )

    # Preference order:
    # 1. Main sleep
    # 2. Staged sleep record
    # 3. Longest duration
    # 4. Latest ending record
    dataframe = dataframe.sort_values(
        by=KEYS + [
            "_is_main_sleep",
            "_is_stages",
            "duration",
            "sleep_end_timestamp",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    selected = dataframe.drop_duplicates(
        subset=KEYS,
        keep="first",
    ).copy()

    selected = selected.rename(
        columns={
            "minutesasleep": "sleep_minutes",
            "minutesawake": "sleep_awake_minutes",
            "minutesafterwakeup": (
                "sleep_after_wakeup_minutes"
            ),
            "timeinbed": "sleep_time_in_bed_minutes",
            "minutestofallasleep": (
                "sleep_onset_minutes"
            ),
            "efficiency": "sleep_efficiency",
            "_is_main_sleep": "sleep_selected_main",
            "_is_stages": "sleep_selected_stages",
        }
    )

    output_columns = KEYS + [
        "sleep_minutes",
        "sleep_awake_minutes",
        "sleep_after_wakeup_minutes",
        "sleep_time_in_bed_minutes",
        "sleep_onset_minutes",
        "sleep_efficiency",
        "sleep_selected_main",
        "sleep_selected_stages",
    ]

    selected = selected[output_columns]

    assert_unique_daily_keys(selected, "sleep")

    print(
        "Sleep: selected one best sleep record for each "
        f"of {len(selected):,} participant-study-days."
    )

    return selected


def build_activity() -> pd.DataFrame:
    dataframe = load_csv("active_minutes.csv")

    dataframe = dataframe.drop_duplicates().copy()

    numeric_columns = [
        "sedentary",
        "lightly",
        "moderately",
        "very",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    daily = (
        dataframe
        .dropna(subset=KEYS)
        .groupby(KEYS, as_index=False)
        .agg(
            sedentary_minutes=("sedentary", "median"),
            light_activity_minutes=("lightly", "median"),
            moderate_activity_minutes=(
                "moderately",
                "median",
            ),
            vigorous_activity_minutes=("very", "median"),
            activity_record_count=("lightly", "size"),
        )
    )

    daily["total_active_minutes"] = (
        daily["light_activity_minutes"].fillna(0)
        + daily["moderate_activity_minutes"].fillna(0)
        + daily["vigorous_activity_minutes"].fillna(0)
    )

    assert_unique_daily_keys(daily, "activity")

    return daily


def add_modality_flags(
    master: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()

    master["has_rhr"] = master["rhr_bpm"].notna()

    master["has_temperature"] = (
        master["skin_temperature_c"].notna()
    )

    master["has_sleep"] = (
        master["sleep_minutes"].notna()
    )

    master["has_activity"] = (
        master["light_activity_minutes"].notna()
        | master["moderate_activity_minutes"].notna()
        | master["vigorous_activity_minutes"].notna()
    )

    modality_columns = [
        "has_rhr",
        "has_temperature",
        "has_sleep",
        "has_activity",
    ]

    master["available_modality_count"] = (
        master[modality_columns]
        .sum(axis=1)
        .astype(int)
    )

    def make_pattern(row: pd.Series) -> str:
        names = []

        if row["has_rhr"]:
            names.append("RHR")

        if row["has_temperature"]:
            names.append("Temperature")

        if row["has_sleep"]:
            names.append("Sleep")

        if row["has_activity"]:
            names.append("Activity")

        return "+".join(names) if names else "None"

    master["modality_pattern"] = master.apply(
        make_pattern,
        axis=1,
    )

    return master


def make_json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def create_report(
    master: pd.DataFrame,
) -> dict[str, Any]:
    feature_columns = [
        "rhr_bpm",
        "skin_temperature_c",
        "sleep_minutes",
        "sleep_efficiency",
        "light_activity_minutes",
        "moderate_activity_minutes",
        "vigorous_activity_minutes",
        "total_active_minutes",
    ]

    phase_counts = (
        master["phase"]
        .value_counts()
        .to_dict()
    )

    missingness = (
        master[feature_columns]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    modality_patterns = (
        master["modality_pattern"]
        .value_counts()
        .head(15)
        .to_dict()
    )

    study_intervals = (
        master["study_interval"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report = {
        "rows": int(len(master)),
        "participants": int(master["id"].nunique()),
        "duplicate_daily_keys": int(
            master.duplicated(
                subset=KEYS,
                keep=False,
            ).sum()
        ),
        "phase_counts": {
            str(key): int(value)
            for key, value in phase_counts.items()
        },
        "study_interval_counts": {
            str(key): int(value)
            for key, value in study_intervals.items()
        },
        "feature_missingness_percent": {
            str(key): float(value)
            for key, value in missingness.items()
        },
        "top_modality_patterns": {
            str(key): int(value)
            for key, value in modality_patterns.items()
        },
        "days_with_all_four_modalities": int(
            (master["available_modality_count"] == 4).sum()
        ),
        "days_with_at_least_three_modalities": int(
            (master["available_modality_count"] >= 3).sum()
        ),
    }

    return report


def main() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE MASTER TABLE BUILDER")
    print("=" * 76)

    labels = build_labels()
    resting_heart_rate = build_resting_heart_rate()
    temperature = build_temperature()
    sleep = build_sleep()
    activity = build_activity()

    master = labels.copy()

    tables = [
        ("resting_heart_rate", resting_heart_rate),
        ("temperature", temperature),
        ("sleep", sleep),
        ("activity", activity),
    ]

    for table_name, table in tables:
        print(f"Merging {table_name}...")

        master = master.merge(
            table,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    master = add_modality_flags(master)

    master = master.sort_values(
        KEYS
    ).reset_index(drop=True)

    assert_unique_daily_keys(master, "master table")

    parquet_path = (
        PROCESSED_DIR
        / "mosaic_phase_master_daily.parquet"
    )

    csv_path = (
        PROCESSED_DIR
        / "mosaic_phase_master_daily.csv"
    )

    master.to_parquet(
        parquet_path,
        index=False,
    )

    master.to_csv(
        csv_path,
        index=False,
    )

    report = create_report(master)

    report_path = (
        REPORTS_DIR
        / "checkpoint3_master_table_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=make_json_safe,
        )

    print("\n" + "=" * 76)
    print("MASTER TABLE COMPLETE")
    print("=" * 76)

    print(f"Rows: {report['rows']:,}")
    print(f"Participants: {report['participants']:,}")
    print(
        "Duplicate participant-study-day keys: "
        f"{report['duplicate_daily_keys']:,}"
    )

    print("\nPhase counts:")
    for phase, count in report["phase_counts"].items():
        print(f"  {phase:12s}: {count:,}")

    print("\nFeature missingness:")
    for feature, percentage in (
        report["feature_missingness_percent"].items()
    ):
        print(
            f"  {feature:32s}: "
            f"{percentage:6.2f}%"
        )

    print("\nMost common modality patterns:")
    for pattern, count in (
        report["top_modality_patterns"].items()
    ):
        print(f"  {pattern:42s}: {count:,}")

    print(
        "\nDays with all four passive modalities: "
        f"{report['days_with_all_four_modalities']:,}"
    )

    print(
        "Days with at least three passive modalities: "
        f"{report['days_with_at_least_three_modalities']:,}"
    )

    print(f"\nParquet file:\n{parquet_path}")
    print(f"\nCSV file:\n{csv_path}")
    print(f"\nSafe aggregate report:\n{report_path}")

    print(
        "\nProcessed files remain restricted research data. "
        "Do not commit them to a public repository."
    )


if __name__ == "__main__":
    main()