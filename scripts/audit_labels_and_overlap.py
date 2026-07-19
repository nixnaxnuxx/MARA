from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"

JOIN_KEYS = ["id", "study_interval", "day_in_study"]

TABLE_CONFIG = {
    "labels": {
        "filename": "hormones_and_selfreport.csv",
        "day_column": "day_in_study",
    },
    "resting_heart_rate": {
        "filename": "resting_heart_rate.csv",
        "day_column": "day_in_study",
    },
    "temperature": {
        "filename": "computed_temperature.csv",
        "day_column": "sleep_start_day_in_study",
    },
    "sleep": {
        "filename": "sleep.csv",
        "day_column": "sleep_start_day_in_study",
    },
    "sleep_score": {
        "filename": "sleep_score.csv",
        "day_column": "day_in_study",
    },
    "activity": {
        "filename": "active_minutes.csv",
        "day_column": "day_in_study",
    },
    "glucose": {
        "filename": "glucose.csv",
        "day_column": "day_in_study",
    },
}


def to_builtin(value: Any) -> Any:
    """Convert pandas and NumPy scalar values to JSON-compatible types."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def load_table(name: str) -> pd.DataFrame:
    config = TABLE_CONFIG[name]
    path = RAW_DIR / config["filename"]

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    print(f"Loading {config['filename']}...")
    df = pd.read_csv(path, low_memory=False)

    source_day_column = config["day_column"]

    if source_day_column != "day_in_study":
        if source_day_column not in df.columns:
            raise KeyError(
                f"{config['filename']} does not contain "
                f"the expected day column: {source_day_column}"
            )

        df = df.rename(columns={source_day_column: "day_in_study"})

    missing_join_keys = [
        column for column in JOIN_KEYS if column not in df.columns
    ]

    if missing_join_keys:
        raise KeyError(
            f"{config['filename']} is missing join columns: "
            f"{missing_join_keys}"
        )

    return df


def unique_day_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique non-missing participant-study-day keys."""
    return (
        df.dropna(subset=JOIN_KEYS)
        [JOIN_KEYS]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    keys = unique_day_keys(df)

    duplicate_rows = int(
        df.duplicated(subset=JOIN_KEYS, keep=False).sum()
    )

    return {
        "rows": int(len(df)),
        "participants": int(df["id"].nunique(dropna=True)),
        "study_intervals": int(
            df["study_interval"].nunique(dropna=True)
        ),
        "unique_participant_days": int(len(keys)),
        "rows_in_duplicate_day_keys": duplicate_rows,
    }


def print_dataset_summary(
    name: str,
    summary: dict[str, Any],
) -> None:
    print("\n" + "-" * 72)
    print(name.upper())
    print("-" * 72)
    print(f"Rows: {summary['rows']:,}")
    print(f"Participants: {summary['participants']:,}")
    print(f"Study intervals: {summary['study_intervals']:,}")
    print(
        "Unique participant-study-days: "
        f"{summary['unique_participant_days']:,}"
    )
    print(
        "Rows belonging to duplicate daily keys: "
        f"{summary['rows_in_duplicate_day_keys']:,}"
    )


def analyse_labels(labels: pd.DataFrame) -> dict[str, Any]:
    if "phase" not in labels.columns:
        raise KeyError(
            "The labels file does not contain a 'phase' column."
        )

    print("\n" + "=" * 72)
    print("PHASE LABEL AUDIT")
    print("=" * 72)

    phase_counts_with_missing = (
        labels["phase"]
        .fillna("<MISSING>")
        .astype(str)
        .value_counts(dropna=False)
    )

    print("\nRaw phase value counts:")
    print(phase_counts_with_missing.to_string())

    labelled = labels.dropna(subset=["phase"]).copy()

    # Detect days with more than one distinct non-missing phase.
    phase_conflicts = (
        labelled.groupby(JOIN_KEYS, dropna=False)["phase"]
        .nunique(dropna=True)
    )

    conflicting_day_count = int((phase_conflicts > 1).sum())

    # Remove exact repeated daily phase labels for support calculations.
    labelled_unique = labelled.drop_duplicates(
        subset=JOIN_KEYS + ["phase"]
    )

    phase_support = (
        labelled_unique.groupby("phase")
        .agg(
            labelled_days=("day_in_study", "size"),
            participants=("id", "nunique"),
        )
        .sort_values("labelled_days", ascending=False)
    )

    print("\nSupport by phase:")
    print(phase_support.to_string())

    participant_days = (
        labelled.drop_duplicates(subset=JOIN_KEYS)
        .groupby("id")
        .size()
    )

    participant_phase_counts = (
        labelled.groupby("id")["phase"]
        .nunique(dropna=True)
    )

    phase_coverage_distribution = (
        participant_phase_counts
        .value_counts()
        .sort_index()
    )

    print("\nParticipants by number of distinct phases represented:")
    print(phase_coverage_distribution.to_string())

    print("\nLabelled-day count per participant:")
    print(
        f"Minimum: {int(participant_days.min()):,}\n"
        f"Median: {float(participant_days.median()):.1f}\n"
        f"Maximum: {int(participant_days.max()):,}"
    )

    duplicate_label_rows = int(
        labelled.duplicated(subset=JOIN_KEYS, keep=False).sum()
    )

    print("\nLabel integrity:")
    print(
        "Rows belonging to duplicated labelled-day keys: "
        f"{duplicate_label_rows:,}"
    )
    print(
        "Daily keys containing conflicting phase labels: "
        f"{conflicting_day_count:,}"
    )

    phase_support_json = {
        str(index): {
            "labelled_days": int(row["labelled_days"]),
            "participants": int(row["participants"]),
        }
        for index, row in phase_support.iterrows()
    }

    return {
        "phase_counts_including_missing": {
            str(index): int(value)
            for index, value in phase_counts_with_missing.items()
        },
        "phase_support": phase_support_json,
        "duplicate_label_rows": duplicate_label_rows,
        "conflicting_daily_phase_keys": conflicting_day_count,
        "participants_by_number_of_phases": {
            str(index): int(value)
            for index, value in phase_coverage_distribution.items()
        },
        "labelled_days_per_participant": {
            "minimum": int(participant_days.min()),
            "median": float(participant_days.median()),
            "maximum": int(participant_days.max()),
        },
    }


def print_safe_categorical_audits(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    print("\n" + "=" * 72)
    print("CATEGORICAL FIELD AUDIT")
    print("=" * 72)

    fields = [
        ("temperature", "type"),
        ("sleep", "type"),
        ("sleep", "mainsleep"),
    ]

    for table_name, column in fields:
        df = tables[table_name]

        if column not in df.columns:
            print(f"\n{table_name}.{column}: COLUMN NOT FOUND")
            continue

        counts = (
            df[column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts(dropna=False)
        )

        print(f"\n{table_name}.{column}:")
        print(counts.to_string())

        output[f"{table_name}.{column}"] = {
            str(index): int(value)
            for index, value in counts.items()
        }

    return output


def analyse_sensor_overlap(
    labels: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    labelled_days = (
        labels.dropna(subset=["phase"])
        [JOIN_KEYS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    coverage_frame = labelled_days.copy()
    coverage_results: dict[str, Any] = {}

    print("\n" + "=" * 72)
    print("SENSOR COVERAGE ON LABELLED DAYS")
    print("=" * 72)

    for table_name, df in tables.items():
        if table_name == "labels":
            continue

        sensor_keys = unique_day_keys(df)

        indicator_column = f"has_{table_name}"

        sensor_keys[indicator_column] = True

        coverage_frame = coverage_frame.merge(
            sensor_keys,
            on=JOIN_KEYS,
            how="left",
            validate="one_to_one",
        )

        coverage_frame[indicator_column] = (
            coverage_frame[indicator_column]
            .fillna(False)
            .astype(bool)
        )

        covered_days = int(coverage_frame[indicator_column].sum())
        labelled_day_count = int(len(coverage_frame))

        coverage_percentage = (
            100.0 * covered_days / labelled_day_count
            if labelled_day_count
            else 0.0
        )

        coverage_results[table_name] = {
            "covered_labelled_days": covered_days,
            "total_labelled_days": labelled_day_count,
            "coverage_percent": coverage_percentage,
        }

        print(
            f"{table_name:24s} "
            f"{covered_days:6,d} / {labelled_day_count:6,d} "
            f"({coverage_percentage:6.2f}%)"
        )

    indicator_columns = [
        column
        for column in coverage_frame.columns
        if column.startswith("has_")
    ]

    combination_counts = (
        coverage_frame.groupby(indicator_columns, dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(15)
        .rename("labelled_days")
        .reset_index()
    )

    print("\nMost common sensor-availability combinations:")
    print(combination_counts.to_string(index=False))

    coverage_results["top_sensor_combinations"] = [
        {
            column: to_builtin(row[column])
            for column in combination_counts.columns
        }
        for _, row in combination_counts.iterrows()
    ]

    return coverage_results, coverage_frame


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MOSAIC-PHASE CHECKPOINT 2 AUDIT")
    print("=" * 72)

    tables = {
        name: load_table(name)
        for name in TABLE_CONFIG
    }

    summaries: dict[str, Any] = {}

    print("\n" + "=" * 72)
    print("TABLE SUMMARIES")
    print("=" * 72)

    for name, df in tables.items():
        summary = dataset_summary(df)
        summaries[name] = summary
        print_dataset_summary(name, summary)

    label_results = analyse_labels(tables["labels"])

    categorical_results = print_safe_categorical_audits(tables)

    coverage_results, coverage_frame = analyse_sensor_overlap(
        labels=tables["labels"],
        tables=tables,
    )

    report = {
        "table_summaries": summaries,
        "label_audit": label_results,
        "categorical_audit": categorical_results,
        "sensor_coverage": coverage_results,
        "labelled_day_count": int(len(coverage_frame)),
    }

    report_path = REPORTS_DIR / "checkpoint2_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 72)
    print("CHECKPOINT 2 AUDIT COMPLETE")
    print("=" * 72)
    print(f"Safe aggregate report saved to:\n{report_path}")
    print("\nNo participant-level rows were printed or exported.")


if __name__ == "__main__":
    main()