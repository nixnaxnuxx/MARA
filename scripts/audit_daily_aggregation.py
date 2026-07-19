from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"

JOIN_KEYS = ["id", "study_interval", "day_in_study"]

TABLE_CONFIG = {
    "resting_heart_rate": {
        "filename": "resting_heart_rate.csv",
        "day_column": "day_in_study",
        "numeric_columns": [
            "value",
            "error",
        ],
    },
    "temperature": {
        "filename": "computed_temperature.csv",
        "day_column": "sleep_start_day_in_study",
        "numeric_columns": [
            "temperature_samples",
            "nightly_temperature",
            "baseline_relative_sample_sum",
            "baseline_relative_sample_sum_of_squares",
            "baseline_relative_nightly_standard_deviation",
            "baseline_relative_sample_standard_deviation",
        ],
    },
    "sleep": {
        "filename": "sleep.csv",
        "day_column": "sleep_start_day_in_study",
        "numeric_columns": [
            "duration",
            "minutestofallasleep",
            "minutesasleep",
            "minutesawake",
            "minutesafterwakeup",
            "timeinbed",
            "efficiency",
        ],
    },
    "sleep_score": {
        "filename": "sleep_score.csv",
        "day_column": "day_in_study",
        "numeric_columns": [
            "overall_score",
            "composition_score",
            "revitalization_score",
            "duration_score",
            "deep_sleep_in_minutes",
            "resting_heart_rate",
            "restlessness",
        ],
    },
    "activity": {
        "filename": "active_minutes.csv",
        "day_column": "day_in_study",
        "numeric_columns": [
            "sedentary",
            "lightly",
            "moderately",
            "very",
        ],
    },
    "glucose": {
        "filename": "glucose.csv",
        "day_column": "day_in_study",
        "numeric_columns": [
            "glucose_value",
        ],
    },
}


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def load_table(table_name: str) -> pd.DataFrame:
    config = TABLE_CONFIG[table_name]
    path = RAW_DIR / config["filename"]

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"Loading {config['filename']}...")

    dataframe = pd.read_csv(path, low_memory=False)

    source_day_column = config["day_column"]

    if source_day_column != "day_in_study":
        if source_day_column not in dataframe.columns:
            raise KeyError(
                f"{config['filename']} is missing "
                f"{source_day_column}"
            )

        dataframe = dataframe.rename(
            columns={source_day_column: "day_in_study"}
        )

    missing_keys = [
        column
        for column in JOIN_KEYS
        if column not in dataframe.columns
    ]

    if missing_keys:
        raise KeyError(
            f"{config['filename']} is missing join keys: "
            f"{missing_keys}"
        )

    return dataframe


def describe_group_sizes(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    group_sizes = (
        dataframe
        .dropna(subset=JOIN_KEYS)
        .groupby(JOIN_KEYS, dropna=False)
        .size()
    )

    duplicate_groups = group_sizes[group_sizes > 1]

    if len(duplicate_groups) == 0:
        return {
            "unique_daily_keys": int(len(group_sizes)),
            "duplicate_daily_keys": 0,
            "rows_in_duplicate_keys": 0,
            "median_rows_per_duplicate_key": None,
            "maximum_rows_per_daily_key": 1,
        }

    return {
        "unique_daily_keys": int(len(group_sizes)),
        "duplicate_daily_keys": int(len(duplicate_groups)),
        "rows_in_duplicate_keys": int(duplicate_groups.sum()),
        "median_rows_per_duplicate_key": float(
            duplicate_groups.median()
        ),
        "maximum_rows_per_daily_key": int(group_sizes.max()),
    }


def numeric_column_audit(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    missing_percent = float(numeric.isna().mean() * 100)

    valid = numeric.dropna()

    if valid.empty:
        overall_summary = {
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    else:
        overall_summary = {
            "minimum": float(valid.min()),
            "median": float(valid.median()),
            "maximum": float(valid.max()),
        }

    temporary = dataframe[JOIN_KEYS].copy()
    temporary["numeric_value"] = numeric

    group_sizes = (
        temporary
        .dropna(subset=JOIN_KEYS)
        .groupby(JOIN_KEYS, dropna=False)
        .size()
    )

    duplicate_group_keys = group_sizes[group_sizes > 1].index

    if len(duplicate_group_keys) == 0:
        varying_duplicate_groups = 0
        duplicate_groups_with_values = 0
        varying_percent = 0.0
    else:
        distinct_values = (
            temporary
            .groupby(JOIN_KEYS, dropna=False)["numeric_value"]
            .nunique(dropna=True)
            .reindex(duplicate_group_keys)
        )

        groups_with_values = distinct_values[
            distinct_values > 0
        ]

        duplicate_groups_with_values = int(
            len(groups_with_values)
        )

        varying_duplicate_groups = int(
            (groups_with_values > 1).sum()
        )

        varying_percent = (
            100.0
            * varying_duplicate_groups
            / duplicate_groups_with_values
            if duplicate_groups_with_values
            else 0.0
        )

    return {
        "missing_percent": missing_percent,
        "minimum": overall_summary["minimum"],
        "median": overall_summary["median"],
        "maximum": overall_summary["maximum"],
        "duplicate_groups_with_values": (
            duplicate_groups_with_values
        ),
        "duplicate_groups_with_varying_values": (
            varying_duplicate_groups
        ),
        "varying_duplicate_group_percent": (
            varying_percent
        ),
    }


def audit_sleep_sessions(
    sleep: pd.DataFrame,
) -> dict[str, Any]:
    if "mainsleep" not in sleep.columns:
        return {
            "error": "mainsleep column was not found"
        }

    valid_keys = (
        sleep
        .dropna(subset=JOIN_KEYS)
        [JOIN_KEYS]
        .drop_duplicates()
    )

    main_sleep_mask = (
        sleep["mainsleep"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    main_sleep_counts = (
        sleep.loc[main_sleep_mask]
        .groupby(JOIN_KEYS, dropna=False)
        .size()
        .rename("main_sleep_sessions")
        .reset_index()
    )

    daily_counts = valid_keys.merge(
        main_sleep_counts,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    daily_counts["main_sleep_sessions"] = (
        daily_counts["main_sleep_sessions"]
        .fillna(0)
        .astype(int)
    )

    zero_main = int(
        (daily_counts["main_sleep_sessions"] == 0).sum()
    )

    one_main = int(
        (daily_counts["main_sleep_sessions"] == 1).sum()
    )

    multiple_main = int(
        (daily_counts["main_sleep_sessions"] > 1).sum()
    )

    return {
        "daily_keys": int(len(daily_counts)),
        "days_with_no_main_sleep": zero_main,
        "days_with_one_main_sleep": one_main,
        "days_with_multiple_main_sleep": multiple_main,
        "maximum_main_sleep_sessions_in_one_day": int(
            daily_counts["main_sleep_sessions"].max()
        ),
    }


def audit_table(
    table_name: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print(table_name.upper())
    print("=" * 76)

    exact_duplicate_rows = int(
        dataframe.duplicated(keep=False).sum()
    )

    group_summary = describe_group_sizes(dataframe)

    print(f"Rows: {len(dataframe):,}")
    print(
        "Unique participant-study-days: "
        f"{group_summary['unique_daily_keys']:,}"
    )
    print(
        "Daily keys containing more than one row: "
        f"{group_summary['duplicate_daily_keys']:,}"
    )
    print(
        "Rows belonging to duplicate daily keys: "
        f"{group_summary['rows_in_duplicate_keys']:,}"
    )
    print(
        "Median rows per duplicated daily key: "
        f"{group_summary['median_rows_per_duplicate_key']}"
    )
    print(
        "Maximum rows in one daily key: "
        f"{group_summary['maximum_rows_per_daily_key']:,}"
    )
    print(
        "Rows belonging to exact duplicate records: "
        f"{exact_duplicate_rows:,}"
    )

    numeric_results: dict[str, Any] = {}

    configured_columns = TABLE_CONFIG[
        table_name
    ]["numeric_columns"]

    print("\nNumeric feature audit:")

    for column in configured_columns:
        if column not in dataframe.columns:
            print(f"\n{column}: COLUMN NOT FOUND")
            continue

        result = numeric_column_audit(
            dataframe,
            column,
        )

        numeric_results[column] = result

        print(f"\n{column}")
        print(
            f"  Missing: "
            f"{result['missing_percent']:.2f}%"
        )
        print(
            "  Overall range: "
            f"{result['minimum']} to "
            f"{result['maximum']}"
        )
        print(
            f"  Overall median: "
            f"{result['median']}"
        )
        print(
            "  Duplicate groups with usable values: "
            f"{result['duplicate_groups_with_values']:,}"
        )
        print(
            "  Duplicate groups with differing values: "
            f"{result['duplicate_groups_with_varying_values']:,}"
        )
        print(
            "  Varying duplicate-group percentage: "
            f"{result['varying_duplicate_group_percent']:.2f}%"
        )

    result: dict[str, Any] = {
        "rows": int(len(dataframe)),
        "exact_duplicate_rows": exact_duplicate_rows,
        "group_summary": group_summary,
        "numeric_columns": numeric_results,
    }

    if table_name == "sleep":
        sleep_session_results = audit_sleep_sessions(
            dataframe
        )

        result["sleep_session_audit"] = (
            sleep_session_results
        )

        print("\nMain-sleep session audit:")
        for key, value in sleep_session_results.items():
            print(f"  {key}: {value}")

    return result


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE DAILY AGGREGATION AUDIT")
    print("=" * 76)

    results: dict[str, Any] = {}

    for table_name in TABLE_CONFIG:
        dataframe = load_table(table_name)

        results[table_name] = audit_table(
            table_name,
            dataframe,
        )

        del dataframe

    report_path = (
        REPORTS_DIR
        / "checkpoint2_5_aggregation_audit.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            default=make_json_safe,
        )

    print("\n" + "=" * 76)
    print("AGGREGATION AUDIT COMPLETE")
    print("=" * 76)
    print(f"Safe aggregate report saved to:\n{report_path}")
    print(
        "\nNo participant-level rows were printed "
        "or exported."
    )


if __name__ == "__main__":
    main()