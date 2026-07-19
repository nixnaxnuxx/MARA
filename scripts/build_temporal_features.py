from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

INPUT_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_master_daily_with_split.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_temporal_features.parquet"
)

FEATURE_MANIFEST_PATH = (
    ARTIFACTS_DIR
    / "temporal_feature_manifest.json"
)

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint4b_temporal_features_report.json"
)

KEY_COLUMNS = [
    "id",
    "study_interval",
    "day_in_study",
]

TARGET_COLUMNS = [
    "phase",
    "split",
]

GROUP_COLUMNS = [
    "id",
    "study_interval",
]

MODALITY_SIGNALS = {
    "rhr": [
        "rhr_bpm",
    ],
    "temperature": [
        "skin_temperature_c",
    ],
    "sleep": [
        "sleep_minutes",
        "sleep_efficiency",
        "sleep_awake_minutes",
        "sleep_onset_minutes",
    ],
    "activity": [
        "total_active_minutes",
        "moderate_activity_minutes",
        "vigorous_activity_minutes",
    ],
}

QUALITY_FEATURES = {
    "rhr": [
        "rhr_error_median",
        "rhr_estimate_count",
        "rhr_within_day_std",
        "has_rhr",
    ],
    "temperature": [
        "temperature_samples_max",
        "temperature_record_count",
        "temperature_within_day_std",
        "temperature_nightly_sd",
        "temperature_sample_sd",
        "has_temperature",
    ],
    "sleep": [
        "sleep_selected_main",
        "sleep_selected_stages",
        "has_sleep",
    ],
    "activity": [
        "activity_record_count",
        "has_activity",
    ],
    "context": [
        "is_weekend",
        "available_modality_count",
    ],
}


def json_safe(value: Any) -> Any:
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


def load_master() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Split master table was not found: {INPUT_PATH}"
        )

    dataframe = pd.read_parquet(INPUT_PATH)

    required_columns = (
        KEY_COLUMNS
        + TARGET_COLUMNS
        + [
            signal
            for signals in MODALITY_SIGNALS.values()
            for signal in signals
        ]
        + [
            feature
            for features in QUALITY_FEATURES.values()
            for feature in features
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "The split master table is missing columns: "
            f"{missing_columns}"
        )

    dataframe = dataframe.sort_values(
        KEY_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_count = int(
        dataframe.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Input contains {duplicate_count:,} rows "
            "belonging to duplicate daily keys."
        )

    return dataframe


def convert_boolean_context(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    boolean_columns = [
        "has_rhr",
        "has_temperature",
        "has_sleep",
        "has_activity",
        "sleep_selected_main",
        "sleep_selected_stages",
    ]

    for column in boolean_columns:
        dataframe[column] = (
            dataframe[column]
            .fillna(False)
            .astype(bool)
            .astype(int)
        )

    weekend_values = (
        dataframe["is_weekend"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["is_weekend"] = (
        weekend_values
        .isin(["true", "1", "yes"])
        .astype(int)
    )

    return dataframe


def add_temporal_features(
    dataframe: pd.DataFrame,
    signal: str,
) -> tuple[pd.DataFrame, list[str]]:
    dataframe = dataframe.copy()

    generated_columns: list[str] = []

    grouped = dataframe.groupby(
        GROUP_COLUMNS,
        sort=False,
        group_keys=False,
    )[signal]

    feature_name = f"{signal}__missing"
    dataframe[feature_name] = (
        dataframe[signal]
        .isna()
        .astype(int)
    )
    generated_columns.append(feature_name)

    feature_name = f"{signal}__lag1"
    dataframe[feature_name] = grouped.shift(1)
    generated_columns.append(feature_name)

    feature_name = f"{signal}__past_mean_3"
    dataframe[feature_name] = grouped.transform(
        lambda series: (
            series
            .shift(1)
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
        )
    )
    generated_columns.append(feature_name)

    feature_name = f"{signal}__past_mean_7"
    dataframe[feature_name] = grouped.transform(
        lambda series: (
            series
            .shift(1)
            .rolling(
                window=7,
                min_periods=1,
            )
            .mean()
        )
    )
    generated_columns.append(feature_name)

    feature_name = f"{signal}__past_std_7"
    dataframe[feature_name] = grouped.transform(
        lambda series: (
            series
            .shift(1)
            .rolling(
                window=7,
                min_periods=2,
            )
            .std()
        )
    )
    generated_columns.append(feature_name)

    feature_name = f"{signal}__past_count_7"
    dataframe[feature_name] = grouped.transform(
        lambda series: (
            series
            .shift(1)
            .notna()
            .astype(int)
            .rolling(
                window=7,
                min_periods=1,
            )
            .sum()
        )
    )
    generated_columns.append(feature_name)

    feature_name = f"{signal}__prior_median"
    dataframe[feature_name] = grouped.transform(
        lambda series: (
            series
            .shift(1)
            .expanding(
                min_periods=3,
            )
            .median()
        )
    )
    generated_columns.append(feature_name)

    dataframe[
        f"{signal}__delta_from_lag1"
    ] = (
        dataframe[signal]
        - dataframe[f"{signal}__lag1"]
    )
    generated_columns.append(
        f"{signal}__delta_from_lag1"
    )

    dataframe[
        f"{signal}__delta_from_past_mean_3"
    ] = (
        dataframe[signal]
        - dataframe[f"{signal}__past_mean_3"]
    )
    generated_columns.append(
        f"{signal}__delta_from_past_mean_3"
    )

    dataframe[
        f"{signal}__delta_from_past_mean_7"
    ] = (
        dataframe[signal]
        - dataframe[f"{signal}__past_mean_7"]
    )
    generated_columns.append(
        f"{signal}__delta_from_past_mean_7"
    )

    dataframe[
        f"{signal}__delta_from_prior_median"
    ] = (
        dataframe[signal]
        - dataframe[f"{signal}__prior_median"]
    )
    generated_columns.append(
        f"{signal}__delta_from_prior_median"
    )

    return dataframe, generated_columns


def add_gap_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    dataframe = dataframe.copy()

    dataframe["previous_day_in_study"] = (
        dataframe
        .groupby(
            GROUP_COLUMNS,
            sort=False,
        )["day_in_study"]
        .shift(1)
    )

    dataframe["days_since_previous_record"] = (
        dataframe["day_in_study"]
        - dataframe["previous_day_in_study"]
    )

    dataframe["is_first_record_in_interval"] = (
        dataframe["previous_day_in_study"]
        .isna()
        .astype(int)
    )

    dataframe["has_calendar_gap"] = (
        dataframe["days_since_previous_record"]
        .fillna(1)
        .gt(1)
        .astype(int)
    )

    dataframe = dataframe.drop(
        columns=["previous_day_in_study"]
    )

    return dataframe, [
        "days_since_previous_record",
        "is_first_record_in_interval",
        "has_calendar_gap",
    ]


def build_feature_manifest(
    generated_by_signal: dict[str, list[str]],
    gap_features: list[str],
) -> dict[str, Any]:
    modality_features: dict[str, list[str]] = {}

    for modality, signals in MODALITY_SIGNALS.items():
        features: list[str] = []

        for signal in signals:
            features.append(signal)
            features.extend(
                generated_by_signal[signal]
            )

        features.extend(
            QUALITY_FEATURES.get(
                modality,
                [],
            )
        )

        modality_features[modality] = (
            sorted(set(features))
        )

    context_features = (
        QUALITY_FEATURES["context"]
        + gap_features
    )

    all_model_features = sorted(
        set(
            feature
            for features in modality_features.values()
            for feature in features
        )
        | set(context_features)
    )

    return {
        "task_definition": (
            "Current-day phase estimation using current "
            "passive measurements and past-only temporal context."
        ),
        "target": "phase",
        "grouping_unit": "participant id",
        "temporal_grouping": [
            "id",
            "study_interval",
        ],
        "forbidden_model_inputs": [
            "phase",
            "lh",
            "estrogen",
            "pdg",
            "flow_volume",
            "flow_color",
            "participant id",
            "study interval",
            "day in study",
            "split",
        ],
        "modality_features": modality_features,
        "context_features": context_features,
        "all_model_features": all_model_features,
        "feature_count": len(all_model_features),
    }


def create_report(
    dataframe: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    feature_columns = (
        manifest["all_model_features"]
    )

    missingness = (
        dataframe[feature_columns]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    rows_with_any_model_feature = (
        dataframe[feature_columns]
        .notna()
        .any(axis=1)
    )

    rows_with_all_model_features = (
        dataframe[feature_columns]
        .notna()
        .all(axis=1)
    )

    report = {
        "rows": int(len(dataframe)),
        "participants": int(
            dataframe["id"].nunique()
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "split_counts": {
            str(key): int(value)
            for key, value in (
                dataframe["split"]
                .value_counts()
                .items()
            )
        },
        "phase_counts": {
            str(key): int(value)
            for key, value in (
                dataframe["phase"]
                .value_counts()
                .items()
            )
        },
        "rows_with_any_model_feature": int(
            rows_with_any_model_feature.sum()
        ),
        "rows_with_all_model_features": int(
            rows_with_all_model_features.sum()
        ),
        "rows_with_all_model_features_percent": float(
            rows_with_all_model_features.mean()
            * 100
        ),
        "highest_missingness_features": {
            str(key): float(value)
            for key, value in (
                missingness.head(20).items()
            )
        },
        "lowest_missingness_features": {
            str(key): float(value)
            for key, value in (
                missingness.tail(20).items()
            )
        },
        "calendar_gap_rows": int(
            dataframe["has_calendar_gap"].sum()
        ),
        "first_interval_rows": int(
            dataframe[
                "is_first_record_in_interval"
            ].sum()
        ),
    }

    return report


def main() -> None:
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE TEMPORAL FEATURE BUILDER")
    print("=" * 76)

    dataframe = load_master()
    dataframe = convert_boolean_context(
        dataframe
    )

    generated_by_signal: dict[
        str,
        list[str],
    ] = {}

    for modality, signals in (
        MODALITY_SIGNALS.items()
    ):
        print(
            f"\nGenerating {modality} features..."
        )

        for signal in signals:
            print(f"  {signal}")

            dataframe, generated = (
                add_temporal_features(
                    dataframe,
                    signal,
                )
            )

            generated_by_signal[
                signal
            ] = generated

    dataframe, gap_features = (
        add_gap_features(dataframe)
    )

    manifest = build_feature_manifest(
        generated_by_signal,
        gap_features,
    )

    output_columns = (
        KEY_COLUMNS
        + TARGET_COLUMNS
        + manifest["all_model_features"]
    )

    output_columns = list(
        dict.fromkeys(output_columns)
    )

    feature_table = dataframe[
        output_columns
    ].copy()

    duplicate_count = int(
        feature_table.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Temporal feature table contains "
            f"{duplicate_count:,} duplicated daily rows."
        )

    feature_table.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    with FEATURE_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            default=json_safe,
        )

    report = create_report(
        feature_table,
        manifest,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("TEMPORAL FEATURE TABLE COMPLETE")
    print("=" * 76)

    print(f"Rows: {report['rows']:,}")
    print(
        f"Participants: "
        f"{report['participants']:,}"
    )
    print(
        f"Model features: "
        f"{report['feature_count']:,}"
    )

    print("\nSplit counts:")

    for split_name, count in (
        report["split_counts"].items()
    ):
        print(
            f"  {split_name:12s}: "
            f"{count:,}"
        )

    print("\nPhase counts:")

    for phase, count in (
        report["phase_counts"].items()
    ):
        print(
            f"  {phase:12s}: "
            f"{count:,}"
        )

    print(
        "\nRows containing at least one model feature: "
        f"{report['rows_with_any_model_feature']:,}"
    )

    print(
        "Rows containing every model feature: "
        f"{report['rows_with_all_model_features']:,} "
        f"("
        f"{report['rows_with_all_model_features_percent']:.2f}%"
        f")"
    )

    print(
        "\nRows following a calendar gap: "
        f"{report['calendar_gap_rows']:,}"
    )

    print(
        "First rows of participant-study intervals: "
        f"{report['first_interval_rows']:,}"
    )

    print("\nFeatures with highest missingness:")

    for feature, percentage in (
        report[
            "highest_missingness_features"
        ].items()
    ):
        print(
            f"  {feature:48s}: "
            f"{percentage:6.2f}%"
        )

    print(f"\nFeature table:\n{OUTPUT_PATH}")
    print(
        f"\nFeature manifest:\n"
        f"{FEATURE_MANIFEST_PATH}"
    )
    print(f"\nSafe report:\n{REPORT_PATH}")

    print(
        "\nThe feature table remains restricted "
        "research data and must not be published."
    )


if __name__ == "__main__":
    main()