from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

FEATURE_TABLE_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_temporal_features.parquet"
)

MODELLING_MANIFEST_PATH = (
    ARTIFACTS_DIR
    / "modelling_feature_manifest.json"
)

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint5c_robust_feature_sets.json"
)

RECOMMENDATION_PATH = (
    ARTIFACTS_DIR
    / "recommended_feature_subset.json"
)

RANDOM_SEED = 20260719
MISSINGNESS_THRESHOLD = 0.95
CV_SPLITS = 4

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = {
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

AVAILABILITY_FEATURES = {
    "rhr": "has_rhr",
    "temperature": "has_temperature",
    "sleep": "has_sleep",
    "activity": "has_activity",
}

CONTEXT_FEATURES = [
    "is_weekend",
    "days_since_previous_record",
    "is_first_record_in_interval",
    "has_calendar_gap",
]

ALL_MODALITY_CONTEXT = (
    CONTEXT_FEATURES
    + [
        "available_modality_count",
        "has_rhr",
        "has_temperature",
        "has_sleep",
        "has_activity",
    ]
)

RELATIVE_SUFFIXES = (
    "__missing",
    "__past_std_7",
    "__past_count_7",
    "__delta_from_lag1",
    "__delta_from_past_mean_3",
    "__delta_from_past_mean_7",
    "__delta_from_prior_median",
)


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


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE_PATH}"
        )

    if not MODELLING_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Modelling manifest not found: "
            f"{MODELLING_MANIFEST_PATH}"
        )

    dataframe = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    with MODELLING_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return dataframe, manifest


def unique_preserving_order(
    values: list[str],
) -> list[str]:
    return list(dict.fromkeys(values))


def derived_features(
    available_features: list[str],
    signals: list[str],
) -> list[str]:
    output = []

    for feature in available_features:
        for signal in signals:
            if (
                feature == signal
                or feature.startswith(
                    signal + "__"
                )
            ):
                output.append(feature)
                break

    return output


def relative_features(
    available_features: list[str],
    signals: list[str],
) -> list[str]:
    output = []

    signal_features = derived_features(
        available_features,
        signals,
    )

    for feature in signal_features:
        if feature.endswith(
            RELATIVE_SUFFIXES
        ):
            output.append(feature)

    return output


def build_feature_sets(
    temporal_features: list[str],
) -> dict[str, list[str]]:
    all_signals = [
        signal
        for modality_signals
        in SIGNALS.values()
        for signal in modality_signals
    ]

    physiological_features = (
        derived_features(
            temporal_features,
            all_signals,
        )
    )

    physiological_relative = (
        relative_features(
            temporal_features,
            all_signals,
        )
    )

    current_signals = [
        signal
        for signal in all_signals
        if signal in temporal_features
    ]

    no_activity_signals = (
        SIGNALS["rhr"]
        + SIGNALS["temperature"]
        + SIGNALS["sleep"]
    )

    rhr_temperature_signals = (
        SIGNALS["rhr"]
        + SIGNALS["temperature"]
    )

    feature_sets = {
        "All temporal features": (
            temporal_features
        ),

        "Core temporal physiology": (
            physiological_features
            + ALL_MODALITY_CONTEXT
        ),

        "Current plus relative physiology": (
            current_signals
            + physiological_relative
            + ALL_MODALITY_CONTEXT
        ),

        "Relative-only physiology": (
            physiological_relative
            + ALL_MODALITY_CONTEXT
        ),

        "No-activity physiology": (
            derived_features(
                temporal_features,
                no_activity_signals,
            )
            + CONTEXT_FEATURES
            + [
                "has_rhr",
                "has_temperature",
                "has_sleep",
            ]
        ),

        "RHR and temperature only": (
            derived_features(
                temporal_features,
                rhr_temperature_signals,
            )
            + CONTEXT_FEATURES
            + [
                "has_rhr",
                "has_temperature",
            ]
        ),
    }

    return {
        name: unique_preserving_order(
            [
                feature
                for feature in features
                if feature in temporal_features
            ]
        )
        for name, features in feature_sets.items()
    }


def prune_feature_set(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], dict[str, Any]]:
    missing_features = [
        feature
        for feature in features
        if feature not in train.columns
    ]

    if missing_features:
        raise KeyError(
            f"Feature columns missing from table: "
            f"{missing_features}"
        )

    numeric_features = [
        feature
        for feature in features
        if pd.api.types.is_numeric_dtype(
            train[feature]
        )
    ]

    missingness = (
        train[numeric_features]
        .isna()
        .mean()
    )

    excessive_missingness = (
        missingness[
            missingness
            > MISSINGNESS_THRESHOLD
        ]
        .index
        .tolist()
    )

    remaining = [
        feature
        for feature in numeric_features
        if feature
        not in excessive_missingness
    ]

    constants = [
        feature
        for feature in remaining
        if train[feature].nunique(
            dropna=True
        ) <= 1
    ]

    selected = [
        feature
        for feature in remaining
        if feature not in constants
    ]

    return selected, {
        "original_count": len(features),
        "selected_count": len(selected),
        "removed_excessive_missingness": (
            excessive_missingness
        ),
        "removed_constant": constants,
    }


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def calculate_metrics(
    model: Pipeline,
    dataframe: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    truth = dataframe["phase"]
    prediction = model.predict(
        dataframe[features]
    )

    probabilities = model.predict_proba(
        dataframe[features]
    )

    classes = list(model.classes_)

    return {
        "accuracy": float(
            accuracy_score(
                truth,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                truth,
                prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=PHASES,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                truth,
                probabilities,
                labels=classes,
            )
        ),
    }


def grouped_cross_validation(
    train: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    splitter = GroupKFold(
        n_splits=CV_SPLITS
    )

    x = train[features]
    y = train["phase"]
    groups = train["id"]

    fold_results = []

    for fold_number, (
        fit_indices,
        evaluation_indices,
    ) in enumerate(
        splitter.split(
            x,
            y,
            groups=groups,
        ),
        start=1,
    ):
        fit_data = train.iloc[
            fit_indices
        ]

        evaluation_data = train.iloc[
            evaluation_indices
        ]

        model = make_model()

        model.fit(
            fit_data[features],
            fit_data["phase"],
        )

        metrics = calculate_metrics(
            model,
            evaluation_data,
            features,
        )

        metrics["fold"] = fold_number
        metrics["evaluation_participants"] = int(
            evaluation_data[
                "id"
            ].nunique()
        )

        fold_results.append(metrics)

    frame = pd.DataFrame(
        fold_results
    )

    return {
        "folds": fold_results,
        "mean_balanced_accuracy": float(
            frame[
                "balanced_accuracy"
            ].mean()
        ),
        "std_balanced_accuracy": float(
            frame[
                "balanced_accuracy"
            ].std(ddof=0)
        ),
        "mean_macro_f1": float(
            frame["macro_f1"].mean()
        ),
        "std_macro_f1": float(
            frame["macro_f1"].std(ddof=0)
        ),
        "mean_log_loss": float(
            frame["log_loss"].mean()
        ),
    }


def participant_accuracy_summary(
    model: Pipeline,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    predictions = model.predict(
        validation[features]
    )

    temporary = validation[
        ["id", "phase"]
    ].copy()

    temporary["prediction"] = predictions
    temporary["correct"] = (
        temporary["phase"]
        == temporary["prediction"]
    )

    participant_accuracy = (
        temporary.groupby("id")[
            "correct"
        ].mean()
    )

    return {
        "minimum": float(
            participant_accuracy.min()
        ),
        "median": float(
            participant_accuracy.median()
        ),
        "maximum": float(
            participant_accuracy.max()
        ),
    }


def evaluate_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    cross_validation = (
        grouped_cross_validation(
            train,
            features,
        )
    )

    model = make_model()

    model.fit(
        train[features],
        train["phase"],
    )

    overall = calculate_metrics(
        model,
        validation,
        features,
    )

    interval_results = {}

    for interval, subset in (
        validation.groupby(
            "study_interval"
        )
    ):
        interval_results[
            str(interval)
        ] = calculate_metrics(
            model,
            subset,
            features,
        )

    interval_balanced_values = [
        values["balanced_accuracy"]
        for values
        in interval_results.values()
    ]

    interval_gap = (
        max(interval_balanced_values)
        - min(interval_balanced_values)
    )

    participant_summary = (
        participant_accuracy_summary(
            model,
            validation,
            features,
        )
    )

    return {
        "feature_count": len(features),
        "training_grouped_cv": (
            cross_validation
        ),
        "validation_overall": overall,
        "validation_by_interval": (
            interval_results
        ),
        "validation_interval_gap": float(
            interval_gap
        ),
        "validation_participant_accuracy": (
            participant_summary
        ),
    }


def print_results(
    results: dict[str, Any],
) -> None:
    print("\n" + "=" * 124)
    print("FEATURE-SET COMPARISON")
    print("=" * 124)

    header = (
        f"{'Feature set':38s}"
        f"{'Features':>10s}"
        f"{'CV Bal':>10s}"
        f"{'CV SD':>9s}"
        f"{'Val Bal':>10s}"
        f"{'2022 Bal':>11s}"
        f"{'2024 Bal':>11s}"
        f"{'Gap':>9s}"
        f"{'Val F1':>10s}"
    )

    print(header)
    print("-" * len(header))

    for name, result in results.items():
        cv = result[
            "training_grouped_cv"
        ]

        validation = result[
            "validation_overall"
        ]

        intervals = result[
            "validation_by_interval"
        ]

        result_2022 = intervals.get(
            "2022",
            {},
        ).get(
            "balanced_accuracy",
            float("nan"),
        )

        result_2024 = intervals.get(
            "2024",
            {},
        ).get(
            "balanced_accuracy",
            float("nan"),
        )

        print(
            f"{name:38s}"
            f"{result['feature_count']:10d}"
            f"{cv['mean_balanced_accuracy']:10.4f}"
            f"{cv['std_balanced_accuracy']:9.4f}"
            f"{validation['balanced_accuracy']:10.4f}"
            f"{result_2022:11.4f}"
            f"{result_2024:11.4f}"
            f"{result['validation_interval_gap']:9.4f}"
            f"{validation['macro_f1']:10.4f}"
        )


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE ROBUST FEATURE-SET AUDIT")
    print("=" * 76)

    dataframe, manifest = load_inputs()

    train = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation = dataframe[
        dataframe["split"]
        == "validation"
    ].copy()

    test = dataframe[
        dataframe["split"] == "test"
    ]

    print(f"Training rows: {len(train):,}")
    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )
    print(
        "Test rows reserved and not evaluated: "
        f"{len(test):,}"
    )

    temporal_features = manifest[
        "temporal_features"
    ]

    raw_feature_sets = (
        build_feature_sets(
            temporal_features
        )
    )

    results = {}
    selected_features_by_set = {}
    pruning_audits = {}

    for feature_set_name, raw_features in (
        raw_feature_sets.items()
    ):
        print(
            f"\nEvaluating: "
            f"{feature_set_name}"
        )

        selected_features, audit = (
            prune_feature_set(
                train,
                raw_features,
            )
        )

        if not selected_features:
            raise ValueError(
                f"No usable features remained for "
                f"{feature_set_name}."
            )

        selected_features_by_set[
            feature_set_name
        ] = selected_features

        pruning_audits[
            feature_set_name
        ] = audit

        results[
            feature_set_name
        ] = evaluate_feature_set(
            train,
            validation,
            selected_features,
        )

    print_results(results)

    # One-standard-error stability rule:
    # 1. Find the feature set with the highest grouped-CV mean.
    # 2. Retain feature sets within one standard error of that result.
    # 3. Among those statistically competitive models, prefer:
    #    a. the smallest study-interval performance gap,
    #    b. fewer features,
    #    c. higher overall validation balanced accuracy.

    best_cv_name = max(
        results,
        key=lambda name: (
            results[name][
                "training_grouped_cv"
            ][
                "mean_balanced_accuracy"
            ]
        ),
    )

    best_cv_result = results[
        best_cv_name
    ]["training_grouped_cv"]

    best_cv_mean = float(
        best_cv_result[
            "mean_balanced_accuracy"
        ]
    )

    best_cv_standard_error = float(
        best_cv_result[
            "std_balanced_accuracy"
        ]
        / np.sqrt(CV_SPLITS)
    )

    eligibility_threshold = (
        best_cv_mean
        - best_cv_standard_error
    )

    eligible_feature_sets = [
        name
        for name, result in results.items()
        if (
            result[
                "training_grouped_cv"
            ][
                "mean_balanced_accuracy"
            ]
            >= eligibility_threshold
        )
    ]

    recommended_name = min(
        eligible_feature_sets,
        key=lambda name: (
            results[name][
                "validation_interval_gap"
            ],
            results[name][
                "feature_count"
            ],
            -results[name][
                "validation_overall"
            ][
                "balanced_accuracy"
            ],
        ),
    )

    recommended_result = results[
        recommended_name
    ]

    selection_details = {
        "rule": (
            "One-standard-error stability rule"
        ),
        "best_cv_feature_set": (
            best_cv_name
        ),
        "best_cv_mean_balanced_accuracy": (
            best_cv_mean
        ),
        "best_cv_standard_error": (
            best_cv_standard_error
        ),
        "eligibility_threshold": (
            eligibility_threshold
        ),
        "eligible_feature_sets": (
            eligible_feature_sets
        ),
        "secondary_selection_order": [
            "smallest validation study-interval gap",
            "fewest features",
            "highest validation balanced accuracy",
        ],
    }

    print(
        "\nBest raw grouped-CV feature set:"
    )
    print(f"  {best_cv_name}")
    print(
        "  Mean balanced accuracy: "
        f"{best_cv_mean:.4f}"
    )
    print(
        "  Standard error: "
        f"{best_cv_standard_error:.4f}"
    )
    print(
        "  One-SE eligibility threshold: "
        f"{eligibility_threshold:.4f}"
    )

    print("\nEligible feature sets:")

    for feature_set_name in (
        eligible_feature_sets
    ):
        print(f"  - {feature_set_name}")

    print(
        "\nRecommended by stability-aware rule:"
    )
    print(f"  {recommended_name}")
    print(
        "  Feature count: "
        f"{recommended_result['feature_count']}"
    )
    print(
        "  CV balanced accuracy: "
        f"{recommended_result['training_grouped_cv']['mean_balanced_accuracy']:.4f}"
    )
    print(
        "  Validation balanced accuracy: "
        f"{recommended_result['validation_overall']['balanced_accuracy']:.4f}"
    )
    print(
        "  2022–2024 interval gap: "
        f"{recommended_result['validation_interval_gap']:.4f}"
    )

    report = {
        "test_evaluated": False,
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "reserved_test_rows": int(
            len(test)
        ),
        "recommended_feature_set": (
            recommended_name
        ),
        "selection_details": (
            selection_details
        ),
        "results": results,
        "pruning_audits": pruning_audits,
    }

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

    recommendation = {
        "feature_set_name": (
            recommended_name
        ),
        "features": (
            selected_features_by_set[
                recommended_name
            ]
        ),
        "feature_count": len(
            selected_features_by_set[
                recommended_name
            ]
        ),
        "selection_method": (
            "One-standard-error stability rule. "
            "Feature sets within one standard error "
            "of the best participant-grouped CV score "
            "were considered statistically competitive. "
            "The final set was selected by the smallest "
            "2022–2024 validation interval gap, then "
            "feature count, then validation balanced "
            "accuracy."
        ),
        "selection_details": (
            selection_details
        ),
        "validation_metrics": (
            recommended_result[
                "validation_overall"
            ]
        ),
        "validation_by_interval": (
            recommended_result[
                "validation_by_interval"
            ]
        ),
        "test_evaluated": False,
    }

    with RECOMMENDATION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            recommendation,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(f"\nSafe comparison report:\n{REPORT_PATH}")
    print(
        "\nRecommended feature subset:\n"
        f"{RECOMMENDATION_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()