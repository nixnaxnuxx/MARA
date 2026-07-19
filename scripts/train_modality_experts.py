from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
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

TEMPORAL_MANIFEST_PATH = (
    ARTIFACTS_DIR
    / "temporal_feature_manifest.json"
)

RECOMMENDED_FEATURE_PATH = (
    ARTIFACTS_DIR
    / "recommended_feature_subset.json"
)

MODEL_BUNDLE_PATH = (
    ARTIFACTS_DIR
    / "modality_expert_models.joblib"
)

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint6a_modality_experts.json"
)

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint6a_expert_validation_predictions.csv"
)

RANDOM_SEED = 20260719
CV_SPLITS = 4
MISSINGNESS_THRESHOLD = 0.95

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

# Context that does not reveal whether unrelated sensors are present.
SHARED_CONTEXT_FEATURES = [
    "is_weekend",
    "days_since_previous_record",
    "is_first_record_in_interval",
    "has_calendar_gap",
]


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


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    required_paths = [
        FEATURE_TABLE_PATH,
        TEMPORAL_MANIFEST_PATH,
        RECOMMENDED_FEATURE_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    dataframe = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    with TEMPORAL_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        temporal_manifest = json.load(file)

    with RECOMMENDED_FEATURE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        recommendation = json.load(file)

    return (
        dataframe,
        temporal_manifest,
        recommendation,
    )


def unique_preserving_order(
    values: list[str],
) -> list[str]:
    return list(dict.fromkeys(values))


def build_expert_feature_sets(
    temporal_manifest: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, list[str]]:
    modality_features = temporal_manifest[
        "modality_features"
    ]

    all_model_features = temporal_manifest[
        "all_model_features"
    ]

    feature_sets = {
        "RHR expert": (
            modality_features["rhr"]
            + SHARED_CONTEXT_FEATURES
        ),

        "Temperature expert": (
            modality_features["temperature"]
            + SHARED_CONTEXT_FEATURES
        ),

        "Sleep expert": (
            modality_features["sleep"]
            + SHARED_CONTEXT_FEATURES
        ),

        "Activity expert": (
            modality_features["activity"]
            + SHARED_CONTEXT_FEATURES
        ),

        "Stable RHR + temperature core": (
            recommendation["features"]
        ),

        "Full passive reference": (
            all_model_features
        ),
    }

    return {
        name: unique_preserving_order(features)
        for name, features in feature_sets.items()
    }


def prune_features(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], dict[str, Any]]:
    missing_columns = [
        feature
        for feature in features
        if feature not in train.columns
    ]

    if missing_columns:
        raise KeyError(
            "Feature columns missing from table: "
            f"{missing_columns}"
        )

    numeric_features = [
        feature
        for feature in features
        if pd.api.types.is_numeric_dtype(
            train[feature]
        )
    ]

    non_numeric_features = [
        feature
        for feature in features
        if feature not in numeric_features
    ]

    missingness = (
        train[numeric_features]
        .isna()
        .mean()
    )

    excessive_missingness = (
        missingness[
            missingness > MISSINGNESS_THRESHOLD
        ]
        .index
        .tolist()
    )

    remaining = [
        feature
        for feature in numeric_features
        if feature not in excessive_missingness
    ]

    constant_features = [
        feature
        for feature in remaining
        if train[feature].nunique(
            dropna=True
        ) <= 1
    ]

    selected = [
        feature
        for feature in remaining
        if feature not in constant_features
    ]

    return selected, {
        "candidate_count": len(features),
        "selected_count": len(selected),
        "non_numeric_features": non_numeric_features,
        "removed_excessive_missingness": (
            excessive_missingness
        ),
        "removed_constant": constant_features,
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


def align_probabilities(
    probabilities: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    aligned = np.zeros(
        (
            probabilities.shape[0],
            len(PHASES),
        ),
        dtype=float,
    )

    for target_index, phase in enumerate(PHASES):
        if phase not in classes:
            continue

        source_index = classes.index(phase)

        aligned[:, target_index] = (
            probabilities[:, source_index]
        )

    return aligned


def calculate_metrics(
    model: Pipeline,
    dataframe: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    truth = dataframe["phase"]

    predictions = model.predict(
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
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                truth,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
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

    folds = []

    for fold_number, (
        fit_indices,
        evaluation_indices,
    ) in enumerate(
        splitter.split(
            train[features],
            train["phase"],
            groups=train["id"],
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
        metrics["participants"] = int(
            evaluation_data["id"].nunique()
        )

        folds.append(metrics)

    frame = pd.DataFrame(folds)

    return {
        "folds": folds,
        "mean_balanced_accuracy": float(
            frame["balanced_accuracy"].mean()
        ),
        "std_balanced_accuracy": float(
            frame["balanced_accuracy"].std(
                ddof=0
            )
        ),
        "mean_macro_f1": float(
            frame["macro_f1"].mean()
        ),
        "std_macro_f1": float(
            frame["macro_f1"].std(
                ddof=0
            )
        ),
        "mean_log_loss": float(
            frame["log_loss"].mean()
        ),
    }


def create_prediction_frame(
    model_name: str,
    model: Pipeline,
    validation: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    probabilities = model.predict_proba(
        validation[features]
    )

    classes = list(model.classes_)

    aligned = align_probabilities(
        probabilities,
        classes,
    )

    predictions = np.array(PHASES)[
        aligned.argmax(axis=1)
    ]

    output = validation[
        [
            "id",
            "study_interval",
            "day_in_study",
            "phase",
        ]
    ].copy()

    output["expert"] = model_name
    output["predicted_phase"] = predictions

    output["correct"] = (
        output["phase"]
        == output["predicted_phase"]
    )

    for phase_index, phase in enumerate(PHASES):
        output[
            f"probability_{phase.lower()}"
        ] = aligned[:, phase_index]

    output["maximum_probability"] = (
        aligned.max(axis=1)
    )

    return output


def evaluate_expert(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[
    Pipeline,
    dict[str, Any],
]:
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

    overall_metrics = calculate_metrics(
        model,
        validation,
        features,
    )

    interval_metrics = {}

    for interval, subset in (
        validation.groupby(
            "study_interval"
        )
    ):
        interval_metrics[str(interval)] = (
            calculate_metrics(
                model,
                subset,
                features,
            )
        )

    interval_balanced = [
        values["balanced_accuracy"]
        for values in interval_metrics.values()
    ]

    interval_gap = float(
        max(interval_balanced)
        - min(interval_balanced)
    )

    return model, {
        "feature_count": len(features),
        "training_grouped_cv": (
            cross_validation
        ),
        "validation_overall": (
            overall_metrics
        ),
        "validation_by_interval": (
            interval_metrics
        ),
        "validation_interval_gap": (
            interval_gap
        ),
    }


def print_results(
    results: dict[str, Any],
) -> None:
    print("\n" + "=" * 126)
    print("MODALITY-EXPERT COMPARISON")
    print("=" * 126)

    header = (
        f"{'Expert':38s}"
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
        cv = result["training_grouped_cv"]
        validation = result[
            "validation_overall"
        ]

        interval_2022 = (
            result["validation_by_interval"]
            .get("2022", {})
            .get(
                "balanced_accuracy",
                float("nan"),
            )
        )

        interval_2024 = (
            result["validation_by_interval"]
            .get("2024", {})
            .get(
                "balanced_accuracy",
                float("nan"),
            )
        )

        print(
            f"{name:38s}"
            f"{result['feature_count']:10d}"
            f"{cv['mean_balanced_accuracy']:10.4f}"
            f"{cv['std_balanced_accuracy']:9.4f}"
            f"{validation['balanced_accuracy']:10.4f}"
            f"{interval_2022:11.4f}"
            f"{interval_2024:11.4f}"
            f"{result['validation_interval_gap']:9.4f}"
            f"{validation['macro_f1']:10.4f}"
        )


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
    print("MOSAIC-PHASE MODALITY EXPERTS")
    print("=" * 76)

    (
        dataframe,
        temporal_manifest,
        recommendation,
    ) = load_inputs()

    train = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test = dataframe[
        dataframe["split"] == "test"
    ].copy()

    print(f"Training rows: {len(train):,}")
    print(
        f"Validation rows: {len(validation):,}"
    )
    print(
        "Test rows reserved and not evaluated: "
        f"{len(test):,}"
    )

    raw_feature_sets = (
        build_expert_feature_sets(
            temporal_manifest,
            recommendation,
        )
    )

    expert_models = {}
    results = {}
    pruning_audits = {}
    prediction_frames = []

    for expert_name, raw_features in (
        raw_feature_sets.items()
    ):
        print(
            f"\nTraining: {expert_name}"
        )

        selected_features, pruning_audit = (
            prune_features(
                train,
                raw_features,
            )
        )

        if not selected_features:
            raise ValueError(
                f"No usable features remained for "
                f"{expert_name}."
            )

        model, result = evaluate_expert(
            train,
            validation,
            selected_features,
        )

        prediction_frame = (
            create_prediction_frame(
                model_name=expert_name,
                model=model,
                validation=validation,
                features=selected_features,
            )
        )

        expert_models[expert_name] = {
            "model": model,
            "features": selected_features,
            "phase_order": PHASES,
            "validation_metrics": result,
        }

        results[expert_name] = result
        pruning_audits[
            expert_name
        ] = pruning_audit

        prediction_frames.append(
            prediction_frame
        )

    print_results(results)

    eligible_experts = []

    for expert_name in [
        "RHR expert",
        "Temperature expert",
        "Sleep expert",
        "Activity expert",
    ]:
        result = results[expert_name]

        cv_balanced = result[
            "training_grouped_cv"
        ]["mean_balanced_accuracy"]

        validation_balanced = result[
            "validation_overall"
        ]["balanced_accuracy"]

        # Conservative screening rule:
        # both grouped CV and validation must exceed
        # random four-class performance by at least 0.03.
        if (
            cv_balanced >= 0.28
            and validation_balanced >= 0.28
        ):
            eligible_experts.append(
                expert_name
            )

    print("\nExperts eligible for fusion screening:")

    if eligible_experts:
        for expert_name in eligible_experts:
            print(f"  - {expert_name}")
    else:
        print("  None")

    output_bundle = {
        "models": expert_models,
        "fusion_screening_candidates": (
            eligible_experts
        ),
        "test_evaluated": False,
    }

    joblib.dump(
        output_bundle,
        MODEL_BUNDLE_PATH,
    )

    report = {
        "test_evaluated": False,
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "reserved_test_rows": int(len(test)),
        "experts": results,
        "pruning_audits": pruning_audits,
        "fusion_screening_candidates": (
            eligible_experts
        ),
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

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        f"\nExpert model bundle:\n"
        f"{MODEL_BUNDLE_PATH}"
    )

    print(
        f"\nSafe expert report:\n"
        f"{REPORT_PATH}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()