from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

FEATURE_TABLE_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_temporal_features.parquet"
)

FEATURE_MANIFEST_PATH = (
    ARTIFACTS_DIR
    / "temporal_feature_manifest.json"
)

OUTPUT_METRICS_PATH = (
    REPORTS_DIR
    / "checkpoint5a_validation_baselines.json"
)

OUTPUT_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint5a_validation_predictions.csv"
)

MODELLING_MANIFEST_PATH = (
    ARTIFACTS_DIR
    / "modelling_feature_manifest.json"
)

BEST_MODEL_PATH = (
    ARTIFACTS_DIR
    / "baseline_best_validation_model.joblib"
)

RANDOM_SEED = 20260719
MISSINGNESS_THRESHOLD = 0.95

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

NON_CURRENT_FEATURES = {
    "days_since_previous_record",
    "is_first_record_in_interval",
    "has_calendar_gap",
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


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE_PATH}"
        )

    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {FEATURE_MANIFEST_PATH}"
        )

    dataframe = pd.read_parquet(FEATURE_TABLE_PATH)

    with FEATURE_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    required_columns = [
        "id",
        "study_interval",
        "day_in_study",
        "phase",
        "split",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_required:
        raise KeyError(
            f"Feature table is missing: {missing_required}"
        )

    return dataframe, manifest


def prune_features(
    train: pd.DataFrame,
    candidate_features: list[str],
) -> tuple[list[str], dict[str, Any]]:
    missing_from_table = [
        feature
        for feature in candidate_features
        if feature not in train.columns
    ]

    if missing_from_table:
        raise KeyError(
            "Manifest features missing from table: "
            f"{missing_from_table}"
        )

    non_numeric_features = [
        feature
        for feature in candidate_features
        if not pd.api.types.is_numeric_dtype(
            train[feature]
        )
    ]

    usable_candidates = [
        feature
        for feature in candidate_features
        if feature not in non_numeric_features
    ]

    missingness = (
        train[usable_candidates]
        .isna()
        .mean()
    )

    excessive_missingness = sorted(
        missingness[
            missingness > MISSINGNESS_THRESHOLD
        ].index.tolist()
    )

    after_missingness = [
        feature
        for feature in usable_candidates
        if feature not in excessive_missingness
    ]

    constant_features = []

    for feature in after_missingness:
        unique_count = train[feature].nunique(
            dropna=True
        )

        if unique_count <= 1:
            constant_features.append(feature)

    selected_features = [
        feature
        for feature in after_missingness
        if feature not in constant_features
    ]

    audit = {
        "candidate_count": len(candidate_features),
        "selected_count": len(selected_features),
        "non_numeric_features": non_numeric_features,
        "excessive_missingness_features": (
            excessive_missingness
        ),
        "constant_features": constant_features,
        "missingness_threshold": (
            MISSINGNESS_THRESHOLD
        ),
    }

    return selected_features, audit


def identify_current_features(
    all_features: list[str],
) -> list[str]:
    current_features = []

    for feature in all_features:
        is_temporal = "__" in feature

        if is_temporal:
            continue

        if feature in NON_CURRENT_FEATURES:
            continue

        current_features.append(feature)

    return current_features


def make_logistic_pipeline() -> Pipeline:
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


def make_gradient_boosting_pipeline() -> Pipeline:
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
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_model(
    model_name: str,
    model: Any,
    features: list[str],
    validation: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x_validation = validation[features]
    y_validation = validation["phase"]

    predictions = model.predict(x_validation)

    probabilities = model.predict_proba(
        x_validation
    )

    model_classes = list(model.classes_)

    aligned_probabilities = np.zeros(
        (
            len(validation),
            len(PHASES),
        ),
        dtype=float,
    )

    for phase_index, phase in enumerate(PHASES):
        if phase not in model_classes:
            continue

        source_index = model_classes.index(phase)

        aligned_probabilities[
            :,
            phase_index,
        ] = probabilities[:, source_index]

    metrics = {
        "model": model_name,
        "feature_count": len(features),
        "accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_validation,
                probabilities,
                labels=model_classes,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_validation,
                predictions,
                labels=PHASES,
            )
            .astype(int)
            .tolist()
        ),
        "classification_report": (
            classification_report(
                y_validation,
                predictions,
                labels=PHASES,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    prediction_frame = validation[
        [
            "id",
            "study_interval",
            "day_in_study",
            "phase",
        ]
    ].copy()

    prediction_frame["model"] = model_name
    prediction_frame["predicted_phase"] = predictions
    prediction_frame["correct"] = (
        prediction_frame["phase"]
        == prediction_frame["predicted_phase"]
    )

    for phase_index, phase in enumerate(PHASES):
        prediction_frame[
            f"probability_{phase.lower()}"
        ] = aligned_probabilities[
            :,
            phase_index,
        ]

    prediction_frame["maximum_probability"] = (
        aligned_probabilities.max(axis=1)
    )

    return metrics, prediction_frame


def print_metric_summary(
    results: dict[str, dict[str, Any]],
) -> None:
    print("\n" + "=" * 92)
    print("VALIDATION RESULTS")
    print("=" * 92)

    header = (
        f"{'Model':38s}"
        f"{'Features':>10s}"
        f"{'Accuracy':>11s}"
        f"{'Balanced':>11s}"
        f"{'Macro F1':>11s}"
        f"{'Log loss':>11s}"
    )

    print(header)
    print("-" * len(header))

    for model_name, metrics in results.items():
        print(
            f"{model_name:38s}"
            f"{metrics['feature_count']:10d}"
            f"{metrics['accuracy']:11.4f}"
            f"{metrics['balanced_accuracy']:11.4f}"
            f"{metrics['macro_f1']:11.4f}"
            f"{metrics['log_loss']:11.4f}"
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
    print("MOSAIC-PHASE VALIDATION BASELINES")
    print("=" * 76)

    dataframe, original_manifest = load_inputs()

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
    print(f"Validation rows: {len(validation):,}")
    print(
        "Test rows reserved and not evaluated: "
        f"{len(test):,}"
    )

    candidate_features = (
        original_manifest[
            "all_model_features"
        ]
    )

    temporal_features, pruning_audit = (
        prune_features(
            train,
            candidate_features,
        )
    )

    current_features = identify_current_features(
        temporal_features
    )

    print(
        "\nCandidate features: "
        f"{pruning_audit['candidate_count']}"
    )

    print(
        "Selected temporal features: "
        f"{len(temporal_features)}"
    )

    print(
        "Selected current-day features: "
        f"{len(current_features)}"
    )

    print(
        "Removed for excessive missingness: "
        f"{pruning_audit['excessive_missingness_features']}"
    )

    print(
        "Removed as constant: "
        f"{pruning_audit['constant_features']}"
    )

    x_train_current = train[current_features]
    x_train_temporal = train[temporal_features]
    y_train = train["phase"]

    models: dict[str, tuple[Any, list[str]]] = {}

    dummy = DummyClassifier(
        strategy="most_frequent",
    )

    dummy.fit(
        x_train_current,
        y_train,
    )

    models["Majority-class baseline"] = (
        dummy,
        current_features,
    )

    current_logistic = make_logistic_pipeline()

    current_logistic.fit(
        x_train_current,
        y_train,
    )

    models["Current-day logistic regression"] = (
        current_logistic,
        current_features,
    )

    temporal_logistic = make_logistic_pipeline()

    temporal_logistic.fit(
        x_train_temporal,
        y_train,
    )

    models["Temporal logistic regression"] = (
        temporal_logistic,
        temporal_features,
    )

    temporal_boosting = (
        make_gradient_boosting_pipeline()
    )

    balanced_sample_weights = (
        compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
    )

    temporal_boosting.fit(
        x_train_temporal,
        y_train,
        model__sample_weight=(
            balanced_sample_weights
        ),
    )

    models["Temporal gradient boosting"] = (
        temporal_boosting,
        temporal_features,
    )

    results: dict[str, dict[str, Any]] = {}
    prediction_frames = []

    for model_name, (
        model,
        features,
    ) in models.items():
        print(f"\nEvaluating {model_name}...")

        metrics, prediction_frame = (
            evaluate_model(
                model_name=model_name,
                model=model,
                features=features,
                validation=validation,
            )
        )

        results[model_name] = metrics
        prediction_frames.append(
            prediction_frame
        )

    print_metric_summary(results)

    ranked_models = sorted(
        results.items(),
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["macro_f1"],
        ),
        reverse=True,
    )

    best_model_name = ranked_models[0][0]
    best_model, best_features = (
        models[best_model_name]
    )

    print("\nBest validation model:")
    print(f"  {best_model_name}")
    print(
        "  Balanced accuracy: "
        f"{results[best_model_name]['balanced_accuracy']:.4f}"
    )
    print(
        "  Macro F1: "
        f"{results[best_model_name]['macro_f1']:.4f}"
    )

    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "features": best_features,
        "phase_order": PHASES,
        "validation_metrics": (
            results[best_model_name]
        ),
        "test_evaluated": False,
    }

    joblib.dump(
        model_bundle,
        BEST_MODEL_PATH,
    )

    modelling_manifest = {
        "random_seed": RANDOM_SEED,
        "test_evaluated": False,
        "candidate_features": (
            candidate_features
        ),
        "temporal_features": temporal_features,
        "current_features": current_features,
        "pruning_audit": pruning_audit,
        "phase_order": PHASES,
        "best_validation_model": (
            best_model_name
        ),
    }

    with MODELLING_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            modelling_manifest,
            file,
            indent=2,
            default=json_safe,
        )

    output_payload = {
        "test_evaluated": False,
        "train_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "reserved_test_rows": int(len(test)),
        "best_validation_model": (
            best_model_name
        ),
        "models": results,
        "feature_pruning": pruning_audit,
    }

    with OUTPUT_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_payload,
            file,
            indent=2,
            default=json_safe,
        )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(f"\nBest model bundle:\n{BEST_MODEL_PATH}")
    print(
        f"\nModelling manifest:\n"
        f"{MODELLING_MANIFEST_PATH}"
    )
    print(
        f"\nValidation metrics:\n"
        f"{OUTPUT_METRICS_PATH}"
    )
    print(
        f"\nValidation predictions:\n"
        f"{OUTPUT_PREDICTIONS_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()