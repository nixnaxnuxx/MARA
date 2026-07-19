from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

METRICS_PATH = (
    REPORTS_DIR
    / "checkpoint5a_validation_baselines.json"
)

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint5a_validation_predictions.csv"
)

FEATURES_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_temporal_features.parquet"
)

OUTPUT_PATH = (
    REPORTS_DIR
    / "checkpoint5b_validation_diagnostics.json"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

KEYS = [
    "id",
    "study_interval",
    "day_in_study",
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
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in [
        METRICS_PATH,
        PREDICTIONS_PATH,
        FEATURES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    return metrics, predictions, features


def calculate_metrics(
    truth: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
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
    }


def summarize_participant_performance(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    participant_results = []

    for _, participant in dataframe.groupby(
        "id"
    ):
        metrics = calculate_metrics(
            participant["phase"],
            participant["predicted_phase"],
        )

        metrics["days"] = int(
            len(participant)
        )

        participant_results.append(
            metrics
        )

    result_frame = pd.DataFrame(
        participant_results
    )

    output: dict[str, Any] = {}

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "days",
    ]:
        output[metric] = {
            "minimum": float(
                result_frame[metric].min()
            ),
            "median": float(
                result_frame[metric].median()
            ),
            "maximum": float(
                result_frame[metric].max()
            ),
        }

    return output


def summarize_group(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for group_value, subset in dataframe.groupby(
        column,
        dropna=False,
    ):
        output[str(group_value)] = {
            "rows": int(len(subset)),
            "represented_phases": int(
                subset["phase"].nunique()
            ),
            **calculate_metrics(
                subset["phase"],
                subset["predicted_phase"],
            ),
        }

    return output


def build_confidence_table(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    boundaries = [
        0.00,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        1.000001,
    ]

    labels = [
        "0.00–0.35",
        "0.35–0.45",
        "0.45–0.55",
        "0.55–0.65",
        "0.65–0.75",
        "0.75–0.85",
        "0.85–1.00",
    ]

    dataframe = dataframe.copy()

    dataframe["confidence_bin"] = pd.cut(
        dataframe["maximum_probability"],
        bins=boundaries,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    table = []

    for confidence_bin, subset in dataframe.groupby(
        "confidence_bin",
        observed=True,
    ):
        if subset.empty:
            continue

        table.append(
            {
                "confidence_bin": str(
                    confidence_bin
                ),
                "rows": int(len(subset)),
                "mean_confidence": float(
                    subset[
                        "maximum_probability"
                    ].mean()
                ),
                "accuracy": float(
                    subset["correct"].mean()
                ),
            }
        )

    return table


def print_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    decimals: int | None = None,
) -> None:
    frame = pd.DataFrame(
        matrix,
        index=[
            f"True {phase}"
            for phase in PHASES
        ],
        columns=[
            f"Pred {phase}"
            for phase in PHASES
        ],
    )

    print("\n" + title)

    if decimals is None:
        print(frame.to_string())
    else:
        print(
            frame.round(decimals).to_string()
        )


def main() -> None:
    print("=" * 76)
    print("MOSAIC-PHASE VALIDATION DIAGNOSTICS")
    print("=" * 76)

    metrics, predictions, features = (
        load_inputs()
    )

    best_model = metrics[
        "best_validation_model"
    ]

    print(f"Best validation model: {best_model}")

    best_predictions = predictions[
        predictions["model"] == best_model
    ].copy()

    if best_predictions.empty:
        raise ValueError(
            "No prediction rows were found for "
            f"{best_model}."
        )

    validation_features = features[
        features["split"] == "validation"
    ].copy()

    context_columns = KEYS + [
        "available_modality_count",
        "has_rhr",
        "has_temperature",
        "has_sleep",
        "has_activity",
    ]

    validation_context = validation_features[
        context_columns
    ].copy()

    combined = best_predictions.merge(
        validation_context,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if combined[
        "available_modality_count"
    ].isna().any():
        raise ValueError(
            "Some prediction rows could not be matched "
            "to validation features."
        )

    overall_metrics = calculate_metrics(
        combined["phase"],
        combined["predicted_phase"],
    )

    raw_matrix = confusion_matrix(
        combined["phase"],
        combined["predicted_phase"],
        labels=PHASES,
    )

    row_totals = raw_matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        raw_matrix,
        row_totals,
        out=np.zeros_like(
            raw_matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    report = classification_report(
        combined["phase"],
        combined["predicted_phase"],
        labels=PHASES,
        output_dict=True,
        zero_division=0,
    )

    prediction_distribution = (
        combined["predicted_phase"]
        .value_counts()
        .reindex(
            PHASES,
            fill_value=0,
        )
    )

    participant_summary = (
        summarize_participant_performance(
            combined
        )
    )

    modality_count_summary = summarize_group(
        combined,
        "available_modality_count",
    )

    study_interval_summary = summarize_group(
        combined,
        "study_interval",
    )

    confidence_table = (
        build_confidence_table(
            combined
        )
    )

    correct_confidence = float(
        combined.loc[
            combined["correct"],
            "maximum_probability",
        ].mean()
    )

    incorrect_confidence = float(
        combined.loc[
            ~combined["correct"],
            "maximum_probability",
        ].mean()
    )

    print("\nOverall metrics:")

    for metric, value in overall_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")

    print("\nPer-phase results:")

    for phase in PHASES:
        phase_result = report[phase]

        print(
            f"  {phase:12s} "
            f"precision={phase_result['precision']:.4f} "
            f"recall={phase_result['recall']:.4f} "
            f"f1={phase_result['f1-score']:.4f} "
            f"support={int(phase_result['support'])}"
        )

    print_confusion_matrix(
        raw_matrix,
        "Confusion matrix — counts:",
    )

    print_confusion_matrix(
        normalized_matrix,
        "Confusion matrix — row-normalized:",
        decimals=3,
    )

    print("\nPrediction distribution:")

    for phase, count in (
        prediction_distribution.items()
    ):
        print(
            f"  {phase:12s}: {count}"
        )

    print("\nParticipant-level performance:")

    for metric, values in (
        participant_summary.items()
    ):
        print(
            f"  {metric:20s} "
            f"min={values['minimum']:.4f} "
            f"median={values['median']:.4f} "
            f"max={values['maximum']:.4f}"
        )

    print("\nPerformance by available modality count:")

    for group_value, values in (
        modality_count_summary.items()
    ):
        print(
            f"  {group_value} modalities: "
            f"n={values['rows']}, "
            f"phases={values['represented_phases']}, "
            f"accuracy={values['accuracy']:.4f}, "
            f"balanced={values['balanced_accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )

    print("\nPerformance by study interval:")

    for interval, values in (
        study_interval_summary.items()
    ):
        print(
            f"  Interval {interval}: "
            f"n={values['rows']}, "
            f"accuracy={values['accuracy']:.4f}, "
            f"balanced={values['balanced_accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )

    print("\nConfidence reliability:")

    for row in confidence_table:
        print(
            f"  {row['confidence_bin']:12s} "
            f"n={row['rows']:4d} "
            f"mean confidence={row['mean_confidence']:.4f} "
            f"accuracy={row['accuracy']:.4f}"
        )

    print(
        "\nMean confidence on correct predictions: "
        f"{correct_confidence:.4f}"
    )

    print(
        "Mean confidence on incorrect predictions: "
        f"{incorrect_confidence:.4f}"
    )

    output = {
        "best_validation_model": best_model,
        "overall_metrics": overall_metrics,
        "per_phase_report": report,
        "confusion_matrix": (
            raw_matrix.astype(int).tolist()
        ),
        "normalized_confusion_matrix": (
            normalized_matrix.tolist()
        ),
        "prediction_distribution": {
            phase: int(
                prediction_distribution[phase]
            )
            for phase in PHASES
        },
        "participant_performance": (
            participant_summary
        ),
        "performance_by_modality_count": (
            modality_count_summary
        ),
        "performance_by_study_interval": (
            study_interval_summary
        ),
        "confidence_table": confidence_table,
        "correct_prediction_mean_confidence": (
            correct_confidence
        ),
        "incorrect_prediction_mean_confidence": (
            incorrect_confidence
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 76)

    print(f"Safe aggregate report:\n{OUTPUT_PATH}")
    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()