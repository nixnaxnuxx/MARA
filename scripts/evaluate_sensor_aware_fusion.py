from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

FEATURE_TABLE_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_temporal_features.parquet"
)

EXPERT_REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint6a_modality_experts.json"
)

EXPERT_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint6a_expert_validation_predictions.csv"
)

OUTPUT_REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint6b_sensor_aware_fusion.json"
)

OUTPUT_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint6b_fusion_validation_predictions.csv"
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

PROBABILITY_COLUMNS = [
    f"probability_{phase.lower()}"
    for phase in PHASES
]

NO_CALL = "NO_CALL"


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
    pd.DataFrame,
    dict[str, Any],
]:
    required_paths = [
        FEATURE_TABLE_PATH,
        EXPERT_REPORT_PATH,
        EXPERT_PREDICTIONS_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    features = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    predictions = pd.read_csv(
        EXPERT_PREDICTIONS_PATH
    )

    with EXPERT_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        expert_report = json.load(file)

    return features, predictions, expert_report


def attach_expert(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    expert_name: str,
    prefix: str,
) -> pd.DataFrame:
    subset = predictions[
        predictions["expert"] == expert_name
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No predictions found for {expert_name}."
        )

    duplicate_count = int(
        subset.duplicated(
            subset=KEYS,
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{expert_name} contains duplicated keys."
        )

    selected_columns = (
        KEYS
        + PROBABILITY_COLUMNS
    )

    subset = subset[selected_columns]

    rename_map = {
        column: f"{prefix}__{column}"
        for column in PROBABILITY_COLUMNS
    }

    subset = subset.rename(
        columns=rename_map
    )

    return base.merge(
        subset,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )


def get_probability_matrix(
    dataframe: pd.DataFrame,
    prefix: str,
) -> np.ndarray:
    columns = [
        f"{prefix}__{column}"
        for column in PROBABILITY_COLUMNS
    ]

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Missing probability columns: {missing}"
        )

    matrix = dataframe[
        columns
    ].to_numpy(dtype=float)

    row_sums = matrix.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{prefix} probabilities do not sum to one."
        )

    return matrix


def probabilities_to_predictions(
    probability_matrix: np.ndarray,
) -> np.ndarray:
    phase_array = np.asarray(
        PHASES,
        dtype=object,
    )

    return phase_array[
        probability_matrix.argmax(axis=1)
    ]


def calculate_metrics(
    truth: pd.Series,
    prediction: pd.Series,
) -> dict[str, Any]:
    represented_phases = [
        phase
        for phase in PHASES
        if phase in set(truth)
    ]

    return {
        "rows": int(len(truth)),
        "represented_phases": int(
            len(represented_phases)
        ),
        "accuracy": float(
            accuracy_score(
                truth,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                truth,
                prediction,
                labels=represented_phases,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=represented_phases,
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_method(
    dataframe: pd.DataFrame,
    prediction_column: str,
    confidence_column: str,
) -> dict[str, Any]:
    called = (
        dataframe[prediction_column]
        != NO_CALL
    )

    called_data = dataframe[
        called
    ].copy()

    coverage = float(
        called.mean()
    )

    result: dict[str, Any] = {
        "total_rows": int(len(dataframe)),
        "called_rows": int(called.sum()),
        "no_call_rows": int((~called).sum()),
        "coverage": coverage,
        "no_call_rate": float(
            1.0 - coverage
        ),
    }

    if called_data.empty:
        result["overall"] = None
        result["by_interval"] = {}
        result["interval_gap"] = None
        result["mean_confidence"] = None
        return result

    result["overall"] = calculate_metrics(
        called_data["phase"],
        called_data[prediction_column],
    )

    result["mean_confidence"] = float(
        called_data[
            confidence_column
        ].mean()
    )

    interval_results: dict[str, Any] = {}

    for interval, subset in (
        called_data.groupby(
            "study_interval"
        )
    ):
        interval_results[str(interval)] = (
            calculate_metrics(
                subset["phase"],
                subset[prediction_column],
            )
        )

    result["by_interval"] = interval_results

    interval_balanced = [
        values["balanced_accuracy"]
        for values in interval_results.values()
    ]

    if len(interval_balanced) >= 2:
        result["interval_gap"] = float(
            max(interval_balanced)
            - min(interval_balanced)
        )
    else:
        result["interval_gap"] = None

    return result


def create_router_probabilities(
    dataframe: pd.DataFrame,
    both_sensor_probabilities: np.ndarray,
    rhr_probabilities: np.ndarray,
    temperature_probabilities: np.ndarray,
) -> np.ndarray:
    output = np.full(
        (
            len(dataframe),
            len(PHASES),
        ),
        np.nan,
        dtype=float,
    )

    has_rhr = (
        dataframe["has_rhr"]
        .astype(bool)
        .to_numpy()
    )

    has_temperature = (
        dataframe["has_temperature"]
        .astype(bool)
        .to_numpy()
    )

    both = (
        has_rhr
        & has_temperature
    )

    rhr_only = (
        has_rhr
        & ~has_temperature
    )

    temperature_only = (
        ~has_rhr
        & has_temperature
    )

    output[both] = (
        both_sensor_probabilities[both]
    )

    output[rhr_only] = (
        rhr_probabilities[rhr_only]
    )

    output[temperature_only] = (
        temperature_probabilities[
            temperature_only
        ]
    )

    return output


def add_probability_method(
    dataframe: pd.DataFrame,
    method_name: str,
    probability_matrix: np.ndarray,
) -> tuple[str, str]:
    prediction_column = (
        f"prediction__{method_name}"
    )

    confidence_column = (
        f"confidence__{method_name}"
    )

    predictions = np.full(
        len(dataframe),
        NO_CALL,
        dtype=object,
    )

    confidence = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    called = ~np.isnan(
        probability_matrix
    ).all(axis=1)

    predictions[called] = (
        probabilities_to_predictions(
            probability_matrix[called]
        )
    )

    confidence[called] = (
        probability_matrix[
            called
        ].max(axis=1)
    )

    dataframe[
        prediction_column
    ] = predictions

    dataframe[
        confidence_column
    ] = confidence

    return (
        prediction_column,
        confidence_column,
    )


def print_method_table(
    results: dict[str, Any],
) -> None:
    print("\n" + "=" * 126)
    print("FUSION AND ROUTING COMPARISON")
    print("=" * 126)

    header = (
        f"{'Method':42s}"
        f"{'Coverage':>10s}"
        f"{'Bal Acc':>10s}"
        f"{'Macro F1':>10s}"
        f"{'2022 Bal':>11s}"
        f"{'2024 Bal':>11s}"
        f"{'Gap':>9s}"
        f"{'Confidence':>12s}"
    )

    print(header)
    print("-" * len(header))

    for method_name, result in (
        results.items()
    ):
        overall = result["overall"]

        if overall is None:
            print(
                f"{method_name:42s}"
                f"{result['coverage']:10.4f}"
                f"{'N/A':>10s}"
            )
            continue

        interval_2022 = (
            result["by_interval"]
            .get("2022", {})
            .get(
                "balanced_accuracy",
                float("nan"),
            )
        )

        interval_2024 = (
            result["by_interval"]
            .get("2024", {})
            .get(
                "balanced_accuracy",
                float("nan"),
            )
        )

        gap = result["interval_gap"]

        print(
            f"{method_name:42s}"
            f"{result['coverage']:10.4f}"
            f"{overall['balanced_accuracy']:10.4f}"
            f"{overall['macro_f1']:10.4f}"
            f"{interval_2022:11.4f}"
            f"{interval_2024:11.4f}"
            f"{gap:9.4f}"
            f"{result['mean_confidence']:12.4f}"
        )


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE SENSOR-AWARE FUSION")
    print("=" * 76)

    (
        features,
        expert_predictions,
        expert_report,
    ) = load_inputs()

    validation = features[
        features["split"] == "validation"
    ][
        KEYS
        + [
            "phase",
            "has_rhr",
            "has_temperature",
        ]
    ].copy()

    dataframe = validation.copy()

    expert_definitions = [
        (
            "RHR expert",
            "rhr",
        ),
        (
            "Temperature expert",
            "temperature",
        ),
        (
            "Stable RHR + temperature core",
            "core",
        ),
        (
            "Full passive reference",
            "full",
        ),
    ]

    for expert_name, prefix in (
        expert_definitions
    ):
        dataframe = attach_expert(
            dataframe,
            expert_predictions,
            expert_name,
            prefix,
        )

    rhr_probabilities = (
        get_probability_matrix(
            dataframe,
            "rhr",
        )
    )

    temperature_probabilities = (
        get_probability_matrix(
            dataframe,
            "temperature",
        )
    )

    core_probabilities = (
        get_probability_matrix(
            dataframe,
            "core",
        )
    )

    full_probabilities = (
        get_probability_matrix(
            dataframe,
            "full",
        )
    )

    rhr_cv = expert_report[
        "experts"
    ][
        "RHR expert"
    ][
        "training_grouped_cv"
    ]

    temperature_cv = expert_report[
        "experts"
    ][
        "Temperature expert"
    ][
        "training_grouped_cv"
    ]

    rhr_reliability = max(
        (
            rhr_cv[
                "mean_balanced_accuracy"
            ]
            - 0.25
        )
        / (
            1.0
            + rhr_cv[
                "std_balanced_accuracy"
            ]
        ),
        0.000001,
    )

    temperature_reliability = max(
        (
            temperature_cv[
                "mean_balanced_accuracy"
            ]
            - 0.25
        )
        / (
            1.0
            + temperature_cv[
                "std_balanced_accuracy"
            ]
        ),
        0.000001,
    )

    reliability_total = (
        rhr_reliability
        + temperature_reliability
    )

    rhr_weight = (
        rhr_reliability
        / reliability_total
    )

    temperature_weight = (
        temperature_reliability
        / reliability_total
    )

    print("\nFusion weights derived from training CV:")

    print(
        f"  RHR weight: "
        f"{rhr_weight:.4f}"
    )

    print(
        f"  Temperature weight: "
        f"{temperature_weight:.4f}"
    )

    equal_fusion = (
        0.5 * rhr_probabilities
        + 0.5 * temperature_probabilities
    )

    reliability_fusion = (
        rhr_weight * rhr_probabilities
        + temperature_weight
        * temperature_probabilities
    )

    router_core = (
        create_router_probabilities(
            dataframe=dataframe,
            both_sensor_probabilities=(
                core_probabilities
            ),
            rhr_probabilities=(
                rhr_probabilities
            ),
            temperature_probabilities=(
                temperature_probabilities
            ),
        )
    )

    router_fusion = (
        create_router_probabilities(
            dataframe=dataframe,
            both_sensor_probabilities=(
                reliability_fusion
            ),
            rhr_probabilities=(
                rhr_probabilities
            ),
            temperature_probabilities=(
                temperature_probabilities
            ),
        )
    )

    probability_methods = {
        "RHR expert": rhr_probabilities,
        "Temperature expert": (
            temperature_probabilities
        ),
        "Stable core": core_probabilities,
        "Full passive reference": (
            full_probabilities
        ),
        "Equal expert fusion": equal_fusion,
        "CV-reliability fusion": (
            reliability_fusion
        ),
        "Sensor router — core when both": (
            router_core
        ),
        "Sensor router — fusion when both": (
            router_fusion
        ),
    }

    method_columns: dict[
        str,
        tuple[str, str],
    ] = {}

    results: dict[str, Any] = {}

    for method_name, probability_matrix in (
        probability_methods.items()
    ):
        safe_name = (
            method_name
            .lower()
            .replace(" ", "_")
            .replace("—", "")
            .replace("+", "plus")
        )

        (
            prediction_column,
            confidence_column,
        ) = add_probability_method(
            dataframe,
            safe_name,
            probability_matrix,
        )

        method_columns[
            method_name
        ] = (
            prediction_column,
            confidence_column,
        )

        results[method_name] = (
            evaluate_method(
                dataframe,
                prediction_column,
                confidence_column,
            )
        )

    print_method_table(results)

    has_rhr = (
        dataframe["has_rhr"]
        .astype(bool)
    )

    has_temperature = (
        dataframe["has_temperature"]
        .astype(bool)
    )

    availability_masks = {
        "Both RHR and temperature": (
            has_rhr
            & has_temperature
        ),
        "RHR only": (
            has_rhr
            & ~has_temperature
        ),
        "Temperature only": (
            ~has_rhr
            & has_temperature
        ),
        "Neither": (
            ~has_rhr
            & ~has_temperature
        ),
    }

    availability_counts = {
        name: int(mask.sum())
        for name, mask
        in availability_masks.items()
    }

    print("\nNatural sensor availability:")

    for name, count in (
        availability_counts.items()
    ):
        percentage = (
            100.0 * count / len(dataframe)
        )

        print(
            f"  {name:28s}: "
            f"{count:4d} ({percentage:6.2f}%)"
        )

    both_mask = availability_masks[
        "Both RHR and temperature"
    ]

    both_data = dataframe[
        both_mask
    ].copy()

    dropout_methods = {
        "Both sensors — stable core": (
            method_columns[
                "Stable core"
            ]
        ),
        "Both sensors — reliability fusion": (
            method_columns[
                "CV-reliability fusion"
            ]
        ),
        "Temperature removed — RHR expert": (
            method_columns[
                "RHR expert"
            ]
        ),
        "RHR removed — temperature expert": (
            method_columns[
                "Temperature expert"
            ]
        ),
    }

    dropout_results: dict[str, Any] = {}

    print("\n" + "=" * 100)
    print("CONTROLLED SENSOR-DROPOUT TEST")
    print("=" * 100)

    for method_name, (
        prediction_column,
        confidence_column,
    ) in dropout_methods.items():
        result = evaluate_method(
            both_data,
            prediction_column,
            confidence_column,
        )

        dropout_results[
            method_name
        ] = result

        overall = result["overall"]

        print(
            f"{method_name:42s} "
            f"n={overall['rows']:4d} "
            f"balanced={overall['balanced_accuracy']:.4f} "
            f"macro_f1={overall['macro_f1']:.4f} "
            f"confidence={result['mean_confidence']:.4f}"
        )

    rhr_predictions = (
        probabilities_to_predictions(
            rhr_probabilities
        )
    )

    temperature_predictions = (
        probabilities_to_predictions(
            temperature_probabilities
        )
    )

    weighted_predictions = (
        probabilities_to_predictions(
            reliability_fusion
        )
    )

    dataframe["rhr_expert_prediction"] = (
        rhr_predictions
    )

    dataframe[
        "temperature_expert_prediction"
    ] = temperature_predictions

    dataframe[
        "reliability_fusion_prediction"
    ] = weighted_predictions

    dataframe["experts_agree"] = (
        dataframe[
            "rhr_expert_prediction"
        ]
        == dataframe[
            "temperature_expert_prediction"
        ]
    )

    agreement_results: dict[str, Any] = {}

    print("\n" + "=" * 100)
    print("EXPERT AGREEMENT ANALYSIS")
    print("=" * 100)

    for label, mask in {
        "Experts agree": dataframe[
            "experts_agree"
        ],
        "Experts disagree": ~dataframe[
            "experts_agree"
        ],
    }.items():
        subset = dataframe[mask]

        metrics = calculate_metrics(
            subset["phase"],
            subset[
                "reliability_fusion_prediction"
            ],
        )

        agreement_results[label] = metrics

        print(
            f"{label:18s}: "
            f"n={len(subset):4d}, "
            f"balanced={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    disagreement = dataframe[
        ~dataframe["experts_agree"]
    ].copy()

    if disagreement.empty:
        truth_in_expert_union = None
    else:
        truth_in_expert_union = float(
            (
                (
                    disagreement["phase"]
                    == disagreement[
                        "rhr_expert_prediction"
                    ]
                )
                |
                (
                    disagreement["phase"]
                    == disagreement[
                        "temperature_expert_prediction"
                    ]
                )
            ).mean()
        )

    print(
        "\nWhen experts disagree, true phase is "
        "within their two proposed phases: "
        f"{truth_in_expert_union:.4f}"
    )

    output = {
        "test_evaluated": False,
        "validation_rows": int(
            len(dataframe)
        ),
        "fusion_weights": {
            "rhr": float(rhr_weight),
            "temperature": float(
                temperature_weight
            ),
            "weight_source": (
                "Training participant-grouped CV "
                "balanced accuracy above random, "
                "penalized by fold variability."
            ),
        },
        "method_results": results,
        "natural_sensor_availability": (
            availability_counts
        ),
        "controlled_dropout_results": (
            dropout_results
        ),
        "expert_agreement_results": (
            agreement_results
        ),
        "truth_in_expert_union_when_disagree": (
            truth_in_expert_union
        ),
    }

    with OUTPUT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            default=json_safe,
        )

    dataframe.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        f"\nSafe aggregate report:\n"
        f"{OUTPUT_REPORT_PATH}"
    )

    print(
        "\nRestricted validation predictions:\n"
        f"{OUTPUT_PREDICTIONS_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()