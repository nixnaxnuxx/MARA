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

EXPERT_BUNDLE_PATH = (
    ARTIFACTS_DIR
    / "modality_expert_models.joblib"
)

CALIBRATION_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "uncertainty_calibration.json"
)

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint7a_uncertainty_calibration.json"
)

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint7a_validation_prediction_sets.csv"
)

RANDOM_SEED = 20260719
CV_SPLITS = 4

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PHASE_TO_INDEX = {
    phase: index
    for index, phase in enumerate(PHASES)
}

NOMINAL_COVERAGES = [
    0.80,
    0.85,
    0.90,
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
]:
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE_PATH}"
        )

    if not EXPERT_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Expert bundle not found: {EXPERT_BUNDLE_PATH}"
        )

    dataframe = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    bundle = joblib.load(
        EXPERT_BUNDLE_PATH
    )

    required_experts = [
        "RHR expert",
        "Temperature expert",
    ]

    missing_experts = [
        expert
        for expert in required_experts
        if expert not in bundle["models"]
    ]

    if missing_experts:
        raise KeyError(
            f"Missing experts from bundle: {missing_experts}"
        )

    return dataframe, bundle


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


def predict_aligned(
    model: Pipeline,
    dataframe: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    probabilities = model.predict_proba(
        dataframe[features]
    )

    return align_probabilities(
        probabilities,
        list(model.classes_),
    )


def create_equal_sensor_router(
    dataframe: pd.DataFrame,
    rhr_probabilities: np.ndarray,
    temperature_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    routed = np.full(
        (
            len(dataframe),
            len(PHASES),
        ),
        np.nan,
        dtype=float,
    )

    routes = np.full(
        len(dataframe),
        "Neither sensor",
        dtype=object,
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

    routed[both] = (
        0.5 * rhr_probabilities[both]
        + 0.5 * temperature_probabilities[both]
    )

    routed[rhr_only] = (
        rhr_probabilities[rhr_only]
    )

    routed[temperature_only] = (
        temperature_probabilities[
            temperature_only
        ]
    )

    routes[both] = "Equal fusion"
    routes[rhr_only] = "RHR expert"
    routes[
        temperature_only
    ] = "Temperature expert"

    return routed, routes


def create_oof_predictions(
    train: pd.DataFrame,
    rhr_features: list[str],
    temperature_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupKFold(
        n_splits=CV_SPLITS
    )

    routed_probabilities = np.full(
        (
            len(train),
            len(PHASES),
        ),
        np.nan,
        dtype=float,
    )

    routes = np.full(
        len(train),
        "Unassigned",
        dtype=object,
    )

    for fold_number, (
        fit_indices,
        evaluation_indices,
    ) in enumerate(
        splitter.split(
            train,
            train["phase"],
            groups=train["id"],
        ),
        start=1,
    ):
        print(
            f"  Building OOF fold "
            f"{fold_number}/{CV_SPLITS}..."
        )

        fit_data = train.iloc[
            fit_indices
        ]

        evaluation_data = train.iloc[
            evaluation_indices
        ]

        rhr_model = make_model()
        temperature_model = make_model()

        rhr_model.fit(
            fit_data[rhr_features],
            fit_data["phase"],
        )

        temperature_model.fit(
            fit_data[temperature_features],
            fit_data["phase"],
        )

        rhr_probabilities = predict_aligned(
            rhr_model,
            evaluation_data,
            rhr_features,
        )

        temperature_probabilities = (
            predict_aligned(
                temperature_model,
                evaluation_data,
                temperature_features,
            )
        )

        fold_probabilities, fold_routes = (
            create_equal_sensor_router(
                evaluation_data,
                rhr_probabilities,
                temperature_probabilities,
            )
        )

        routed_probabilities[
            evaluation_indices
        ] = fold_probabilities

        routes[
            evaluation_indices
        ] = fold_routes

    return routed_probabilities, routes


def apply_temperature_scaling(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    transformed = np.power(
        clipped,
        1.0 / temperature,
    )

    return (
        transformed
        / transformed.sum(
            axis=1,
            keepdims=True,
        )
    )


def negative_log_likelihood(
    probabilities: np.ndarray,
    truth: pd.Series,
) -> float:
    target_indices = np.array(
        [
            PHASE_TO_INDEX[value]
            for value in truth
        ],
        dtype=int,
    )

    true_probabilities = probabilities[
        np.arange(len(probabilities)),
        target_indices,
    ]

    return float(
        -np.log(
            np.clip(
                true_probabilities,
                1e-12,
                1.0,
            )
        ).mean()
    )


def fit_temperature(
    probabilities: np.ndarray,
    truth: pd.Series,
) -> tuple[float, float]:
    candidates = np.exp(
        np.linspace(
            np.log(0.25),
            np.log(5.0),
            800,
        )
    )

    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in candidates:
        scaled = apply_temperature_scaling(
            probabilities,
            float(temperature),
        )

        loss = negative_log_likelihood(
            scaled,
            truth,
        )

        if loss < best_loss:
            best_loss = loss
            best_temperature = float(
                temperature
            )

    return best_temperature, best_loss


def expected_calibration_error(
    truth: pd.Series,
    probabilities: np.ndarray,
    bin_count: int = 10,
) -> float:
    predictions = probabilities.argmax(
        axis=1
    )

    target_indices = np.array(
        [
            PHASE_TO_INDEX[value]
            for value in truth
        ]
    )

    correct = (
        predictions == target_indices
    )

    confidence = probabilities.max(
        axis=1
    )

    boundaries = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
    )

    error = 0.0

    for index in range(bin_count):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        if index == bin_count - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if not mask.any():
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()

        error += (
            mask.mean()
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(error)


def multiclass_brier_score(
    truth: pd.Series,
    probabilities: np.ndarray,
) -> float:
    target = np.zeros_like(
        probabilities
    )

    target_indices = np.array(
        [
            PHASE_TO_INDEX[value]
            for value in truth
        ]
    )

    target[
        np.arange(len(target)),
        target_indices,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - target
                )
                ** 2,
                axis=1,
            )
        )
    )


def evaluate_probabilities(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    called = np.isfinite(
        probabilities
    ).all(axis=1)

    called_data = dataframe.loc[
        called
    ]

    called_probabilities = (
        probabilities[called]
    )

    predictions = np.asarray(
        PHASES,
        dtype=object,
    )[
        called_probabilities.argmax(
            axis=1
        )
    ]

    return {
        "total_rows": int(
            len(dataframe)
        ),
        "called_rows": int(
            called.sum()
        ),
        "coverage": float(
            called.mean()
        ),
        "accuracy": float(
            accuracy_score(
                called_data["phase"],
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                called_data["phase"],
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                called_data["phase"],
                predictions,
                labels=PHASES,
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": (
            negative_log_likelihood(
                called_probabilities,
                called_data["phase"],
            )
        ),
        "brier_score": (
            multiclass_brier_score(
                called_data["phase"],
                called_probabilities,
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                called_data["phase"],
                called_probabilities,
            )
        ),
        "mean_confidence": float(
            called_probabilities.max(
                axis=1
            ).mean()
        ),
    }


def calculate_aps_scores(
    probabilities: np.ndarray,
    truth: pd.Series,
) -> np.ndarray:
    target_indices = np.array(
        [
            PHASE_TO_INDEX[value]
            for value in truth
        ],
        dtype=int,
    )

    order = np.argsort(
        -probabilities,
        axis=1,
    )

    sorted_probabilities = (
        np.take_along_axis(
            probabilities,
            order,
            axis=1,
        )
    )

    cumulative = np.cumsum(
        sorted_probabilities,
        axis=1,
    )

    target_positions = np.argmax(
        order
        == target_indices[:, None],
        axis=1,
    )

    return cumulative[
        np.arange(len(probabilities)),
        target_positions,
    ]


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    sample_count = len(scores)

    quantile_level = min(
        1.0,
        np.ceil(
            (sample_count + 1)
            * (1.0 - alpha)
        )
        / sample_count,
    )

    try:
        return float(
            np.quantile(
                scores,
                quantile_level,
                method="higher",
            )
        )
    except TypeError:
        return float(
            np.quantile(
                scores,
                quantile_level,
                interpolation="higher",
            )
        )


def create_prediction_sets(
    probabilities: np.ndarray,
    qhat: float,
) -> list[list[str]]:
    output: list[list[str]] = []

    for row in probabilities:
        if not np.isfinite(row).all():
            output.append([])
            continue

        order = np.argsort(-row)
        cumulative = np.cumsum(
            row[order]
        )

        reaching = np.where(
            cumulative >= qhat
        )[0]

        if len(reaching) == 0:
            final_position = len(PHASES) - 1
        else:
            final_position = int(
                reaching[0]
            )

        selected_indices = order[
            : final_position + 1
        ]

        output.append(
            [
                PHASES[index]
                for index in selected_indices
            ]
        )

    return output


def evaluate_prediction_sets(
    dataframe: pd.DataFrame,
    prediction_sets: list[list[str]],
) -> dict[str, Any]:
    set_sizes = np.array(
        [
            len(values)
            for values in prediction_sets
        ],
        dtype=int,
    )

    sensor_called = set_sizes > 0

    contains_truth = np.array(
        [
            (
                dataframe.iloc[index]["phase"]
                in prediction_sets[index]
            )
            if set_sizes[index] > 0
            else False
            for index in range(len(dataframe))
        ],
        dtype=bool,
    )

    singleton = set_sizes == 1
    two_phase = set_sizes == 2
    large_set = set_sizes >= 3

    decision_mask = (
        singleton
        | two_phase
    )

    result: dict[str, Any] = {
        "total_rows": int(
            len(dataframe)
        ),
        "sensor_called_rows": int(
            sensor_called.sum()
        ),
        "sensor_coverage": float(
            sensor_called.mean()
        ),
        "empirical_set_coverage": float(
            contains_truth[
                sensor_called
            ].mean()
        ),
        "average_set_size": float(
            set_sizes[
                sensor_called
            ].mean()
        ),
        "singleton_rows": int(
            singleton.sum()
        ),
        "singleton_rate_among_sensor_called": float(
            singleton[
                sensor_called
            ].mean()
        ),
        "two_phase_rows": int(
            two_phase.sum()
        ),
        "two_phase_rate_among_sensor_called": float(
            two_phase[
                sensor_called
            ].mean()
        ),
        "large_set_rows": int(
            large_set.sum()
        ),
        "large_set_rate_among_sensor_called": float(
            large_set[
                sensor_called
            ].mean()
        ),
        "decision_rows": int(
            decision_mask.sum()
        ),
        "decision_coverage": float(
            decision_mask.mean()
        ),
        "policy_no_call_rate": float(
            1.0
            - decision_mask.mean()
        ),
        "decision_set_coverage": float(
            contains_truth[
                decision_mask
            ].mean()
        ),
    }

    if singleton.any():
        result["singleton_accuracy"] = float(
            contains_truth[
                singleton
            ].mean()
        )
    else:
        result["singleton_accuracy"] = None

    if two_phase.any():
        result["two_phase_set_coverage"] = float(
            contains_truth[
                two_phase
            ].mean()
        )
    else:
        result[
            "two_phase_set_coverage"
        ] = None

    interval_results = {}

    for interval, indices in dataframe.groupby(
        "study_interval"
    ).groups.items():
        indices = np.asarray(
            list(indices),
            dtype=int,
        )

        interval_called = (
            sensor_called[indices]
        )

        interval_truth_covered = (
            contains_truth[indices]
        )

        interval_sizes = (
            set_sizes[indices]
        )

        interval_decisions = (
            (
                interval_sizes == 1
            )
            | (
                interval_sizes == 2
            )
        )

        interval_results[str(interval)] = {
            "rows": int(
                len(indices)
            ),
            "sensor_coverage": float(
                interval_called.mean()
            ),
            "empirical_set_coverage": float(
                interval_truth_covered[
                    interval_called
                ].mean()
            ),
            "average_set_size": float(
                interval_sizes[
                    interval_called
                ].mean()
            ),
            "decision_coverage": float(
                interval_decisions.mean()
            ),
        }

    result["by_interval"] = (
        interval_results
    )

    return result


def print_probability_metrics(
    name: str,
    metrics: dict[str, Any],
) -> None:
    print(
        f"{name:28s} "
        f"coverage={metrics['coverage']:.4f} "
        f"balanced={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"NLL={metrics['negative_log_likelihood']:.4f} "
        f"Brier={metrics['brier_score']:.4f} "
        f"ECE={metrics['expected_calibration_error']:.4f} "
        f"confidence={metrics['mean_confidence']:.4f}"
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
    print("MOSAIC-PHASE UNCERTAINTY CALIBRATION")
    print("=" * 76)

    dataframe, bundle = load_inputs()

    train = dataframe[
        dataframe["split"] == "train"
    ].copy().reset_index(drop=True)

    validation = dataframe[
        dataframe["split"] == "validation"
    ].copy().reset_index(drop=True)

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

    rhr_entry = bundle[
        "models"
    ]["RHR expert"]

    temperature_entry = bundle[
        "models"
    ]["Temperature expert"]

    rhr_features = rhr_entry[
        "features"
    ]

    temperature_features = (
        temperature_entry["features"]
    )

    print("\nGenerating participant-grouped OOF predictions...")

    (
        oof_probabilities,
        oof_routes,
    ) = create_oof_predictions(
        train,
        rhr_features,
        temperature_features,
    )

    oof_called = np.isfinite(
        oof_probabilities
    ).all(axis=1)

    print(
        "OOF rows available for calibration: "
        f"{oof_called.sum():,} / {len(train):,}"
    )

    calibration_temperature, oof_nll = (
        fit_temperature(
            oof_probabilities[
                oof_called
            ],
            train.loc[
                oof_called,
                "phase",
            ],
        )
    )

    print(
        "\nFitted probability temperature: "
        f"{calibration_temperature:.4f}"
    )

    print(
        "Calibrated OOF negative log-likelihood: "
        f"{oof_nll:.4f}"
    )

    calibrated_oof = np.full_like(
        oof_probabilities,
        np.nan,
    )

    calibrated_oof[
        oof_called
    ] = apply_temperature_scaling(
        oof_probabilities[
            oof_called
        ],
        calibration_temperature,
    )

    rhr_validation_probabilities = (
        predict_aligned(
            rhr_entry["model"],
            validation,
            rhr_features,
        )
    )

    temperature_validation_probabilities = (
        predict_aligned(
            temperature_entry["model"],
            validation,
            temperature_features,
        )
    )

    (
        validation_probabilities,
        validation_routes,
    ) = create_equal_sensor_router(
        validation,
        rhr_validation_probabilities,
        temperature_validation_probabilities,
    )

    validation_called = np.isfinite(
        validation_probabilities
    ).all(axis=1)

    calibrated_validation = np.full_like(
        validation_probabilities,
        np.nan,
    )

    calibrated_validation[
        validation_called
    ] = apply_temperature_scaling(
        validation_probabilities[
            validation_called
        ],
        calibration_temperature,
    )

    oof_raw_metrics = evaluate_probabilities(
        train,
        oof_probabilities,
    )

    oof_calibrated_metrics = (
        evaluate_probabilities(
            train,
            calibrated_oof,
        )
    )

    validation_raw_metrics = (
        evaluate_probabilities(
            validation,
            validation_probabilities,
        )
    )

    validation_calibrated_metrics = (
        evaluate_probabilities(
            validation,
            calibrated_validation,
        )
    )

    print("\n" + "=" * 76)
    print("PROBABILITY CALIBRATION RESULTS")
    print("=" * 76)

    print_probability_metrics(
        "Training OOF — raw",
        oof_raw_metrics,
    )

    print_probability_metrics(
        "Training OOF — calibrated",
        oof_calibrated_metrics,
    )

    print_probability_metrics(
        "Validation — raw router",
        validation_raw_metrics,
    )

    print_probability_metrics(
        "Validation — calibrated",
        validation_calibrated_metrics,
    )

    oof_scores = calculate_aps_scores(
        calibrated_oof[
            oof_called
        ],
        train.loc[
            oof_called,
            "phase",
        ],
    )

    prediction_set_results = {}
    conformal_thresholds = {}

    validation_output = validation[
        [
            "id",
            "study_interval",
            "day_in_study",
            "phase",
            "has_rhr",
            "has_temperature",
        ]
    ].copy()

    validation_output["sensor_route"] = (
        validation_routes
    )

    validation_output[
        "raw_prediction"
    ] = np.where(
        validation_called,
        np.asarray(PHASES)[
            np.nanargmax(
                np.where(
                    np.isnan(
                        validation_probabilities
                    ),
                    -np.inf,
                    validation_probabilities,
                ),
                axis=1,
            )
        ],
        "NO_CALL",
    )

    validation_output[
        "calibrated_prediction"
    ] = np.where(
        validation_called,
        np.asarray(PHASES)[
            np.nanargmax(
                np.where(
                    np.isnan(
                        calibrated_validation
                    ),
                    -np.inf,
                    calibrated_validation,
                ),
                axis=1,
            )
        ],
        "NO_CALL",
    )

    validation_output[
        "calibrated_confidence"
    ] = np.where(
        validation_called,
        np.nanmax(
            calibrated_validation,
            axis=1,
        ),
        np.nan,
    )

    for phase_index, phase in enumerate(PHASES):
        validation_output[
            f"calibrated_probability_{phase.lower()}"
        ] = calibrated_validation[
            :,
            phase_index,
        ]

    print("\n" + "=" * 106)
    print("ADAPTIVE PREDICTION-SET RESULTS")
    print("=" * 106)

    header = (
        f"{'Nominal':>10s}"
        f"{'Q-hat':>10s}"
        f"{'Coverage':>11s}"
        f"{'Avg size':>11s}"
        f"{'Singleton':>12s}"
        f"{'Two phase':>12s}"
        f"{'Large set':>12s}"
        f"{'Decision':>11s}"
        f"{'No-call':>10s}"
    )

    print(header)
    print("-" * len(header))

    for nominal_coverage in (
        NOMINAL_COVERAGES
    ):
        alpha = (
            1.0
            - nominal_coverage
        )

        qhat = conformal_quantile(
            oof_scores,
            alpha,
        )

        prediction_sets = (
            create_prediction_sets(
                calibrated_validation,
                qhat,
            )
        )

        set_metrics = (
            evaluate_prediction_sets(
                validation,
                prediction_sets,
            )
        )

        key = (
            f"{int(nominal_coverage * 100)}"
        )

        conformal_thresholds[key] = qhat

        prediction_set_results[key] = {
            "nominal_coverage": (
                nominal_coverage
            ),
            "alpha": alpha,
            "qhat": qhat,
            **set_metrics,
        }

        print(
            f"{nominal_coverage:10.2f}"
            f"{qhat:10.4f}"
            f"{set_metrics['empirical_set_coverage']:11.4f}"
            f"{set_metrics['average_set_size']:11.4f}"
            f"{set_metrics['singleton_rate_among_sensor_called']:12.4f}"
            f"{set_metrics['two_phase_rate_among_sensor_called']:12.4f}"
            f"{set_metrics['large_set_rate_among_sensor_called']:12.4f}"
            f"{set_metrics['decision_coverage']:11.4f}"
            f"{set_metrics['policy_no_call_rate']:10.4f}"
        )

        validation_output[
            f"prediction_set_{key}"
        ] = [
            " | ".join(values)
            if values
            else "NO_CALL"
            for values in prediction_sets
        ]

        validation_output[
            f"prediction_set_size_{key}"
        ] = [
            len(values)
            for values in prediction_sets
        ]

    route_counts = (
        pd.Series(
            validation_routes
        )
        .value_counts()
        .to_dict()
    )

    rhr_predictions = np.asarray(
        PHASES
    )[
        rhr_validation_probabilities.argmax(
            axis=1
        )
    ]

    temperature_predictions = np.asarray(
        PHASES
    )[
        temperature_validation_probabilities.argmax(
            axis=1
        )
    ]

    experts_agree = (
        rhr_predictions
        == temperature_predictions
    )

    calibrated_predictions = (
        np.asarray(PHASES)[
            calibrated_validation[
                validation_called
            ].argmax(axis=1)
        ]
    )

    called_agreement = (
        experts_agree[
            validation_called
        ]
    )

    called_truth = validation.loc[
        validation_called,
        "phase",
    ]

    agreement_results = {}

    for label, mask in {
        "agree": called_agreement,
        "disagree": ~called_agreement,
    }.items():
        subset_truth = called_truth.loc[
            mask
        ]

        subset_prediction = (
            calibrated_predictions[mask]
        )

        agreement_results[label] = {
            "rows": int(
                mask.sum()
            ),
            "accuracy": float(
                accuracy_score(
                    subset_truth,
                    subset_prediction,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    subset_truth,
                    subset_prediction,
                )
            ),
            "macro_f1": float(
                f1_score(
                    subset_truth,
                    subset_prediction,
                    labels=PHASES,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    calibration_artifact = {
        "phase_order": PHASES,
        "router": {
            "both_sensors": (
                "equal average of RHR and "
                "temperature probabilities"
            ),
            "rhr_only": "RHR expert",
            "temperature_only": (
                "Temperature expert"
            ),
            "neither": "NO_CALL",
        },
        "temperature_scaling": float(
            calibration_temperature
        ),
        "conformal_method": (
            "Adaptive prediction sets using "
            "participant-grouped OOF calibration "
            "predictions."
        ),
        "conformal_thresholds": (
            conformal_thresholds
        ),
        "test_evaluated": False,
    }

    with CALIBRATION_ARTIFACT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration_artifact,
            file,
            indent=2,
            default=json_safe,
        )

    report = {
        "test_evaluated": False,
        "training_rows": int(
            len(train)
        ),
        "validation_rows": int(
            len(validation)
        ),
        "reserved_test_rows": int(
            len(test)
        ),
        "oof_called_rows": int(
            oof_called.sum()
        ),
        "calibration_temperature": float(
            calibration_temperature
        ),
        "probability_metrics": {
            "oof_raw": (
                oof_raw_metrics
            ),
            "oof_calibrated": (
                oof_calibrated_metrics
            ),
            "validation_raw": (
                validation_raw_metrics
            ),
            "validation_calibrated": (
                validation_calibrated_metrics
            ),
        },
        "validation_route_counts": {
            str(key): int(value)
            for key, value
            in route_counts.items()
        },
        "prediction_set_results": (
            prediction_set_results
        ),
        "expert_agreement_results": (
            agreement_results
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

    validation_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nValidation sensor routes:")

    for route, count in route_counts.items():
        print(
            f"  {route:20s}: {count}"
        )

    print("\nExpert agreement after routing:")

    for label, values in (
        agreement_results.items()
    ):
        print(
            f"  {label:10s}: "
            f"n={values['rows']}, "
            f"balanced={values['balanced_accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        f"\nCalibration artifact:\n"
        f"{CALIBRATION_ARTIFACT_PATH}"
    )

    print(
        f"\nSafe aggregate report:\n"
        f"{REPORT_PATH}"
    )

    print(
        "\nRestricted validation predictions:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()