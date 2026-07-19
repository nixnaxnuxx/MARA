from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from calibrate_uncertainty import (
    PHASES,
    PHASE_TO_INDEX,
    apply_temperature_scaling,
    create_equal_sensor_router,
    fit_temperature,
    json_safe,
    make_model,
    predict_aligned,
)


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

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint7c_robust_policy.json"
)

CANDIDATE_PATH = (
    ARTIFACTS_DIR
    / "robust_policy_candidates.json"
)

CV_SPLITS = 4

SINGLE_THRESHOLDS = np.round(
    np.arange(0.35, 0.81, 0.01),
    2,
)

PAIR_THRESHOLDS = np.round(
    np.arange(0.50, 0.96, 0.01),
    2,
)

MINIMUM_SINGLETON_ROWS = 20
MINIMUM_PAIR_ROWS = 100
MINIMUM_DECISIONS_PER_INTERVAL = 50

ROBUST_TIERS = {
    "Practical robust": {
        "minimum_singleton_accuracy": 0.50,
        "minimum_pair_coverage": 0.65,
        "minimum_overall_set_coverage": 0.65,
        "minimum_worst_interval_coverage": 0.62,
    },
    "Balanced robust": {
        "minimum_singleton_accuracy": 0.55,
        "minimum_pair_coverage": 0.68,
        "minimum_overall_set_coverage": 0.68,
        "minimum_worst_interval_coverage": 0.65,
    },
    "Conservative robust": {
        "minimum_singleton_accuracy": 0.60,
        "minimum_pair_coverage": 0.72,
        "minimum_overall_set_coverage": 0.72,
        "minimum_worst_interval_coverage": 0.68,
    },
}

PAIR_ONLY_REQUIREMENTS = {
    "minimum_pair_coverage": 0.65,
    "minimum_overall_set_coverage": 0.65,
    "minimum_worst_interval_coverage": 0.62,
}


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    for path in [
        FEATURE_TABLE_PATH,
        EXPERT_BUNDLE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    dataframe = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    bundle = joblib.load(
        EXPERT_BUNDLE_PATH
    )

    return dataframe, bundle


def create_oof_expert_outputs(
    train: pd.DataFrame,
    rhr_features: list[str],
    temperature_features: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    splitter = GroupKFold(
        n_splits=CV_SPLITS
    )

    rhr_output = np.full(
        (len(train), len(PHASES)),
        np.nan,
        dtype=float,
    )

    temperature_output = np.full(
        (len(train), len(PHASES)),
        np.nan,
        dtype=float,
    )

    router_output = np.full(
        (len(train), len(PHASES)),
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

        routed_probabilities, fold_routes = (
            create_equal_sensor_router(
                evaluation_data,
                rhr_probabilities,
                temperature_probabilities,
            )
        )

        rhr_output[
            evaluation_indices
        ] = rhr_probabilities

        temperature_output[
            evaluation_indices
        ] = temperature_probabilities

        router_output[
            evaluation_indices
        ] = routed_probabilities

        routes[
            evaluation_indices
        ] = fold_routes

    return (
        rhr_output,
        temperature_output,
        router_output,
        routes,
    )


def prepare_rank_data(
    dataframe: pd.DataFrame,
    router_probabilities: np.ndarray,
    rhr_probabilities: np.ndarray,
    temperature_probabilities: np.ndarray,
    routes: np.ndarray,
) -> dict[str, Any]:
    called = np.isfinite(
        router_probabilities
    ).all(axis=1)

    truth_indices = np.asarray(
        [
            PHASE_TO_INDEX[phase]
            for phase in dataframe["phase"]
        ],
        dtype=int,
    )

    maximum_probability = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    top_two_probability = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    top_one_correct = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    top_two_correct = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    called_probabilities = (
        router_probabilities[called]
    )

    order = np.argsort(
        -called_probabilities,
        axis=1,
    )

    sorted_probabilities = (
        np.take_along_axis(
            called_probabilities,
            order,
            axis=1,
        )
    )

    called_truth = truth_indices[called]

    maximum_probability[called] = (
        sorted_probabilities[:, 0]
    )

    top_two_probability[called] = (
        sorted_probabilities[:, :2]
        .sum(axis=1)
    )

    top_one_correct[called] = (
        order[:, 0] == called_truth
    )

    top_two_correct[called] = (
        (order[:, 0] == called_truth)
        | (order[:, 1] == called_truth)
    )

    rhr_predictions = (
        rhr_probabilities.argmax(axis=1)
    )

    temperature_predictions = (
        temperature_probabilities.argmax(
            axis=1
        )
    )

    experts_agree = (
        rhr_predictions
        == temperature_predictions
    )

    both_sensors = (
        dataframe["has_rhr"]
        .astype(bool)
        .to_numpy()
        & dataframe["has_temperature"]
        .astype(bool)
        .to_numpy()
    )

    return {
        "rows": len(dataframe),
        "called": called,
        "maximum_probability": (
            maximum_probability
        ),
        "top_two_probability": (
            top_two_probability
        ),
        "top_one_correct": (
            top_one_correct
        ),
        "top_two_correct": (
            top_two_correct
        ),
        "experts_agree": experts_agree,
        "both_sensors": both_sensors,
        "study_interval": (
            dataframe["study_interval"]
            .astype(str)
            .to_numpy()
        ),
        "route": np.asarray(
            routes,
            dtype=object,
        ),
    }


def safe_mean(
    values: np.ndarray,
) -> float | None:
    if len(values) == 0:
        return None

    return float(values.mean())


def evaluate_policy(
    prepared: dict[str, Any],
    pair_threshold: float,
    single_threshold: float | None,
    allow_singletons: bool,
    subset_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if subset_mask is None:
        subset_mask = np.ones(
            prepared["rows"],
            dtype=bool,
        )

    called = (
        prepared["called"]
        & subset_mask
    )

    if allow_singletons:
        if single_threshold is None:
            raise ValueError(
                "A single threshold is required."
            )

        singleton = (
            called
            & prepared["both_sensors"]
            & prepared["experts_agree"]
            & (
                prepared[
                    "maximum_probability"
                ]
                >= single_threshold
            )
        )
    else:
        singleton = np.zeros(
            prepared["rows"],
            dtype=bool,
        )

    pair = (
        called
        & ~singleton
        & (
            prepared[
                "top_two_probability"
            ]
            >= pair_threshold
        )
    )

    decision = singleton | pair

    singleton_correct = (
        singleton
        & prepared["top_one_correct"]
    )

    pair_correct = (
        pair
        & prepared["top_two_correct"]
    )

    decision_correct = (
        singleton_correct
        | pair_correct
    )

    subset_rows = int(
        subset_mask.sum()
    )

    called_rows = int(
        called.sum()
    )

    singleton_rows = int(
        singleton.sum()
    )

    pair_rows = int(
        pair.sum()
    )

    decision_rows = int(
        decision.sum()
    )

    if decision_rows:
        overall_set_coverage = float(
            decision_correct[
                decision
            ].mean()
        )

        average_set_size = float(
            (
                singleton_rows
                + 2 * pair_rows
            )
            / decision_rows
        )
    else:
        overall_set_coverage = None
        average_set_size = None

    return {
        "single_threshold": (
            float(single_threshold)
            if single_threshold is not None
            else None
        ),
        "pair_threshold": float(
            pair_threshold
        ),
        "subset_rows": subset_rows,
        "sensor_called_rows": called_rows,
        "sensor_coverage": float(
            called_rows / subset_rows
        ),
        "singleton_rows": singleton_rows,
        "singleton_accuracy": safe_mean(
            prepared["top_one_correct"][
                singleton
            ]
        ),
        "pair_rows": pair_rows,
        "pair_coverage": safe_mean(
            prepared["top_two_correct"][
                pair
            ]
        ),
        "decision_rows": decision_rows,
        "decision_coverage": float(
            decision_rows / subset_rows
        ),
        "overall_set_coverage": (
            overall_set_coverage
        ),
        "average_set_size": (
            average_set_size
        ),
        "policy_no_call_rate": float(
            1.0
            - decision_rows / subset_rows
        ),
    }


def evaluate_by_interval(
    prepared: dict[str, Any],
    pair_threshold: float,
    single_threshold: float | None,
    allow_singletons: bool,
) -> dict[str, Any]:
    output = {}

    for interval in sorted(
        set(prepared["study_interval"]),
        key=str,
    ):
        mask = (
            prepared["study_interval"]
            == interval
        )

        output[str(interval)] = (
            evaluate_policy(
                prepared=prepared,
                pair_threshold=pair_threshold,
                single_threshold=(
                    single_threshold
                ),
                allow_singletons=(
                    allow_singletons
                ),
                subset_mask=mask,
            )
        )

    return output


def attach_interval_robustness(
    result: dict[str, Any],
    interval_results: dict[str, Any],
) -> dict[str, Any]:
    result = dict(result)

    valid_coverages = [
        metrics[
            "overall_set_coverage"
        ]
        for metrics
        in interval_results.values()
        if metrics[
            "overall_set_coverage"
        ] is not None
    ]

    interval_decisions = [
        metrics["decision_rows"]
        for metrics
        in interval_results.values()
    ]

    result["by_interval"] = (
        interval_results
    )

    result[
        "worst_interval_set_coverage"
    ] = (
        float(min(valid_coverages))
        if valid_coverages
        else None
    )

    result[
        "minimum_interval_decision_rows"
    ] = (
        int(min(interval_decisions))
        if interval_decisions
        else 0
    )

    return result


def agreement_policy_eligible(
    result: dict[str, Any],
    requirements: dict[str, float],
) -> bool:
    required_values = [
        result["singleton_accuracy"],
        result["pair_coverage"],
        result["overall_set_coverage"],
        result[
            "worst_interval_set_coverage"
        ],
    ]

    if any(
        value is None
        for value in required_values
    ):
        return False

    return (
        result["singleton_rows"]
        >= MINIMUM_SINGLETON_ROWS
        and result["pair_rows"]
        >= MINIMUM_PAIR_ROWS
        and result[
            "minimum_interval_decision_rows"
        ]
        >= MINIMUM_DECISIONS_PER_INTERVAL
        and result["singleton_accuracy"]
        >= requirements[
            "minimum_singleton_accuracy"
        ]
        and result["pair_coverage"]
        >= requirements[
            "minimum_pair_coverage"
        ]
        and result["overall_set_coverage"]
        >= requirements[
            "minimum_overall_set_coverage"
        ]
        and result[
            "worst_interval_set_coverage"
        ]
        >= requirements[
            "minimum_worst_interval_coverage"
        ]
    )


def pair_only_eligible(
    result: dict[str, Any],
) -> bool:
    required_values = [
        result["pair_coverage"],
        result["overall_set_coverage"],
        result[
            "worst_interval_set_coverage"
        ],
    ]

    if any(
        value is None
        for value in required_values
    ):
        return False

    return (
        result["pair_rows"]
        >= MINIMUM_PAIR_ROWS
        and result[
            "minimum_interval_decision_rows"
        ]
        >= MINIMUM_DECISIONS_PER_INTERVAL
        and result["pair_coverage"]
        >= PAIR_ONLY_REQUIREMENTS[
            "minimum_pair_coverage"
        ]
        and result["overall_set_coverage"]
        >= PAIR_ONLY_REQUIREMENTS[
            "minimum_overall_set_coverage"
        ]
        and result[
            "worst_interval_set_coverage"
        ]
        >= PAIR_ONLY_REQUIREMENTS[
            "minimum_worst_interval_coverage"
        ]
    )


def select_best(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda result: (
            result["decision_coverage"],
            result[
                "worst_interval_set_coverage"
            ],
            result["overall_set_coverage"],
            -result["pair_threshold"],
        ),
    )


def search_agreement_policy(
    prepared: dict[str, Any],
    requirements: dict[str, float],
) -> dict[str, Any] | None:
    candidates = []

    for single_threshold in (
        SINGLE_THRESHOLDS
    ):
        for pair_threshold in (
            PAIR_THRESHOLDS
        ):
            result = evaluate_policy(
                prepared=prepared,
                pair_threshold=float(
                    pair_threshold
                ),
                single_threshold=float(
                    single_threshold
                ),
                allow_singletons=True,
            )

            interval_results = (
                evaluate_by_interval(
                    prepared=prepared,
                    pair_threshold=float(
                        pair_threshold
                    ),
                    single_threshold=float(
                        single_threshold
                    ),
                    allow_singletons=True,
                )
            )

            result = attach_interval_robustness(
                result,
                interval_results,
            )

            if agreement_policy_eligible(
                result,
                requirements,
            ):
                candidates.append(result)

    return select_best(candidates)


def search_pair_only_policy(
    prepared: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = []

    for pair_threshold in PAIR_THRESHOLDS:
        result = evaluate_policy(
            prepared=prepared,
            pair_threshold=float(
                pair_threshold
            ),
            single_threshold=None,
            allow_singletons=False,
        )

        interval_results = (
            evaluate_by_interval(
                prepared=prepared,
                pair_threshold=float(
                    pair_threshold
                ),
                single_threshold=None,
                allow_singletons=False,
            )
        )

        result = attach_interval_robustness(
            result,
            interval_results,
        )

        if pair_only_eligible(result):
            candidates.append(result)

    return select_best(candidates)


def evaluate_selected_on_validation(
    prepared: dict[str, Any],
    selected: dict[str, Any],
    allow_singletons: bool,
) -> dict[str, Any]:
    result = evaluate_policy(
        prepared=prepared,
        pair_threshold=selected[
            "pair_threshold"
        ],
        single_threshold=selected[
            "single_threshold"
        ],
        allow_singletons=allow_singletons,
    )

    interval_results = evaluate_by_interval(
        prepared=prepared,
        pair_threshold=selected[
            "pair_threshold"
        ],
        single_threshold=selected[
            "single_threshold"
        ],
        allow_singletons=allow_singletons,
    )

    return attach_interval_robustness(
        result,
        interval_results,
    )


def format_value(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_result(
    name: str,
    oof: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> None:
    print("\n" + "-" * 100)
    print(name)
    print("-" * 100)

    if oof is None:
        print("No eligible OOF policy.")
        return

    print(
        "OOF thresholds: "
        f"single={oof['single_threshold']}, "
        f"pair={oof['pair_threshold']}"
    )

    print(
        "OOF: "
        f"decision={oof['decision_coverage']:.4f}, "
        f"set coverage="
        f"{oof['overall_set_coverage']:.4f}, "
        f"worst interval="
        f"{oof['worst_interval_set_coverage']:.4f}, "
        f"single accuracy="
        f"{format_value(oof['singleton_accuracy'])}, "
        f"pair coverage="
        f"{format_value(oof['pair_coverage'])}, "
        f"average size="
        f"{format_value(oof['average_set_size'])}"
    )

    if validation is None:
        return

    print(
        "Validation: "
        f"decision="
        f"{validation['decision_coverage']:.4f}, "
        f"set coverage="
        f"{format_value(validation['overall_set_coverage'])}, "
        f"worst interval="
        f"{format_value(validation['worst_interval_set_coverage'])}, "
        f"single accuracy="
        f"{format_value(validation['singleton_accuracy'])}, "
        f"pair coverage="
        f"{format_value(validation['pair_coverage'])}, "
        f"average size="
        f"{format_value(validation['average_set_size'])}, "
        f"no-call="
        f"{validation['policy_no_call_rate']:.4f}"
    )

    print("Validation intervals:")

    for interval, metrics in (
        validation["by_interval"].items()
    ):
        print(
            f"  {interval}: "
            f"decisions={metrics['decision_rows']}, "
            f"decision coverage="
            f"{metrics['decision_coverage']:.4f}, "
            f"set coverage="
            f"{format_value(metrics['overall_set_coverage'])}, "
            f"single accuracy="
            f"{format_value(metrics['singleton_accuracy'])}, "
            f"pair coverage="
            f"{format_value(metrics['pair_coverage'])}"
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
    print("MOSAIC-PHASE ROBUST SELECTIVE POLICY")
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

    print(
        "\nGenerating participant-grouped "
        "OOF expert predictions..."
    )

    (
        oof_rhr,
        oof_temperature,
        oof_router,
        oof_routes,
    ) = create_oof_expert_outputs(
        train,
        rhr_features,
        temperature_features,
    )

    oof_called = np.isfinite(
        oof_router
    ).all(axis=1)

    probability_temperature, _ = (
        fit_temperature(
            oof_router[oof_called],
            train.loc[
                oof_called,
                "phase",
            ],
        )
    )

    calibrated_oof = np.full_like(
        oof_router,
        np.nan,
    )

    calibrated_oof[oof_called] = (
        apply_temperature_scaling(
            oof_router[oof_called],
            probability_temperature,
        )
    )

    validation_rhr = predict_aligned(
        rhr_entry["model"],
        validation,
        rhr_features,
    )

    validation_temperature = (
        predict_aligned(
            temperature_entry["model"],
            validation,
            temperature_features,
        )
    )

    (
        validation_router,
        validation_routes,
    ) = create_equal_sensor_router(
        validation,
        validation_rhr,
        validation_temperature,
    )

    validation_called = np.isfinite(
        validation_router
    ).all(axis=1)

    calibrated_validation = np.full_like(
        validation_router,
        np.nan,
    )

    calibrated_validation[
        validation_called
    ] = apply_temperature_scaling(
        validation_router[
            validation_called
        ],
        probability_temperature,
    )

    prepared_oof = prepare_rank_data(
        dataframe=train,
        router_probabilities=(
            calibrated_oof
        ),
        rhr_probabilities=oof_rhr,
        temperature_probabilities=(
            oof_temperature
        ),
        routes=oof_routes,
    )

    prepared_validation = prepare_rank_data(
        dataframe=validation,
        router_probabilities=(
            calibrated_validation
        ),
        rhr_probabilities=(
            validation_rhr
        ),
        temperature_probabilities=(
            validation_temperature
        ),
        routes=validation_routes,
    )

    print(
        "\nProbability temperature: "
        f"{probability_temperature:.4f}"
    )

    results: dict[str, Any] = {}

    for tier_name, requirements in (
        ROBUST_TIERS.items()
    ):
        print(
            f"\nSearching {tier_name}..."
        )

        selected_oof = (
            search_agreement_policy(
                prepared_oof,
                requirements,
            )
        )

        if selected_oof is None:
            results[tier_name] = None
            continue

        validation_result = (
            evaluate_selected_on_validation(
                prepared_validation,
                selected_oof,
                allow_singletons=True,
            )
        )

        results[tier_name] = {
            "family": (
                "Agreement-gated singleton "
                "plus top-two set"
            ),
            "requirements": requirements,
            "oof": selected_oof,
            "validation": (
                validation_result
            ),
        }

    print(
        "\nSearching robust pair-only fallback..."
    )

    pair_only_oof = (
        search_pair_only_policy(
            prepared_oof
        )
    )

    if pair_only_oof is None:
        results[
            "Pair-only robust fallback"
        ] = None
    else:
        pair_only_validation = (
            evaluate_selected_on_validation(
                prepared_validation,
                pair_only_oof,
                allow_singletons=False,
            )
        )

        results[
            "Pair-only robust fallback"
        ] = {
            "family": (
                "Top-two set or no-call"
            ),
            "requirements": (
                PAIR_ONLY_REQUIREMENTS
            ),
            "oof": pair_only_oof,
            "validation": (
                pair_only_validation
            ),
        }

    print("\n" + "=" * 100)
    print("ROBUST POLICY RESULTS")
    print("=" * 100)

    for policy_name, result in (
        results.items()
    ):
        if result is None:
            print_result(
                policy_name,
                None,
                None,
            )
        else:
            print_result(
                policy_name,
                result["oof"],
                result["validation"],
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
        "probability_temperature": float(
            probability_temperature
        ),
        "singleton_gate": (
            "Singletons require both RHR and "
            "temperature, agreement between the "
            "two experts, and the confidence threshold."
        ),
        "selection_constraint": (
            "Policies must satisfy aggregate OOF "
            "requirements and a minimum OOF coverage "
            "requirement in the worst study interval."
        ),
        "policies": results,
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

    artifact = {
        "phase_order": PHASES,
        "probability_temperature": float(
            probability_temperature
        ),
        "candidate_policies": results,
        "test_evaluated": False,
    }

    with CANDIDATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        f"\nSafe policy report:\n"
        f"{REPORT_PATH}"
    )

    print(
        f"\nCandidate policy artifact:\n"
        f"{CANDIDATE_PATH}"
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()