from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from calibrate_uncertainty import (
    PHASES,
    apply_temperature_scaling,
    create_equal_sensor_router,
    json_safe,
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

POLICY_CANDIDATES_PATH = (
    ARTIFACTS_DIR
    / "robust_policy_candidates.json"
)

FINAL_POLICY_PATH = (
    ARTIFACTS_DIR
    / "final_frozen_policy.json"
)

FINAL_REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint8_final_test_report.json"
)

FINAL_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "checkpoint8_final_test_predictions.csv"
)

POLICY_NAME = "Pair-only robust fallback"


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    required_paths = [
        FEATURE_TABLE_PATH,
        EXPERT_BUNDLE_PATH,
        POLICY_CANDIDATES_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    if FINAL_REPORT_PATH.exists():
        raise RuntimeError(
            "The final test report already exists. "
            "Do not repeatedly evaluate or tune against "
            "the held-out test set."
        )

    dataframe = pd.read_parquet(
        FEATURE_TABLE_PATH
    )

    expert_bundle = joblib.load(
        EXPERT_BUNDLE_PATH
    )

    with POLICY_CANDIDATES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        policy_candidates = json.load(file)

    return (
        dataframe,
        expert_bundle,
        policy_candidates,
    )


def extract_frozen_policy(
    candidates: dict[str, Any],
) -> dict[str, Any]:
    policies = candidates.get(
        "candidate_policies",
        {},
    )

    if POLICY_NAME not in policies:
        raise KeyError(
            f"{POLICY_NAME} was not found in "
            "the robust policy artifact."
        )

    selected = policies[POLICY_NAME]

    if selected is None:
        raise ValueError(
            f"{POLICY_NAME} has no eligible policy."
        )

    oof_policy = selected["oof"]

    pair_threshold = float(
        oof_policy["pair_threshold"]
    )

    single_threshold = (
        oof_policy.get("single_threshold")
    )

    if single_threshold is not None:
        raise ValueError(
            "The frozen final policy must be pair-only."
        )

    probability_temperature = float(
        candidates["probability_temperature"]
    )

    return {
        "policy_name": POLICY_NAME,
        "policy_family": (
            "Top-two phase set or no-call"
        ),
        "phase_order": PHASES,
        "router": {
            "both_rhr_and_temperature": (
                "Equal average of RHR and "
                "temperature expert probabilities"
            ),
            "rhr_only": "RHR expert",
            "temperature_only": (
                "Temperature expert"
            ),
            "neither": "NO_CALL",
        },
        "probability_temperature": (
            probability_temperature
        ),
        "pair_threshold": pair_threshold,
        "singleton_outputs_enabled": False,
        "decision_rule": (
            "Return the two highest-probability phases "
            "when their calibrated probability sum is "
            "at least the pair threshold. Otherwise "
            "return NO_CALL."
        ),
        "selection_data": (
            "Training participant-grouped out-of-fold "
            "predictions, confirmed on validation "
            "participants."
        ),
        "test_metrics_used_for_selection": False,
    }


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator == 0:
        return None

    return float(numerator / denominator)


def format_value(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def evaluate_subset(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    pair_threshold: float,
    subset_mask: np.ndarray,
) -> dict[str, Any]:
    subset_indices = np.flatnonzero(
        subset_mask
    )

    subset = dataframe.iloc[
        subset_indices
    ]

    subset_probabilities = probabilities[
        subset_indices
    ]

    total_rows = len(subset)

    if total_rows == 0:
        raise ValueError(
            "Cannot evaluate an empty subset."
        )

    sensor_called = np.isfinite(
        subset_probabilities
    ).all(axis=1)

    sensor_called_rows = int(
        sensor_called.sum()
    )

    no_sensor_rows = int(
        (~sensor_called).sum()
    )

    result: dict[str, Any] = {
        "rows": int(total_rows),
        "sensor_called_rows": (
            sensor_called_rows
        ),
        "sensor_coverage": float(
            sensor_called_rows / total_rows
        ),
        "no_sensor_rows": no_sensor_rows,
    }

    if sensor_called_rows == 0:
        result.update(
            {
                "unrestricted_top1_accuracy": None,
                "unrestricted_top1_balanced_accuracy": None,
                "unrestricted_top1_macro_f1": None,
                "unrestricted_top2_coverage": None,
                "decision_rows": 0,
                "decision_coverage": 0.0,
                "decision_coverage_among_sensor_called": 0.0,
                "pair_set_coverage": None,
                "average_set_size": None,
                "no_call_rows": int(total_rows),
                "no_call_rate": 1.0,
                "low_evidence_no_call_rows": 0,
                "mean_top_two_probability_on_decisions": None,
            }
        )

        return result

    called_probabilities = (
        subset_probabilities[
            sensor_called
        ]
    )

    called_truth = (
        subset.loc[
            sensor_called,
            "phase",
        ]
        .astype(str)
        .to_numpy()
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

    phase_array = np.asarray(
        PHASES,
        dtype=object,
    )

    top_one_predictions = (
        phase_array[order[:, 0]]
    )

    top_two_predictions = (
        phase_array[order[:, :2]]
    )

    top_two_probability = (
        sorted_probabilities[
            :,
            :2,
        ].sum(axis=1)
    )

    top_two_contains_truth = (
        (
            top_two_predictions[:, 0]
            == called_truth
        )
        |
        (
            top_two_predictions[:, 1]
            == called_truth
        )
    )

    decision = (
        top_two_probability
        >= pair_threshold
    )

    decision_rows = int(
        decision.sum()
    )

    low_evidence_no_call_rows = int(
        (~decision).sum()
    )

    no_call_rows = (
        no_sensor_rows
        + low_evidence_no_call_rows
    )

    result.update(
        {
            "unrestricted_top1_accuracy": float(
                accuracy_score(
                    called_truth,
                    top_one_predictions,
                )
            ),
            "unrestricted_top1_balanced_accuracy": float(
                balanced_accuracy_score(
                    called_truth,
                    top_one_predictions,
                )
            ),
            "unrestricted_top1_macro_f1": float(
                f1_score(
                    called_truth,
                    top_one_predictions,
                    labels=PHASES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "unrestricted_top2_coverage": float(
                top_two_contains_truth.mean()
            ),
            "decision_rows": decision_rows,
            "decision_coverage": float(
                decision_rows / total_rows
            ),
            "decision_coverage_among_sensor_called": float(
                decision_rows
                / sensor_called_rows
            ),
            "pair_set_coverage": (
                float(
                    top_two_contains_truth[
                        decision
                    ].mean()
                )
                if decision_rows
                else None
            ),
            "average_set_size": (
                2.0
                if decision_rows
                else None
            ),
            "no_call_rows": int(
                no_call_rows
            ),
            "no_call_rate": float(
                no_call_rows / total_rows
            ),
            "low_evidence_no_call_rows": (
                low_evidence_no_call_rows
            ),
            "mean_top_two_probability_on_decisions": (
                float(
                    top_two_probability[
                        decision
                    ].mean()
                )
                if decision_rows
                else None
            ),
        }
    )

    return result


def create_prediction_output(
    test: pd.DataFrame,
    calibrated_probabilities: np.ndarray,
    routes: np.ndarray,
    pair_threshold: float,
) -> pd.DataFrame:
    output = test[
        [
            "id",
            "study_interval",
            "day_in_study",
            "phase",
            "has_rhr",
            "has_temperature",
        ]
    ].copy()

    output["sensor_route"] = routes

    sensor_called = np.isfinite(
        calibrated_probabilities
    ).all(axis=1)

    top_one_phase = np.full(
        len(test),
        "NO_CALL",
        dtype=object,
    )

    top_two_set = np.full(
        len(test),
        "NO_CALL",
        dtype=object,
    )

    maximum_probability = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    top_two_probability = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    decision = np.zeros(
        len(test),
        dtype=bool,
    )

    if sensor_called.any():
        called_probabilities = (
            calibrated_probabilities[
                sensor_called
            ]
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

        phase_array = np.asarray(
            PHASES,
            dtype=object,
        )

        called_top_one = (
            phase_array[
                order[:, 0]
            ]
        )

        called_top_two = (
            phase_array[
                order[:, :2]
            ]
        )

        called_top_two_probability = (
            sorted_probabilities[
                :,
                :2,
            ].sum(axis=1)
        )

        called_decision = (
            called_top_two_probability
            >= pair_threshold
        )

        top_one_phase[
            sensor_called
        ] = called_top_one

        maximum_probability[
            sensor_called
        ] = sorted_probabilities[
            :,
            0,
        ]

        top_two_probability[
            sensor_called
        ] = called_top_two_probability

        called_sets = np.asarray(
            [
                (
                    f"{phases[0]} | "
                    f"{phases[1]}"
                )
                if is_decision
                else "NO_CALL"
                for phases, is_decision
                in zip(
                    called_top_two,
                    called_decision,
                )
            ],
            dtype=object,
        )

        top_two_set[
            sensor_called
        ] = called_sets

        decision[
            sensor_called
        ] = called_decision

    output["top_ranked_phase"] = (
        top_one_phase
    )

    output["maximum_probability"] = (
        maximum_probability
    )

    output["top_two_probability"] = (
        top_two_probability
    )

    output["final_output"] = (
        top_two_set
    )

    output["decision_made"] = (
        decision
    )

    output["truth_in_final_set"] = False

    decided_indices = np.flatnonzero(
        decision
    )

    for index in decided_indices:
        selected_phases = set(
            output.iloc[index][
                "final_output"
            ].split(" | ")
        )

        output.loc[
            output.index[index],
            "truth_in_final_set",
        ] = (
            output.iloc[index]["phase"]
            in selected_phases
        )

    for phase_index, phase in enumerate(
        PHASES
    ):
        output[
            f"calibrated_probability_{phase.lower()}"
        ] = calibrated_probabilities[
            :,
            phase_index,
        ]

    return output


def print_metrics(
    title: str,
    metrics: dict[str, Any],
) -> None:
    print("\n" + "-" * 76)
    print(title)
    print("-" * 76)

    print(f"Rows: {metrics['rows']}")
    print(
        "Sensor coverage: "
        f"{metrics['sensor_coverage']:.4f}"
    )

    print(
        "Unrestricted top-1 accuracy: "
        f"{format_value(metrics['unrestricted_top1_accuracy'])}"
    )

    print(
        "Unrestricted top-1 balanced accuracy: "
        f"{format_value(metrics['unrestricted_top1_balanced_accuracy'])}"
    )

    print(
        "Unrestricted top-1 macro F1: "
        f"{format_value(metrics['unrestricted_top1_macro_f1'])}"
    )

    print(
        "Unrestricted top-2 coverage: "
        f"{format_value(metrics['unrestricted_top2_coverage'])}"
    )

    print(
        "Final decision coverage: "
        f"{metrics['decision_coverage']:.4f}"
    )

    print(
        "Final pair-set coverage: "
        f"{format_value(metrics['pair_set_coverage'])}"
    )

    print(
        "Final no-call rate: "
        f"{metrics['no_call_rate']:.4f}"
    )

    print(
        "No-call due to missing sensors: "
        f"{metrics['no_sensor_rows']}"
    )

    print(
        "No-call due to insufficient evidence: "
        f"{metrics['low_evidence_no_call_rows']}"
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
    print("MOSAIC-PHASE FINAL HELD-OUT TEST EVALUATION")
    print("=" * 76)

    (
        dataframe,
        expert_bundle,
        policy_candidates,
    ) = load_inputs()

    frozen_policy = extract_frozen_policy(
        policy_candidates
    )

    with FINAL_POLICY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            frozen_policy,
            file,
            indent=2,
            default=json_safe,
        )

    print("\nFrozen policy:")
    print(
        "  Output: top-two phase set or no-call"
    )
    print(
        "  Probability temperature: "
        f"{frozen_policy['probability_temperature']:.4f}"
    )
    print(
        "  Pair threshold: "
        f"{frozen_policy['pair_threshold']:.4f}"
    )

    test = dataframe[
        dataframe["split"] == "test"
    ].copy().reset_index(drop=True)

    train_validation = dataframe[
        dataframe["split"].isin(
            ["train", "validation"]
        )
    ]

    print(
        f"\nTrain + validation rows not evaluated: "
        f"{len(train_validation):,}"
    )

    print(
        f"Held-out test rows: {len(test):,}"
    )

    rhr_entry = expert_bundle[
        "models"
    ]["RHR expert"]

    temperature_entry = expert_bundle[
        "models"
    ]["Temperature expert"]

    rhr_probabilities = predict_aligned(
        rhr_entry["model"],
        test,
        rhr_entry["features"],
    )

    temperature_probabilities = (
        predict_aligned(
            temperature_entry["model"],
            test,
            temperature_entry[
                "features"
            ],
        )
    )

    (
        router_probabilities,
        routes,
    ) = create_equal_sensor_router(
        test,
        rhr_probabilities,
        temperature_probabilities,
    )

    sensor_called = np.isfinite(
        router_probabilities
    ).all(axis=1)

    calibrated_probabilities = np.full_like(
        router_probabilities,
        np.nan,
    )

    calibrated_probabilities[
        sensor_called
    ] = apply_temperature_scaling(
        router_probabilities[
            sensor_called
        ],
        frozen_policy[
            "probability_temperature"
        ],
    )

    all_mask = np.ones(
        len(test),
        dtype=bool,
    )

    overall = evaluate_subset(
        dataframe=test,
        probabilities=(
            calibrated_probabilities
        ),
        pair_threshold=frozen_policy[
            "pair_threshold"
        ],
        subset_mask=all_mask,
    )

    by_interval = {}

    for interval in sorted(
        test["study_interval"]
        .astype(str)
        .unique()
    ):
        interval_mask = (
            test["study_interval"]
            .astype(str)
            .to_numpy()
            == interval
        )

        by_interval[interval] = (
            evaluate_subset(
                dataframe=test,
                probabilities=(
                    calibrated_probabilities
                ),
                pair_threshold=(
                    frozen_policy[
                        "pair_threshold"
                    ]
                ),
                subset_mask=(
                    interval_mask
                ),
            )
        )

    by_route = {}

    for route in sorted(
        set(routes),
        key=str,
    ):
        route_mask = (
            routes == route
        )

        by_route[str(route)] = (
            evaluate_subset(
                dataframe=test,
                probabilities=(
                    calibrated_probabilities
                ),
                pair_threshold=(
                    frozen_policy[
                        "pair_threshold"
                    ]
                ),
                subset_mask=route_mask,
            )
        )

    by_phase = {}

    for phase in PHASES:
        phase_mask = (
            test["phase"]
            .astype(str)
            .to_numpy()
            == phase
        )

        by_phase[phase] = (
            evaluate_subset(
                dataframe=test,
                probabilities=(
                    calibrated_probabilities
                ),
                pair_threshold=(
                    frozen_policy[
                        "pair_threshold"
                    ]
                ),
                subset_mask=phase_mask,
            )
        )

    prediction_output = (
        create_prediction_output(
            test=test,
            calibrated_probabilities=(
                calibrated_probabilities
            ),
            routes=routes,
            pair_threshold=(
                frozen_policy[
                    "pair_threshold"
                ]
            ),
        )
    )

    route_counts = {
        str(key): int(value)
        for key, value in (
            pd.Series(routes)
            .value_counts()
            .items()
        )
    }

    print_metrics(
        "OVERALL HELD-OUT TEST RESULTS",
        overall,
    )

    print("\nTest sensor routes:")

    for route, count in (
        route_counts.items()
    ):
        print(
            f"  {route:20s}: {count}"
        )

    print("\nResults by study interval:")

    for interval, metrics in (
        by_interval.items()
    ):
        print(
            f"  Interval {interval}: "
            f"n={metrics['rows']}, "
            f"decision={metrics['decision_coverage']:.4f}, "
            f"set coverage="
            f"{format_value(metrics['pair_set_coverage'])}, "
            f"top-1 balanced="
            f"{format_value(metrics['unrestricted_top1_balanced_accuracy'])}, "
            f"no-call={metrics['no_call_rate']:.4f}"
        )

    print("\nResults by sensor route:")

    for route, metrics in (
        by_route.items()
    ):
        print(
            f"  {route:20s}: "
            f"n={metrics['rows']}, "
            f"decision={metrics['decision_coverage']:.4f}, "
            f"set coverage="
            f"{format_value(metrics['pair_set_coverage'])}"
        )

    print("\nResults by true phase:")

    for phase, metrics in (
        by_phase.items()
    ):
        print(
            f"  {phase:12s}: "
            f"n={metrics['rows']}, "
            f"decision={metrics['decision_coverage']:.4f}, "
            f"set coverage="
            f"{format_value(metrics['pair_set_coverage'])}"
        )

    report = {
        "test_evaluated": True,
        "policy_frozen_before_test": True,
        "model_training_data": (
            "Training split only"
        ),
        "policy_selection_data": (
            "Training OOF and validation splits"
        ),
        "held_out_test_rows": int(
            len(test)
        ),
        "frozen_policy": (
            frozen_policy
        ),
        "overall": overall,
        "route_counts": route_counts,
        "by_interval": by_interval,
        "by_route": by_route,
        "by_phase": by_phase,
        "post_test_rule": (
            "No model, feature, threshold or routing "
            "changes may be selected using these test "
            "results."
        ),
    }

    with FINAL_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=json_safe,
        )

    prediction_output.to_csv(
        FINAL_PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 76)

    print(
        f"\nFrozen policy:\n"
        f"{FINAL_POLICY_PATH}"
    )

    print(
        f"\nSafe aggregate test report:\n"
        f"{FINAL_REPORT_PATH}"
    )

    print(
        "\nRestricted test predictions:\n"
        f"{FINAL_PREDICTIONS_PATH}"
    )

    print(
        "\nThe held-out test set has now been opened. "
        "Do not alter the model or policy based on "
        "these results."
    )


if __name__ == "__main__":
    main()