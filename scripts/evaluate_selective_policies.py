from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from calibrate_uncertainty import (
    PHASES,
    PHASE_TO_INDEX,
    apply_temperature_scaling,
    create_equal_sensor_router,
    create_oof_predictions,
    fit_temperature,
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

REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint7b_selective_policies.json"
)

CANDIDATE_PATH = (
    ARTIFACTS_DIR
    / "selective_policy_candidates.json"
)

CV_SPLITS = 4

SINGLE_THRESHOLDS = np.round(
    np.arange(0.30, 0.81, 0.01),
    2,
)

PAIR_THRESHOLDS = np.round(
    np.arange(0.50, 0.96, 0.01),
    2,
)

MINIMUM_SINGLETON_ROWS = 25
MINIMUM_PAIR_ROWS = 50

POLICY_TIERS = {
    "Practical": {
        "minimum_singleton_accuracy": 0.50,
        "minimum_pair_coverage": 0.65,
        "minimum_overall_set_coverage": 0.65,
    },
    "Balanced": {
        "minimum_singleton_accuracy": 0.55,
        "minimum_pair_coverage": 0.70,
        "minimum_overall_set_coverage": 0.70,
    },
    "Conservative": {
        "minimum_singleton_accuracy": 0.60,
        "minimum_pair_coverage": 0.75,
        "minimum_overall_set_coverage": 0.75,
    },
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


def prepare_rank_data(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    routes: np.ndarray,
) -> dict[str, Any]:
    row_count = len(dataframe)

    called = np.isfinite(
        probabilities
    ).all(axis=1)

    truth_indices = np.array(
        [
            PHASE_TO_INDEX[phase]
            for phase in dataframe["phase"]
        ],
        dtype=int,
    )

    maximum_probability = np.full(
        row_count,
        np.nan,
        dtype=float,
    )

    top_two_probability = np.full(
        row_count,
        np.nan,
        dtype=float,
    )

    top_one_correct = np.zeros(
        row_count,
        dtype=bool,
    )

    top_two_correct = np.zeros(
        row_count,
        dtype=bool,
    )

    called_probabilities = probabilities[
        called
    ]

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
        order[:, 0]
        == called_truth
    )

    top_two_correct[called] = (
        (
            order[:, 0]
            == called_truth
        )
        |
        (
            order[:, 1]
            == called_truth
        )
    )

    return {
        "rows": row_count,
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
        "study_interval": (
            dataframe[
                "study_interval"
            ].astype(str).to_numpy()
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
    single_threshold: float,
    pair_threshold: float,
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

    singleton = (
        called
        & (
            prepared[
                "maximum_probability"
            ]
            >= single_threshold
        )
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

    decision = (
        singleton
        | pair
    )

    singleton_correct = (
        prepared["top_one_correct"]
        & singleton
    )

    pair_correct = (
        prepared["top_two_correct"]
        & pair
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

    if subset_rows == 0:
        raise ValueError(
            "Cannot evaluate an empty subset."
        )

    sensor_coverage = (
        called_rows
        / subset_rows
    )

    decision_coverage = (
        decision_rows
        / subset_rows
    )

    if called_rows:
        decision_coverage_among_called = (
            decision_rows
            / called_rows
        )
    else:
        decision_coverage_among_called = 0.0

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

        singleton_rate_among_decisions = (
            singleton_rows
            / decision_rows
        )
    else:
        overall_set_coverage = None
        average_set_size = None
        singleton_rate_among_decisions = 0.0

    singleton_accuracy = safe_mean(
        prepared["top_one_correct"][
            singleton
        ]
    )

    pair_coverage = safe_mean(
        prepared["top_two_correct"][
            pair
        ]
    )

    return {
        "single_threshold": float(
            single_threshold
        ),
        "pair_threshold": float(
            pair_threshold
        ),
        "subset_rows": subset_rows,
        "sensor_called_rows": called_rows,
        "sensor_coverage": float(
            sensor_coverage
        ),
        "singleton_rows": singleton_rows,
        "singleton_accuracy": (
            singleton_accuracy
        ),
        "pair_rows": pair_rows,
        "pair_coverage": pair_coverage,
        "decision_rows": decision_rows,
        "decision_coverage": float(
            decision_coverage
        ),
        "decision_coverage_among_sensor_called": float(
            decision_coverage_among_called
        ),
        "overall_set_coverage": (
            overall_set_coverage
        ),
        "average_set_size": (
            average_set_size
        ),
        "singleton_rate_among_decisions": float(
            singleton_rate_among_decisions
        ),
        "policy_no_call_rate": float(
            1.0 - decision_coverage
        ),
    }


def policy_is_eligible(
    metrics: dict[str, Any],
    requirements: dict[str, float],
) -> bool:
    if (
        metrics["singleton_rows"]
        < MINIMUM_SINGLETON_ROWS
    ):
        return False

    if (
        metrics["pair_rows"]
        < MINIMUM_PAIR_ROWS
    ):
        return False

    if (
        metrics["singleton_accuracy"]
        is None
        or metrics["pair_coverage"]
        is None
        or metrics[
            "overall_set_coverage"
        ] is None
    ):
        return False

    return (
        metrics["singleton_accuracy"]
        >= requirements[
            "minimum_singleton_accuracy"
        ]
        and metrics["pair_coverage"]
        >= requirements[
            "minimum_pair_coverage"
        ]
        and metrics[
            "overall_set_coverage"
        ]
        >= requirements[
            "minimum_overall_set_coverage"
        ]
    )


def select_policy(
    prepared: dict[str, Any],
    requirements: dict[str, float],
) -> dict[str, Any] | None:
    eligible = []

    for single_threshold in (
        SINGLE_THRESHOLDS
    ):
        for pair_threshold in (
            PAIR_THRESHOLDS
        ):
            metrics = evaluate_policy(
                prepared,
                float(single_threshold),
                float(pair_threshold),
            )

            if policy_is_eligible(
                metrics,
                requirements,
            ):
                eligible.append(metrics)

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda result: (
            result["decision_coverage"],
            result[
                "singleton_rate_among_decisions"
            ],
            result["overall_set_coverage"],
            -result["single_threshold"],
            -result["pair_threshold"],
        ),
    )


def baseline_rank_metrics(
    prepared: dict[str, Any],
) -> dict[str, Any]:
    called = prepared["called"]

    return {
        "sensor_called_rows": int(
            called.sum()
        ),
        "sensor_coverage": float(
            called.mean()
        ),
        "unrestricted_top1_accuracy": float(
            prepared["top_one_correct"][
                called
            ].mean()
        ),
        "unrestricted_top2_coverage": float(
            prepared["top_two_correct"][
                called
            ].mean()
        ),
        "mean_maximum_probability": float(
            prepared[
                "maximum_probability"
            ][called].mean()
        ),
        "mean_top_two_probability": float(
            prepared[
                "top_two_probability"
            ][called].mean()
        ),
    }


def grouped_policy_results(
    prepared: dict[str, Any],
    single_threshold: float,
    pair_threshold: float,
    group_name: str,
) -> dict[str, Any]:
    values = prepared[group_name]

    output = {}

    for value in sorted(
        set(values),
        key=str,
    ):
        mask = values == value

        output[str(value)] = (
            evaluate_policy(
                prepared,
                single_threshold,
                pair_threshold,
                subset_mask=mask,
            )
        )

    return output


def format_value(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_policy_table(
    results: dict[str, Any],
) -> None:
    print("\n" + "=" * 154)
    print("SELECTIVE POLICY COMPARISON")
    print("=" * 154)

    header = (
        f"{'Tier':14s}"
        f"{'Single':>9s}"
        f"{'Pair':>9s}"
        f"{'OOF Dec':>10s}"
        f"{'OOF Set':>10s}"
        f"{'OOF 1Acc':>10s}"
        f"{'OOF 2Cov':>10s}"
        f"{'Val Dec':>10s}"
        f"{'Val Set':>10s}"
        f"{'Val 1Acc':>10s}"
        f"{'Val 2Cov':>10s}"
        f"{'Val AvgK':>10s}"
        f"{'Val NoCall':>12s}"
    )

    print(header)
    print("-" * len(header))

    for tier_name, result in results.items():
        if result is None:
            print(
                f"{tier_name:14s}"
                f"{'N/A':>9s}"
                f"{'N/A':>9s}"
                f"{'No eligible OOF policy':>40s}"
            )
            continue

        oof = result["oof"]
        validation = result["validation"]

        print(
            f"{tier_name:14s}"
            f"{oof['single_threshold']:9.2f}"
            f"{oof['pair_threshold']:9.2f}"
            f"{oof['decision_coverage']:10.4f}"
            f"{oof['overall_set_coverage']:10.4f}"
            f"{oof['singleton_accuracy']:10.4f}"
            f"{oof['pair_coverage']:10.4f}"
            f"{validation['decision_coverage']:10.4f}"
            f"{format_value(validation['overall_set_coverage']):>10s}"
            f"{format_value(validation['singleton_accuracy']):>10s}"
            f"{format_value(validation['pair_coverage']):>10s}"
            f"{format_value(validation['average_set_size']):>10s}"
            f"{validation['policy_no_call_rate']:12.4f}"
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
    print("MOSAIC-PHASE SELECTIVE RISK-COVERAGE POLICIES")
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

    rhr_features = rhr_entry["features"]

    temperature_features = (
        temperature_entry["features"]
    )

    print(
        "\nRegenerating participant-grouped "
        "OOF router probabilities..."
    )

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

    temperature_value, _ = fit_temperature(
        oof_probabilities[oof_called],
        train.loc[
            oof_called,
            "phase",
        ],
    )

    calibrated_oof = np.full_like(
        oof_probabilities,
        np.nan,
    )

    calibrated_oof[oof_called] = (
        apply_temperature_scaling(
            oof_probabilities[oof_called],
            temperature_value,
        )
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
        temperature_value,
    )

    prepared_oof = prepare_rank_data(
        train,
        calibrated_oof,
        oof_routes,
    )

    prepared_validation = prepare_rank_data(
        validation,
        calibrated_validation,
        validation_routes,
    )

    oof_baseline = baseline_rank_metrics(
        prepared_oof
    )

    validation_baseline = (
        baseline_rank_metrics(
            prepared_validation
        )
    )

    print(
        "\nFitted probability temperature: "
        f"{temperature_value:.4f}"
    )

    print("\nUnrestricted ranking performance:")

    print(
        "  Training OOF:"
        f" top-1 accuracy="
        f"{oof_baseline['unrestricted_top1_accuracy']:.4f},"
        f" top-2 coverage="
        f"{oof_baseline['unrestricted_top2_coverage']:.4f},"
        f" sensor coverage="
        f"{oof_baseline['sensor_coverage']:.4f}"
    )

    print(
        "  Validation:"
        f" top-1 accuracy="
        f"{validation_baseline['unrestricted_top1_accuracy']:.4f},"
        f" top-2 coverage="
        f"{validation_baseline['unrestricted_top2_coverage']:.4f},"
        f" sensor coverage="
        f"{validation_baseline['sensor_coverage']:.4f}"
    )

    selected_results: dict[
        str,
        Any,
    ] = {}

    for tier_name, requirements in (
        POLICY_TIERS.items()
    ):
        print(
            f"\nSearching OOF policy for "
            f"{tier_name} tier..."
        )

        selected_oof = select_policy(
            prepared_oof,
            requirements,
        )

        if selected_oof is None:
            selected_results[
                tier_name
            ] = None

            print(
                "  No policy satisfied all "
                "OOF requirements."
            )
            continue

        validation_result = evaluate_policy(
            prepared_validation,
            selected_oof[
                "single_threshold"
            ],
            selected_oof[
                "pair_threshold"
            ],
        )

        by_interval = grouped_policy_results(
            prepared_validation,
            selected_oof[
                "single_threshold"
            ],
            selected_oof[
                "pair_threshold"
            ],
            "study_interval",
        )

        by_route = grouped_policy_results(
            prepared_validation,
            selected_oof[
                "single_threshold"
            ],
            selected_oof[
                "pair_threshold"
            ],
            "route",
        )

        selected_results[tier_name] = {
            "requirements": requirements,
            "oof": selected_oof,
            "validation": (
                validation_result
            ),
            "validation_by_interval": (
                by_interval
            ),
            "validation_by_route": (
                by_route
            ),
        }

    print_policy_table(
        selected_results
    )

    print("\nValidation interval results:")

    for tier_name, result in (
        selected_results.items()
    ):
        if result is None:
            continue

        print(f"\n  {tier_name}")

        for interval, metrics in (
            result[
                "validation_by_interval"
            ].items()
        ):
            print(
                f"    Interval {interval}: "
                f"decision={metrics['decision_coverage']:.4f}, "
                f"set coverage="
                f"{format_value(metrics['overall_set_coverage'])}, "
                f"singleton accuracy="
                f"{format_value(metrics['singleton_accuracy'])}, "
                f"pair coverage="
                f"{format_value(metrics['pair_coverage'])}"
            )

    output = {
        "test_evaluated": False,
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "reserved_test_rows": int(
            len(test)
        ),
        "probability_temperature": float(
            temperature_value
        ),
        "policy_definition": {
            "single_phase": (
                "Maximum calibrated probability "
                "is at least the single threshold."
            ),
            "two_phase_set": (
                "Otherwise, the sum of the two "
                "largest probabilities is at least "
                "the pair threshold."
            ),
            "no_call": (
                "Neither rule is satisfied, or "
                "neither predictive sensor exists."
            ),
        },
        "minimum_policy_sample_sizes": {
            "singleton_rows": (
                MINIMUM_SINGLETON_ROWS
            ),
            "pair_rows": MINIMUM_PAIR_ROWS,
        },
        "unrestricted_rank_metrics": {
            "oof": oof_baseline,
            "validation": validation_baseline,
        },
        "policy_tiers": selected_results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            default=json_safe,
        )

    candidate_artifact = {
        "phase_order": PHASES,
        "probability_temperature": float(
            temperature_value
        ),
        "candidates": selected_results,
        "test_evaluated": False,
    }

    with CANDIDATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            candidate_artifact,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        f"\nSafe selective-policy report:\n"
        f"{REPORT_PATH}"
    )

    print(
        f"\nCandidate policy artifact:\n"
        f"{CANDIDATE_PATH}"
    )

    print(
        "\nThese are empirical research operating "
        "points, not clinical guarantees."
    )

    print(
        "\nThe held-out test set was not evaluated."
    )


if __name__ == "__main__":
    main()