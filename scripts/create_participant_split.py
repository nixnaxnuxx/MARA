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

MASTER_PATH = (
    PROCESSED_DIR / "mosaic_phase_master_daily.parquet"
)

OUTPUT_MASTER_PATH = (
    PROCESSED_DIR
    / "mosaic_phase_master_daily_with_split.parquet"
)

SPLIT_ASSIGNMENT_PATH = (
    ARTIFACTS_DIR / "participant_split.json"
)

REPORT_PATH = (
    REPORTS_DIR / "checkpoint4_participant_split_report.json"
)

RANDOM_SEED = 20260719
SEARCH_ITERATIONS = 20_000

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

MODALITIES = [
    "has_rhr",
    "has_temperature",
    "has_sleep",
    "has_activity",
]

SPLIT_PARTICIPANT_COUNTS = {
    "train": 28,
    "validation": 7,
    "test": 7,
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
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master table was not found: {MASTER_PATH}"
        )

    master = pd.read_parquet(MASTER_PATH)

    required_columns = [
        "id",
        "study_interval",
        "day_in_study",
        "phase",
        *MODALITIES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in master.columns
    ]

    if missing:
        raise KeyError(
            f"Master table is missing columns: {missing}"
        )

    if master["id"].nunique() != 42:
        raise ValueError(
            "Expected 42 participants, but found "
            f"{master['id'].nunique()}."
        )

    unexpected_phases = sorted(
        set(master["phase"].dropna().unique())
        - set(PHASES)
    )

    if unexpected_phases:
        raise ValueError(
            f"Unexpected phase labels: {unexpected_phases}"
        )

    return master


def create_participant_arrays(
    master: pd.DataFrame,
) -> dict[str, Any]:
    participant_ids = sorted(
        master["id"].dropna().unique(),
        key=lambda value: str(value),
    )

    interval_values = sorted(
        master["study_interval"]
        .dropna()
        .unique(),
        key=lambda value: str(value),
    )

    row_counts = []
    phase_counts = []
    modality_counts = []
    interval_counts = []

    for participant_id in participant_ids:
        participant = master[
            master["id"] == participant_id
        ]

        row_counts.append(len(participant))

        phase_counts.append([
            int((participant["phase"] == phase).sum())
            for phase in PHASES
        ])

        modality_counts.append([
            int(participant[column].astype(bool).sum())
            for column in MODALITIES
        ])

        interval_counts.append([
            int(
                (
                    participant["study_interval"]
                    == interval
                ).sum()
            )
            for interval in interval_values
        ])

    return {
        "participant_ids": np.array(
            participant_ids,
            dtype=object,
        ),
        "interval_values": interval_values,
        "row_counts": np.asarray(
            row_counts,
            dtype=float,
        ),
        "phase_counts": np.asarray(
            phase_counts,
            dtype=float,
        ),
        "modality_counts": np.asarray(
            modality_counts,
            dtype=float,
        ),
        "interval_counts": np.asarray(
            interval_counts,
            dtype=float,
        ),
    }


def calculate_candidate_score(
    split_indices: dict[str, np.ndarray],
    arrays: dict[str, Any],
) -> float:
    row_counts = arrays["row_counts"]
    phase_counts = arrays["phase_counts"]
    modality_counts = arrays["modality_counts"]
    interval_counts = arrays["interval_counts"]

    total_rows = row_counts.sum()

    global_phase_distribution = (
        phase_counts.sum(axis=0) / total_rows
    )

    global_modality_distribution = (
        modality_counts.sum(axis=0) / total_rows
    )

    global_interval_distribution = (
        interval_counts.sum(axis=0) / total_rows
    )

    target_fractions = {
        split_name: participant_count / 42
        for split_name, participant_count
        in SPLIT_PARTICIPANT_COUNTS.items()
    }

    score = 0.0

    for split_name, indices in split_indices.items():
        split_rows = row_counts[indices].sum()

        if split_rows <= 0:
            return float("inf")

        actual_row_fraction = split_rows / total_rows
        target_row_fraction = target_fractions[split_name]

        size_error = (
            actual_row_fraction - target_row_fraction
        ) ** 2

        split_phase_distribution = (
            phase_counts[indices].sum(axis=0)
            / split_rows
        )

        phase_error = np.mean(
            (
                split_phase_distribution
                - global_phase_distribution
            )
            ** 2
        )

        split_modality_distribution = (
            modality_counts[indices].sum(axis=0)
            / split_rows
        )

        modality_error = np.mean(
            (
                split_modality_distribution
                - global_modality_distribution
            )
            ** 2
        )

        split_interval_distribution = (
            interval_counts[indices].sum(axis=0)
            / split_rows
        )

        interval_error = np.mean(
            (
                split_interval_distribution
                - global_interval_distribution
            )
            ** 2
        )

        score += (
            2.0 * size_error
            + 1.0 * phase_error
            + 0.75 * modality_error
            + 0.50 * interval_error
        )

    return float(score)


def search_balanced_split(
    arrays: dict[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    participant_count = len(
        arrays["participant_ids"]
    )

    expected_count = sum(
        SPLIT_PARTICIPANT_COUNTS.values()
    )

    if participant_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} participants, "
            f"but found {participant_count}."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    best_split: dict[str, np.ndarray] | None = None
    best_score = float("inf")

    train_end = SPLIT_PARTICIPANT_COUNTS["train"]
    validation_end = (
        train_end
        + SPLIT_PARTICIPANT_COUNTS["validation"]
    )

    for _ in range(SEARCH_ITERATIONS):
        permutation = rng.permutation(
            participant_count
        )

        candidate = {
            "train": permutation[:train_end],
            "validation": permutation[
                train_end:validation_end
            ],
            "test": permutation[validation_end:],
        }

        score = calculate_candidate_score(
            candidate,
            arrays,
        )

        if score < best_score:
            best_score = score

            best_split = {
                name: indices.copy()
                for name, indices in candidate.items()
            }

    if best_split is None:
        raise RuntimeError(
            "No valid participant split was found."
        )

    return best_split, best_score


def participant_coverage_audit(
    master: pd.DataFrame,
) -> dict[str, Any]:
    participant_days = (
        master.groupby("id").size()
    )

    interval_counts = (
        master.groupby("id")["study_interval"]
        .nunique()
    )

    modality_coverage: dict[str, Any] = {}

    for column in MODALITIES:
        coverage = (
            master.groupby("id")[column]
            .mean()
            .mul(100)
        )

        modality_coverage[column] = {
            "minimum_percent": float(
                coverage.min()
            ),
            "median_percent": float(
                coverage.median()
            ),
            "maximum_percent": float(
                coverage.max()
            ),
            "participants_below_50_percent": int(
                (coverage < 50).sum()
            ),
        }

    return {
        "days_per_participant": {
            "minimum": int(participant_days.min()),
            "median": float(
                participant_days.median()
            ),
            "maximum": int(participant_days.max()),
        },
        "participants_by_number_of_intervals": {
            str(key): int(value)
            for key, value in (
                interval_counts
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "modality_coverage": modality_coverage,
    }


def summarize_split(
    master: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    subset = master[
        master["split"] == split_name
    ]

    phase_counts = (
        subset["phase"]
        .value_counts()
        .reindex(PHASES, fill_value=0)
    )

    phase_percentages = (
        phase_counts
        .div(len(subset))
        .mul(100)
    )

    modality_percentages = (
        subset[MODALITIES]
        .mean()
        .mul(100)
    )

    interval_counts = (
        subset["study_interval"]
        .value_counts()
        .sort_index()
    )

    return {
        "participants": int(
            subset["id"].nunique()
        ),
        "days": int(len(subset)),
        "phase_counts": {
            phase: int(phase_counts[phase])
            for phase in PHASES
        },
        "phase_percentages": {
            phase: float(
                phase_percentages[phase]
            )
            for phase in PHASES
        },
        "modality_coverage_percent": {
            column: float(
                modality_percentages[column]
            )
            for column in MODALITIES
        },
        "study_interval_counts": {
            str(key): int(value)
            for key, value in interval_counts.items()
        },
        "days_with_all_modalities": int(
            (
                subset["available_modality_count"]
                == 4
            ).sum()
        ),
        "days_with_no_modalities": int(
            (
                subset["available_modality_count"]
                == 0
            ).sum()
        ),
    }


def print_participant_audit(
    audit: dict[str, Any],
) -> None:
    print("\n" + "=" * 76)
    print("PARTICIPANT COVERAGE AUDIT")
    print("=" * 76)

    days = audit["days_per_participant"]

    print("\nLabelled days per participant:")
    print(f"  Minimum: {days['minimum']}")
    print(f"  Median:  {days['median']:.1f}")
    print(f"  Maximum: {days['maximum']}")

    print("\nParticipants by number of study intervals:")

    for interval_count, participant_count in (
        audit[
            "participants_by_number_of_intervals"
        ].items()
    ):
        print(
            f"  {interval_count} interval(s): "
            f"{participant_count} participant(s)"
        )

    print("\nPer-participant modality coverage:")

    for modality, values in (
        audit["modality_coverage"].items()
    ):
        print(f"\n  {modality}")
        print(
            "    Minimum: "
            f"{values['minimum_percent']:.2f}%"
        )
        print(
            "    Median:  "
            f"{values['median_percent']:.2f}%"
        )
        print(
            "    Maximum: "
            f"{values['maximum_percent']:.2f}%"
        )
        print(
            "    Participants below 50% coverage: "
            f"{values['participants_below_50_percent']}"
        )


def print_split_summary(
    split_name: str,
    summary: dict[str, Any],
) -> None:
    print("\n" + "-" * 76)
    print(split_name.upper())
    print("-" * 76)

    print(
        f"Participants: {summary['participants']}"
    )
    print(f"Participant-days: {summary['days']}")

    print("\nPhase distribution:")

    for phase in PHASES:
        count = summary["phase_counts"][phase]
        percentage = (
            summary["phase_percentages"][phase]
        )

        print(
            f"  {phase:12s}: "
            f"{count:4d} ({percentage:6.2f}%)"
        )

    print("\nModality coverage:")

    for modality in MODALITIES:
        percentage = (
            summary[
                "modality_coverage_percent"
            ][modality]
        )

        print(
            f"  {modality:20s}: "
            f"{percentage:6.2f}%"
        )

    print("\nStudy interval counts:")

    for interval, count in (
        summary["study_interval_counts"].items()
    ):
        print(
            f"  Interval {interval}: {count}"
        )

    print(
        "\nDays with all modalities: "
        f"{summary['days_with_all_modalities']}"
    )

    print(
        "Days with no modalities: "
        f"{summary['days_with_no_modalities']}"
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
    print("MOSAIC-PHASE PARTICIPANT SPLIT")
    print("=" * 76)

    master = load_master()

    participant_audit = (
        participant_coverage_audit(master)
    )

    print_participant_audit(
        participant_audit
    )

    arrays = create_participant_arrays(master)

    best_split, best_score = (
        search_balanced_split(arrays)
    )

    participant_ids = arrays["participant_ids"]

    assignment: dict[str, list[str]] = {}

    split_map: dict[str, str] = {}

    for split_name, indices in best_split.items():
        selected_ids = [
            str(participant_ids[index])
            for index in indices
        ]

        assignment[split_name] = selected_ids

        for participant_id in selected_ids:
            split_map[participant_id] = split_name

    master["split"] = (
        master["id"]
        .astype(str)
        .map(split_map)
    )

    if master["split"].isna().any():
        raise ValueError(
            "Some participant rows did not receive "
            "a split assignment."
        )

    leakage_check = (
        master.groupby("id")["split"]
        .nunique()
    )

    participants_in_multiple_splits = int(
        (leakage_check > 1).sum()
    )

    if participants_in_multiple_splits:
        raise ValueError(
            f"{participants_in_multiple_splits} "
            "participants appear in multiple splits."
        )

    split_summaries = {
        split_name: summarize_split(
            master,
            split_name,
        )
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    }

    print("\n" + "=" * 76)
    print("BALANCED SPLIT FOUND")
    print("=" * 76)

    print(f"Search seed: {RANDOM_SEED}")
    print(
        f"Search iterations: "
        f"{SEARCH_ITERATIONS:,}"
    )
    print(f"Balance score: {best_score:.8f}")
    print(
        "Participants appearing in multiple splits: "
        f"{participants_in_multiple_splits}"
    )

    for split_name, summary in (
        split_summaries.items()
    ):
        print_split_summary(
            split_name,
            summary,
        )

    master.to_parquet(
        OUTPUT_MASTER_PATH,
        index=False,
    )

    assignment_payload = {
        "random_seed": RANDOM_SEED,
        "search_iterations": SEARCH_ITERATIONS,
        "balance_score": best_score,
        "splits": assignment,
    }

    with SPLIT_ASSIGNMENT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            assignment_payload,
            file,
            indent=2,
            default=json_safe,
        )

    report_payload = {
        "participant_audit": participant_audit,
        "balance_score": best_score,
        "participants_in_multiple_splits": (
            participants_in_multiple_splits
        ),
        "split_summaries": split_summaries,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report_payload,
            file,
            indent=2,
            default=json_safe,
        )

    print("\n" + "=" * 76)
    print("FILES SAVED")
    print("=" * 76)

    print(
        "\nRestricted master table with split:"
    )
    print(OUTPUT_MASTER_PATH)

    print(
        "\nRestricted participant assignments:"
    )
    print(SPLIT_ASSIGNMENT_PATH)

    print("\nSafe aggregate report:")
    print(REPORT_PATH)

    print(
        "\nDo not publish the master table or "
        "participant assignment file."
    )


if __name__ == "__main__":
    main()