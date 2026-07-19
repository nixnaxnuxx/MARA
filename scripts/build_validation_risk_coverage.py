from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_DIR
    / "reports"
    / "checkpoint7a_validation_prediction_sets.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "app_assets"
    / "validation_risk_coverage.csv"
)

METADATA_PATH = (
    PROJECT_DIR
    / "app_assets"
    / "validation_risk_coverage_metadata.json"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PROBABILITY_COLUMNS = [
    "calibrated_probability_menstrual",
    "calibrated_probability_follicular",
    "calibrated_probability_fertility",
    "calibrated_probability_luteal",
]

FROZEN_THRESHOLD = 0.62


def parse_boolean(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def canonicalize_phase(
    value: object,
) -> str:
    normalized = str(
        value
    ).strip().lower()

    mapping = {
        "menstrual": "Menstrual",
        "follicular": "Follicular",
        "fertility": "Fertility",
        "luteal": "Luteal",
    }

    if normalized not in mapping:
        raise ValueError(
            "Unexpected phase label: "
            f"{value!r}"
        )

    return mapping[normalized]


def build_threshold_grid() -> np.ndarray:
    thresholds = np.arange(
        0.50,
        0.951,
        0.01,
    )

    thresholds = np.append(
        thresholds,
        FROZEN_THRESHOLD,
    )

    return np.unique(
        np.round(
            thresholds,
            2,
        )
    )


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Validation prediction file "
            f"not found: {SOURCE_PATH}"
        )

    dataframe = pd.read_csv(
        SOURCE_PATH
    )

    required_columns = {
        "phase",
        "has_rhr",
        "has_temperature",
        "sensor_route",
        *PROBABILITY_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        formatted = ", ".join(
            sorted(
                missing_columns
            )
        )

        raise ValueError(
            "Required columns are missing: "
            f"{formatted}"
        )

    if dataframe.empty:
        raise ValueError(
            "The validation prediction file "
            "contains no rows."
        )

    true_phases = (
        dataframe["phase"]
        .map(
            canonicalize_phase
        )
    )

    has_rhr = parse_boolean(
        dataframe["has_rhr"]
    )

    has_temperature = (
        parse_boolean(
            dataframe[
                "has_temperature"
            ]
        )
    )

    sensor_available = (
        has_rhr
        | has_temperature
    ).to_numpy(
        dtype=bool
    )

    probabilities_raw = (
        dataframe[
            PROBABILITY_COLUMNS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    finite_probability_rows = (
        np.isfinite(
            probabilities_raw
        ).all(
            axis=1
        )
    )

    safe_probability_values = np.where(
        np.isfinite(
            probabilities_raw
        ),
        probabilities_raw,
        0.0,
    )

    positive_probability_rows = (
        safe_probability_values.sum(
            axis=1
        )
        > 0
    )

    valid_probability_rows = (
        sensor_available
        & finite_probability_rows
        & positive_probability_rows
    )

    invalid_sensor_rows = (
        sensor_available
        & (
            ~finite_probability_rows
            | ~positive_probability_rows
        )
    )

    if invalid_sensor_rows.any():
        invalid_count = int(
            invalid_sensor_rows.sum()
        )

        invalid_routes = (
            dataframe.loc[
                invalid_sensor_rows,
                "sensor_route",
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        raise ValueError(
            "Some validation rows have at least one "
            "sensor available but do not contain valid "
            "calibrated probabilities. "
            f"Count: {invalid_count}. "
            f"Routes: {invalid_routes}"
        )

    # Rows where neither sensor is available may
    # legitimately have missing probabilities.
    probabilities = np.full(
        probabilities_raw.shape,
        np.nan,
        dtype=float,
    )

    valid_probabilities = np.clip(
        probabilities_raw[
            valid_probability_rows
        ],
        0.0,
        None,
    )

    valid_row_sums = (
        valid_probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    valid_probabilities = (
        valid_probabilities
        / valid_row_sums
    )

    probabilities[
        valid_probability_rows
    ] = valid_probabilities

    valid_descending_order = np.argsort(
        -valid_probabilities,
        axis=1,
    )

    valid_top_two_indices = (
        valid_descending_order[
            :,
            :2,
        ]
    )

    top_two_indices = np.full(
        (
            len(dataframe),
            2,
        ),
        -1,
        dtype=int,
    )

    top_two_indices[
        valid_probability_rows
    ] = valid_top_two_indices

    top_two_probability = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    top_two_probability[
        valid_probability_rows
    ] = (
        np.take_along_axis(
            valid_probabilities,
            valid_top_two_indices,
            axis=1,
        )
        .sum(
            axis=1
        )
    )

    phase_to_index = {
        phase: index
        for index, phase
        in enumerate(PHASES)
    }

    true_phase_indices = (
        true_phases
        .map(
            phase_to_index
        )
        .to_numpy(
            dtype=int
        )
    )

    truth_in_top_two = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    truth_in_top_two[
        valid_probability_rows
    ] = np.any(
        valid_top_two_indices
        == true_phase_indices[
            valid_probability_rows
        ][
            :,
            None,
        ],
        axis=1,
    )

    total_days = len(
        dataframe
    )

    missing_sensor_days = int(
        (
            ~sensor_available
        ).sum()
    )

    sensor_available_days = int(
        sensor_available.sum()
    )

    unrestricted_top_two_coverage = float(
        truth_in_top_two[
            valid_probability_rows
        ].mean()
    )

    rows: list[
        dict[str, object]
    ] = []

    for threshold in build_threshold_grid():
        decision_made = (
            sensor_available
            & (
                top_two_probability
                >= threshold
            )
        )

        insufficient_evidence = (
            sensor_available
            & (
                top_two_probability
                < threshold
            )
        )

        decided_days = int(
            decision_made.sum()
        )

        insufficient_evidence_days = int(
            insufficient_evidence.sum()
        )

        decision_coverage = (
            decided_days
            / total_days
        )

        no_call_rate = (
            1.0
            - decision_coverage
        )

        if decided_days > 0:
            pair_set_coverage = float(
                truth_in_top_two[
                    decision_made
                ].mean()
            )

            selective_risk = (
                1.0
                - pair_set_coverage
            )
        else:
            pair_set_coverage = np.nan
            selective_risk = np.nan

        rows.append(
            {
                "Threshold": float(
                    threshold
                ),
                "Total validation days": (
                    total_days
                ),
                "Sensor-available days": (
                    sensor_available_days
                ),
                "Decided days": decided_days,
                "Missing-sensor no-call days": (
                    missing_sensor_days
                ),
                (
                    "Insufficient-evidence "
                    "no-call days"
                ): (
                    insufficient_evidence_days
                ),
                "Decision coverage": (
                    decision_coverage
                ),
                "Pair-set coverage": (
                    pair_set_coverage
                ),
                "Selective risk": (
                    selective_risk
                ),
                "No-call rate": (
                    no_call_rate
                ),
                (
                    "Missing-sensor "
                    "no-call rate"
                ): (
                    missing_sensor_days
                    / total_days
                ),
                (
                    "Insufficient-evidence "
                    "no-call rate"
                ): (
                    insufficient_evidence_days
                    / total_days
                ),
                (
                    "Unrestricted top-two "
                    "coverage"
                ): (
                    unrestricted_top_two_coverage
                ),
                "Frozen policy": bool(
                    np.isclose(
                        threshold,
                        FROZEN_THRESHOLD,
                    )
                ),
            }
        )

    output = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    metadata = {
        "display_name": (
            "Validation risk-coverage curve"
        ),
        "source_split": "validation",
        "source_file": (
            SOURCE_PATH.name
        ),
        "total_validation_days": (
            total_days
        ),
        "sensor_available_days": (
            sensor_available_days
        ),
        "missing_sensor_days": (
            missing_sensor_days
        ),
        "frozen_threshold": (
            FROZEN_THRESHOLD
        ),
        "threshold_minimum": float(
            output["Threshold"].min()
        ),
        "threshold_maximum": float(
            output["Threshold"].max()
        ),
        "threshold_count": int(
            len(output)
        ),
        "contains_participant_rows": False,
        "contains_identifiers": False,
        "interpretation": (
            "Exploratory aggregate derived "
            "from validation predictions. "
            "The held-out test set was not "
            "used to generate this curve."
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    frozen_row = output.loc[
        output["Frozen policy"]
    ]

    print(
        "Validation risk-coverage "
        "aggregation complete."
    )

    print(
        f"Source rows: {total_days}"
    )

    print(
        "Sensor-available rows: "
        f"{sensor_available_days}"
    )

    print(
        "Missing-sensor rows: "
        f"{missing_sensor_days}"
    )

    print(
        f"Threshold rows: {len(output)}"
    )

    print(
        "\nFrozen threshold summary:"
    )

    print(
        frozen_row[
            [
                "Threshold",
                "Decided days",
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
                (
                    "Insufficient-evidence "
                    "no-call days"
                ),
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCreated:"
    )

    print(
        OUTPUT_PATH.relative_to(
            PROJECT_DIR
        )
    )

    print(
        METADATA_PATH.relative_to(
            PROJECT_DIR
        )
    )

    print(
        "\nNo participant identifiers or "
        "daily prediction rows were exported."
    )


if __name__ == "__main__":
    main()