from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"
APP_ASSETS_DIR = PROJECT_DIR / "app_assets"

FINAL_REPORT_PATH = (
    REPORTS_DIR
    / "checkpoint8_final_test_report.json"
)

SUMMARY_PATH = (
    APP_ASSETS_DIR
    / "final_benchmark.json"
)

ROUTE_TABLE_PATH = (
    APP_ASSETS_DIR
    / "route_benchmark.csv"
)

INTERVAL_TABLE_PATH = (
    APP_ASSETS_DIR
    / "interval_benchmark.csv"
)

PHASE_TABLE_PATH = (
    APP_ASSETS_DIR
    / "phase_benchmark.csv"
)

CLAIMS_PATH = (
    APP_ASSETS_DIR
    / "approved_claims.md"
)


def load_final_report() -> dict[str, Any]:
    if not FINAL_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Final test report not found: "
            f"{FINAL_REPORT_PATH}"
        )

    with FINAL_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    if not report.get("test_evaluated"):
        raise ValueError(
            "The report does not contain a completed "
            "held-out test evaluation."
        )

    if not report.get(
        "policy_frozen_before_test"
    ):
        raise ValueError(
            "The report does not confirm that the "
            "policy was frozen before testing."
        )

    return report


def make_route_table(
    report: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for route, metrics in (
        report["by_route"].items()
    ):
        rows.append(
            {
                "Sensor route": route,
                "Participant-days": (
                    metrics["rows"]
                ),
                "Decision coverage": (
                    metrics[
                        "decision_coverage"
                    ]
                ),
                "Pair-set coverage": (
                    metrics[
                        "pair_set_coverage"
                    ]
                ),
                "No-call rate": (
                    metrics["no_call_rate"]
                ),
            }
        )

    return pd.DataFrame(rows)


def make_interval_table(
    report: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for interval, metrics in (
        report["by_interval"].items()
    ):
        rows.append(
            {
                "Study interval": str(interval),
                "Participant-days": (
                    metrics["rows"]
                ),
                "Sensor coverage": (
                    metrics[
                        "sensor_coverage"
                    ]
                ),
                "Decision coverage": (
                    metrics[
                        "decision_coverage"
                    ]
                ),
                "Pair-set coverage": (
                    metrics[
                        "pair_set_coverage"
                    ]
                ),
                "No-call rate": (
                    metrics["no_call_rate"]
                ),
                "Unrestricted top-1 balanced accuracy": (
                    metrics[
                        "unrestricted_top1_balanced_accuracy"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def make_phase_table(
    report: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for phase, metrics in (
        report["by_phase"].items()
    ):
        # Phase-specific balanced accuracy is intentionally
        # excluded because each subset has only one true class.
        rows.append(
            {
                "True phase": phase,
                "Participant-days": (
                    metrics["rows"]
                ),
                "Decision coverage": (
                    metrics[
                        "decision_coverage"
                    ]
                ),
                "Pair-set coverage": (
                    metrics[
                        "pair_set_coverage"
                    ]
                ),
                "No-call rate": (
                    metrics["no_call_rate"]
                ),
            }
        )

    return pd.DataFrame(rows)


def create_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    overall = report["overall"]
    policy = report["frozen_policy"]

    return {
        "project_name": "MOSAIC-Phase",
        "display_name": "PhaseWise",
        "task": (
            "Four-phase menstrual-phase estimation "
            "from passive wearable signals for unseen "
            "participants."
        ),
        "research_only": True,
        "medical_disclaimer": (
            "Research prototype only. It must not be "
            "used for diagnosis, fertility decisions, "
            "contraception, or treatment."
        ),
        "evaluation": {
            "held_out_participants": 7,
            "held_out_participant_days": (
                report[
                    "held_out_test_rows"
                ]
            ),
            "participant_level_split": True,
            "policy_frozen_before_test": True,
            "test_evaluated_once": True,
        },
        "model": {
            "predictive_modalities": [
                "Resting heart rate",
                "Wrist temperature",
            ],
            "both_sensor_router": (
                "Equal probability fusion"
            ),
            "rhr_only_router": (
                "Resting-heart-rate expert"
            ),
            "temperature_only_router": (
                "Temperature expert"
            ),
            "neither_router": "No-call",
            "probability_temperature": (
                policy[
                    "probability_temperature"
                ]
            ),
            "top_two_threshold": (
                policy["pair_threshold"]
            ),
            "singleton_outputs_enabled": False,
        },
        "overall_test_results": {
            "sensor_coverage": (
                overall["sensor_coverage"]
            ),
            "unrestricted_top1_accuracy": (
                overall[
                    "unrestricted_top1_accuracy"
                ]
            ),
            "unrestricted_top1_balanced_accuracy": (
                overall[
                    "unrestricted_top1_balanced_accuracy"
                ]
            ),
            "unrestricted_top1_macro_f1": (
                overall[
                    "unrestricted_top1_macro_f1"
                ]
            ),
            "unrestricted_top2_coverage": (
                overall[
                    "unrestricted_top2_coverage"
                ]
            ),
            "final_decision_coverage": (
                overall[
                    "decision_coverage"
                ]
            ),
            "final_pair_set_coverage": (
                overall[
                    "pair_set_coverage"
                ]
            ),
            "final_no_call_rate": (
                overall["no_call_rate"]
            ),
            "no_calls_missing_sensors": (
                overall["no_sensor_rows"]
            ),
            "no_calls_insufficient_evidence": (
                overall[
                    "low_evidence_no_call_rows"
                ]
            ),
        },
        "approved_language": {
            "primary_result": (
                "On unseen participants, the frozen "
                "policy returned a two-phase plausible "
                "set on 44.84% of participant-days. "
                "The true phase was contained in that "
                "set for 66.51% of those decisions."
            ),
            "abstention_result": (
                "The system returned no-call on 55.16% "
                "of participant-days when predictive "
                "evidence was unavailable or insufficient."
            ),
            "top1_context": (
                "The unrestricted router achieved "
                "34.76% balanced accuracy, compared "
                "with a four-class random reference "
                "of approximately 25%."
            ),
            "limitation": (
                "Performance varied across phases and "
                "study intervals. The output is not a "
                "clinical or fertility prediction."
            ),
        },
        "prohibited_claims": [
            "Clinically validated",
            "Reliable fertility detection",
            "Ovulation confirmation",
            "Diagnostic accuracy",
            "Safe for contraception decisions",
            "Guaranteed uncertainty coverage",
        ],
    }


def write_approved_claims(
    summary: dict[str, Any],
) -> None:
    language = summary[
        "approved_language"
    ]

    content = f"""# Approved MOSAIC-Phase claims

## Primary held-out result

{language["primary_result"]}

## Abstention result

{language["abstention_result"]}

## Unrestricted ranking context

{language["top1_context"]}

## Required limitation

{language["limitation"]}

## Required disclaimer

{summary["medical_disclaimer"]}

## Claims that must not be used

- Clinically validated
- Reliable fertility detection
- Ovulation confirmation
- Diagnostic accuracy
- Safe for contraception decisions
- Guaranteed uncertainty coverage
"""

    CLAIMS_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    APP_ASSETS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MOSAIC-PHASE PUBLIC APP ASSET BUILDER")
    print("=" * 76)

    report = load_final_report()

    summary = create_summary(report)

    route_table = make_route_table(
        report
    )

    interval_table = make_interval_table(
        report
    )

    phase_table = make_phase_table(
        report
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    route_table.to_csv(
        ROUTE_TABLE_PATH,
        index=False,
    )

    interval_table.to_csv(
        INTERVAL_TABLE_PATH,
        index=False,
    )

    phase_table.to_csv(
        PHASE_TABLE_PATH,
        index=False,
    )

    write_approved_claims(summary)

    print("\nPublic-safe files created:")

    for path in [
        SUMMARY_PATH,
        ROUTE_TABLE_PATH,
        INTERVAL_TABLE_PATH,
        PHASE_TABLE_PATH,
        CLAIMS_PATH,
    ]:
        print(f"  {path}")

    print("\nAggregate benchmark summary:")

    overall = summary[
        "overall_test_results"
    ]

    print(
        "  Held-out participant-days: "
        f"{summary['evaluation']['held_out_participant_days']}"
    )

    print(
        "  Sensor coverage: "
        f"{overall['sensor_coverage']:.4f}"
    )

    print(
        "  Top-1 balanced accuracy: "
        f"{overall['unrestricted_top1_balanced_accuracy']:.4f}"
    )

    print(
        "  Final decision coverage: "
        f"{overall['final_decision_coverage']:.4f}"
    )

    print(
        "  Final pair-set coverage: "
        f"{overall['final_pair_set_coverage']:.4f}"
    )

    print(
        "  Final no-call rate: "
        f"{overall['final_no_call_rate']:.4f}"
    )

    print(
        "\nNo participant IDs, daily records or "
        "individual predictions were exported."
    )


if __name__ == "__main__":
    main()