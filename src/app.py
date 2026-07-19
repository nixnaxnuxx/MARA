from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ai_mara_chat import (
    render_ai_mara_chat,
)


from judge_mode import (
    render_judge_mode,
)

from robustness_dashboard import (
    render_robustness_dashboard,
)

from risk_coverage_explorer import (
    render_validation_risk_coverage_explorer,
)

from synthetic_journey import (
    build_synthetic_journey,
    render_synthetic_journey,
)

from ui_theme import (
    apply_mara_theme,
    render_mara_footer,
    render_mara_hero,
    render_mara_metric_strip,
)

from interactive_components import (
    apply_interactive_styles,
    render_decision_explanation,
    render_phase_wheel,
    render_routing_diagram,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_ASSETS_DIR = PROJECT_DIR / "app_assets"

SUMMARY_PATH = APP_ASSETS_DIR / "final_benchmark.json"
ROUTE_TABLE_PATH = APP_ASSETS_DIR / "route_benchmark.csv"
INTERVAL_TABLE_PATH = APP_ASSETS_DIR / "interval_benchmark.csv"
PHASE_TABLE_PATH = APP_ASSETS_DIR / "phase_benchmark.csv"

NAVIGATION_KEY = "mara_main_navigation"

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

NAVIGATION_KEY = "mara_main_navigation"

PAGE_OPTIONS = [
    "Product Overview",
    "Held-out Benchmark",
    "Sensor-Routing Lab",
    "Robustness Audit",
    "MARA AI Guide",
]

QUICK_NAV_PAGES = PAGE_OPTIONS.copy()

PAGE_DESCRIPTIONS = {
    "Product Overview": (
        "View the product summary and demonstration."
    ),
    "Held-out Benchmark": (
        "Review frozen performance and validation behaviour."
    ),
    "Sensor-Routing Lab": (
        "Explore missing sensors, routing and probabilities."
    ),
    "Robustness Audit": (
        "Inspect phase, route and missing-modality robustness."
    ),
    "MARA AI Guide": (
        "Ask questions about MARA and its public results."
    ),
}

PAGE_KEYS = {
    "Product Overview": "product_overview",
    "Held-out Benchmark": "held_out_benchmark",
    "Sensor-Routing Lab": "sensor_routing_lab",
    "Robustness Audit": "robustness_audit",
    "MARA AI Guide": "mara_ai_guide",
}
MARA_PINK = "#D65D8B"
MARA_TEAL = "#4F8F88"
MARA_LAVENDER = "#8B7BBE"
MARA_CORAL = "#F08A7C"
MARA_AMBER = "#D99A45"
MARA_BERRY = "#71324F"

st.set_page_config(
    page_title="MARA | Missing-Modality-Aware Routing and Abstention for Menstrual Phase Estimation",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


apply_mara_theme()
apply_interactive_styles()

@st.cache_data
def load_public_assets() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        SUMMARY_PATH,
        ROUTE_TABLE_PATH,
        INTERVAL_TABLE_PATH,
        PHASE_TABLE_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required public application assets "
            f"were not found:\n{formatted}"
        )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    route_table = pd.read_csv(
        ROUTE_TABLE_PATH
    )

    interval_table = pd.read_csv(
        INTERVAL_TABLE_PATH
    )

    phase_table = pd.read_csv(
        PHASE_TABLE_PATH
    )

    return (
        summary,
        route_table,
        interval_table,
        phase_table,
    )


def percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{100.0 * float(value):.2f}%"


def format_table_percentages(
    dataframe: pd.DataFrame,
    percentage_columns: list[str],
) -> pd.DataFrame:
    output = dataframe.copy()

    for column in percentage_columns:
        if column not in output.columns:
            continue

        output[column] = output[column].apply(
            percent
        )

    return output


def normalize_probabilities(
    values: list[float] | np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        values,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        None,
    )

    total = probabilities.sum()

    if total <= 0:
        return np.full(
            len(PHASES),
            1.0 / len(PHASES),
            dtype=float,
        )

    return probabilities / total


def temperature_scale(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    probabilities = normalize_probabilities(
        probabilities
    )

    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    transformed = np.power(
        clipped,
        1.0 / temperature,
    )

    return transformed / transformed.sum()


def probability_editor(
    title: str,
    defaults: dict[str, float],
    key_prefix: str,
) -> np.ndarray:
    st.markdown(f"#### {title}")

    columns = st.columns(4)
    values = []

    for column, phase in zip(
        columns,
        PHASES,
    ):
        with column:
            value = st.slider(
                phase,
                min_value=0.0,
                max_value=1.0,
                value=float(
                    defaults[phase]
                ),
                step=0.01,
                key=(
                    f"{key_prefix}_"
                    f"{phase.lower()}"
                ),
            )

            values.append(value)

    raw_sum = float(sum(values))

    if not np.isclose(
        raw_sum,
        1.0,
        atol=0.001,
    ):
        st.caption(
            f"Entered values sum to {raw_sum:.2f}. "
            "They are normalized automatically."
        )

    return normalize_probabilities(values)


def get_route_name(
    rhr_enabled: bool,
    temperature_enabled: bool,
) -> str:
    if rhr_enabled and temperature_enabled:
        return "Equal fusion"

    if rhr_enabled:
        return "RHR expert"

    if temperature_enabled:
        return "Temperature expert"

    return "Neither sensor"


def get_route_description(
    route_name: str,
) -> str:
    descriptions = {
        "Equal fusion": (
            "The resting-heart-rate and "
            "temperature expert probabilities "
            "are averaged equally."
        ),
        "RHR expert": (
            "Only the resting-heart-rate "
            "expert is active."
        ),
        "Temperature expert": (
            "Only the wrist-temperature "
            "expert is active."
        ),
        "Neither sensor": (
            "No predictive expert is available. "
            "The system returns no-call."
        ),
    }

    return descriptions[route_name]


def route_benchmark_row(
    route_table: pd.DataFrame,
    route_name: str,
) -> pd.Series | None:
    matches = route_table[
        route_table["Sensor route"]
        == route_name
    ]

    if matches.empty:
        return None

    return matches.iloc[0]

def toggle_more_information() -> None:
    current_value = bool(
        st.session_state.get(
            "show_more_information",
            False,
        )
    )

    st.session_state[
        "show_more_information"
    ] = not current_value



def navigate_to_page(
    page_name: str,
) -> None:
    st.session_state[
        NAVIGATION_KEY
    ] = page_name


def render_page_shortcuts(
    current_page: str,
) -> None:
    st.html(
        """
<section class="quick-navigation-header">
    <div class="quick-navigation-kicker">
        Product navigation
    </div>

    <div class="quick-navigation-title">
        Explore MARA
    </div>

    <div class="quick-navigation-copy">
        Select a section to continue exploring
        the product demonstration.
    </div>
</section>
"""
    )

    navigation_columns = st.columns(5)

    for column, page_name in zip(
        navigation_columns,
        QUICK_NAV_PAGES,
    ):
        is_current_page = (
            current_page == page_name
        )

        with column:
            st.button(
                page_name,
                key=(
                    "quick_nav_"
                    + PAGE_KEYS[
                        page_name
                    ]
                ),
                type=(
                    "primary"
                    if is_current_page
                    else "secondary"
                ),
                width="stretch",
                disabled=is_current_page,
                on_click=navigate_to_page,
                args=(
                    page_name,
                ),
            )

def build_downloadable_sensor_lab_report(
    *,
    report_table: pd.DataFrame,
    probability_table: pd.DataFrame | None,
    probability_chart: alt.Chart | None,
) -> str:
    summary_html = report_table.to_html(
        index=False,
        border=0,
        classes="report-table",
        escape=True,
    )

    if probability_table is not None:
        displayed_probability_table = (
            probability_table.copy()
        )

        displayed_probability_table[
            "Probability"
        ] = displayed_probability_table[
            "Probability"
        ].map(
            lambda value: f"{value:.2f}%"
        )

        probability_values_html = (
            displayed_probability_table.to_html(
                index=False,
                border=0,
                classes="report-table",
                escape=True,
            )
        )
    else:
        probability_values_html = """
        <p class="notice">
            No probability values are available because
            neither predictive sensor was active.
        </p>
        """

    report_css = """
        :root {
            color-scheme: light;
        }

        body {
            margin: 0;
            padding: 32px;
            background: #fff8fb;
            color: #352631;
            font-family:
                Inter,
                Arial,
                sans-serif;
        }

        main {
            max-width: 1000px;
            margin: 0 auto;
        }

        .report-header {
            padding: 28px 30px;
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    #4f2d3e,
                    #a24c76,
                    #d65d8b
                );
            color: #ffffff;
        }

        .report-header h1 {
            margin: 0;
            font-size: 2rem;
        }

        .report-header p {
            max-width: 720px;
            margin: 10px 0 0;
            line-height: 1.6;
            opacity: 0.9;
        }

        section {
            margin-top: 24px;
            padding: 24px;
            border: 1px solid #ead5df;
            border-radius: 20px;
            background: #ffffff;
            box-shadow:
                0 10px 28px
                rgba(113, 50, 79, 0.08);
        }

        h2 {
            margin-top: 0;
            color: #71324f;
            font-size: 1.25rem;
        }

        .report-table {
            width: 100%;
            border-collapse: collapse;
        }

        .report-table th {
            padding: 12px 14px;
            background: #fff0f5;
            color: #71324f;
            text-align: left;
            font-size: 0.9rem;
        }

        .report-table td {
            padding: 12px 14px;
            border-bottom: 1px solid #f0e3e9;
            font-size: 0.92rem;
        }

        .notice {
            padding: 14px 16px;
            border-radius: 12px;
            background: #fff5dc;
            color: #6d5520;
        }

        .disclaimer {
            margin-top: 24px;
            color: #7a6670;
            font-size: 0.82rem;
            line-height: 1.6;
        }
    """

    report_header = f"""
        <main>
            <header class="report-header">
                <h1>MARA Current Scenario Report</h1>

                <p>
                    Missing-Modality-Aware Routing with
                    Abstention for wearable menstrual-phase
                    estimation.
                </p>
            </header>

            <section>
                <h2>Scenario summary</h2>
                {summary_html}
            </section>
    """

    report_footer = f"""
            <section>
                <h2>Probability values</h2>
                {probability_values_html}
            </section>

            <p class="disclaimer">
                This downloadable report contains hypothetical
                demonstration values only. It is not a diagnosis,
                fertility assessment, contraception recommendation
                or clinical decision.
            </p>
        </main>
    """

    if probability_chart is None:
        return f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>MARA Current Scenario Report</title>

    <style>
        {report_css}
    </style>
</head>

<body>
    {report_header}

    <section>
        <h2>Calibrated phase probabilities</h2>

        <p class="notice">
            No probability chart is available because
            neither predictive sensor was active.
        </p>
    </section>

    {report_footer}
</body>
</html>
"""

    chart_document = probability_chart.to_html(
        fullhtml=True,
        embed_options={
            "actions": False,
        },
    )

    chart_document = chart_document.replace(
        "</head>",
        f"""
<style>
    {report_css}
</style>
</head>
""",
        1,
    )

    chart_document = chart_document.replace(
        "<body>",
        f"""
<body>
    {report_header}

    <section>
        <h2>Calibrated phase probabilities</h2>
""",
        1,
    )

    chart_document = chart_document.replace(
        "</body>",
        f"""
    </section>

    {report_footer}
</body>
""",
        1,
    )

    return chart_document

def render_sensor_lab_report(
    *,
    preset_name: str,
    route_name: str,
    rhr_enabled: bool,
    temperature_enabled: bool,
    calibrated_probabilities: (
        np.ndarray | None
    ),
    top_phase: str | None,
    top_two_phases: list[str],
    top_two_probability: float | None,
    threshold: float,
    decision_made: bool,
) -> None:
    st.divider()

    st.subheader(
        "Current scenario report"
    )

    returned_output = (
        " or ".join(top_two_phases)
        if decision_made
        else "No-call"
    )

    if decision_made:
        st.success(
            (
                "MARA returned the phase pair: "
                f"**{returned_output}**."
            )
        )
    else:
        st.warning(
            (
                "MARA returned **no-call** because "
                "the available evidence did not satisfy "
                "the frozen decision policy."
            )
        )

    report_rows = [
        {
            "Report field": "Scenario",
            "Value": preset_name,
        },
        {
            "Report field": "Active route",
            "Value": route_name,
        },
        {
            "Report field": (
                "Resting heart rate"
            ),
            "Value": (
                "Available"
                if rhr_enabled
                else "Missing"
            ),
        },
        {
            "Report field": (
                "Wrist temperature"
            ),
            "Value": (
                "Available"
                if temperature_enabled
                else "Missing"
            ),
        },
        {
            "Report field": (
                "Top-ranked phase"
            ),
            "Value": (
                top_phase
                if top_phase is not None
                else "N/A"
            ),
        },
        {
            "Report field": (
                "Selective output"
            ),
            "Value": returned_output,
        },
        {
            "Report field": (
                "Top-two evidence"
            ),
            "Value": (
                percent(
                    top_two_probability
                )
                if top_two_probability
                is not None
                else "N/A"
            ),
        },
        {
            "Report field": (
                "Decision threshold"
            ),
            "Value": percent(threshold),
        },
    ]

    report_table = pd.DataFrame(
        report_rows
    )

    st.dataframe(
        report_table,
        hide_index=True,
        use_container_width=True,
    )

    if calibrated_probabilities is None:
        st.info(
            """
            No probability chart is available because
            neither predictive sensor is active.
            """
        )

        return

    probability_values = pd.DataFrame(
        {
            "Phase": PHASES,
            "Probability": (
                calibrated_probabilities
                * 100.0
            ),
        }
    )

    chart_column, values_column = (
        st.columns(
            [
                1.45,
                1,
            ]
        )
    )

    with chart_column:
        st.markdown(
            "#### Calibrated phase probabilities"
        )

        chart_data = (
            probability_values
            .set_index("Phase")
        )

        st.bar_chart(
            chart_data,
            y="Probability",
            height=330,
        )

    with values_column:
        st.markdown(
            "#### Probability values"
        )

        displayed_values = (
            probability_values.copy()
        )

        displayed_values[
            "Probability"
        ] = displayed_values[
            "Probability"
        ].map(
            lambda value: (
                f"{value:.2f}%"
            )
        )

        st.dataframe(
            displayed_values,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        """
        This report contains hypothetical demonstration
        values only. It is not a diagnosis, fertility
        assessment or clinical recommendation.
        """
    )
            
def render_overview(
    summary: dict[str, Any],
) -> None:
    results = summary[
        "overall_test_results"
    ]

    evaluation = summary[
        "evaluation"
    ]

    st.html(
        """
<div class="mara-kicker product-overview-kicker">
    Product Overview
</div>
"""
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Held-out participants",
        evaluation[
            "held_out_participants"
        ],
        "Participant-level split",
    )

    metric_columns[1].metric(
        "Held-out days",
        evaluation[
            "held_out_participant_days"
        ],
        "Frozen test benchmark",
    )

    metric_columns[2].metric(
        "Decision coverage",
        percent(
            results[
                "final_decision_coverage"
            ]
        ),
        "Two-phase set returned",
    )

    metric_columns[3].metric(
        "Set coverage",
        percent(
            results[
                "final_pair_set_coverage"
            ]
        ),
        "Truth inside decided sets",
    )
    
    st.divider()
    
    render_judge_mode(
        summary
    )
    
    st.divider()

    show_information = bool(
        st.session_state.get(
            "show_more_information",
            False,
        )
    )

    information_button_label = (
        "Hide Information"
        if show_information
        else "More Information"
    )

    st.button(
        information_button_label,
        key="toggle_more_information_button",
        type=(
            "secondary"
            if show_information
            else "primary"
        ),
        icon="ℹ️",
        width="stretch",
        on_click=toggle_more_information,
    )

    if not show_information:
        return

    with st.container(
        border=True
    ):
        st.subheader(
            "What the system does"
        )

        method_columns = st.columns(4)

        with method_columns[0]:
            st.html(
                """
<div class="method-card">
    <b>1. Passive signals</b>
    <br><br>
    Resting heart rate and nightly wrist
    temperature form the predictive inputs.
</div>
"""
            )

        with method_columns[1]:
            st.html(
                """
<div class="method-card">
    <b>2. Sensor routing</b>
    <br><br>
    Separate experts are activated according
    to which sensor families are available.
</div>
"""
            )

        with method_columns[2]:
            st.html(
                """
<div class="method-card">
    <b>3. Probability fusion</b>
    <br><br>
    RHR and temperature probabilities are
    fused when both modalities exist.
</div>
"""
            )

        with method_columns[3]:
            st.html(
                """
<div class="method-card">
    <b>4. Selective output</b>
    <br><br>
    The system returns two plausible phases
    or abstains when evidence is insufficient.
</div>
"""
            )

        st.subheader(
            "Target phases"
        )

        phase_columns = st.columns(4)

        for column, phase in zip(
            phase_columns,
            PHASES,
        ):
            column.info(
                phase
            )

        st.subheader(
            "Frozen held-out result"
        )

        st.success(
            summary[
                "approved_language"
            ][
                "primary_result"
            ]
        )

        st.info(
            summary[
                "approved_language"
            ][
                "abstention_result"
            ]
        )

        st.caption(
            summary[
                "approved_language"
            ][
                "limitation"
            ]
        )


def render_benchmark(
    summary: dict[str, Any],
    route_table: pd.DataFrame,
) -> None:
    results = summary[
        "overall_test_results"
    ]

    st.header("Held-out benchmark")

    st.write(
        """
        All metrics below come from seven participants
        who were not used to fit the models or select
        the output policy.
        """
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Sensor coverage",
        percent(
            results["sensor_coverage"]
        ),
    )

    metric_columns[1].metric(
        "Top-1 balanced accuracy",
        percent(
            results[
                "unrestricted_top1_balanced_accuracy"
            ]
        ),
    )

    metric_columns[2].metric(
        "Decision coverage",
        percent(
            results[
                "final_decision_coverage"
            ]
        ),
    )

    metric_columns[3].metric(
        "Two-phase set coverage",
        percent(
            results[
                "final_pair_set_coverage"
            ]
        ),
    )

    metric_columns[4].metric(
        "No-call rate",
        percent(
            results[
                "final_no_call_rate"
            ]
        ),
    )

    st.caption(
        """
        Decision coverage is the fraction of all held-out
        days receiving a two-phase set. Set coverage is
        calculated only among those decided days.
        """
    )

    st.divider()
    
    render_validation_risk_coverage_explorer()
    
    st.divider()

    st.subheader("Overall risk–coverage profile")

    chart_data = pd.DataFrame(
        {
            "Metric": [
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
                "Unrestricted top-2 coverage",
            ],
            "Percent": [
                100
                * results[
                    "final_decision_coverage"
                ],
                100
                * results[
                    "final_pair_set_coverage"
                ],
                100
                * results[
                    "final_no_call_rate"
                ],
                100
                * results[
                    "unrestricted_top2_coverage"
                ],
            ],
        }
    ).set_index("Metric")

    st.bar_chart(
        chart_data,
        y="Percent",
        color=MARA_PINK,
        horizontal=True,
        height=330,
    )
    st.subheader("Performance by sensor route")

    route_chart = (
        route_table
        .set_index("Sensor route")[
            [
                "Decision coverage",
                "Pair-set coverage",
            ]
        ]
        * 100
    )

    st.bar_chart(
        route_chart,
        y=[
            "Decision coverage",
            "Pair-set coverage",
        ],
        color=[
            MARA_PINK,
            MARA_TEAL,
        ],
        stack=False,
        height=360,
    )

    displayed_routes = (
        format_table_percentages(
            route_table,
            [
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
            ],
        )
    )

    st.dataframe(
        displayed_routes,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        """
        Route-specific results can be unstable when a route
        contains few observations. They describe the held-out
        cohort and are not clinical performance guarantees.
        """
    )


def render_sensor_lab(
    summary: dict[str, Any],
    route_table: pd.DataFrame,
) -> None:
    model = summary["model"]

    st.header("Sensor-routing laboratory")

    st.write(
        """
        Toggle predictive sensors to inspect the routing
        rule. The probability sandbox below uses hypothetical
        expert outputs—it does not infer a phase from personal
        health measurements.
        """
    )

    control_column, route_column = (
        st.columns([1, 1.4])
    )

    with control_column:
        st.subheader(
            "Available sensors"
        )
    
        st.caption(
            """
            Select which wearable signals are available.
            MARA updates the active route immediately.
            """
        )
    
        if (
            "mara_rhr_available"
            not in st.session_state
        ):
            st.session_state[
                "mara_rhr_available"
            ] = True
    
        if (
            "mara_temperature_available"
            not in st.session_state
        ):
            st.session_state[
                "mara_temperature_available"
            ] = True
    
        with st.container(
            border=True
        ):
            rhr_enabled = st.toggle(
                "Resting heart rate",
                key="mara_rhr_available",
            )
    
            if rhr_enabled:
                st.success(
                    "RHR expert available"
                )
            else:
                st.warning(
                    "RHR expert unavailable"
                )
    
        with st.container(
            border=True
        ):
            temperature_enabled = st.toggle(
                "Wrist temperature",
                key=(
                    "mara_temperature_available"
                ),
            )
    
            if temperature_enabled:
                st.success(
                    "Temperature expert available"
                )
            else:
                st.warning(
                    "Temperature expert unavailable"
                )
    
        st.caption(
            """
            Sleep and activity were evaluated but did not
            provide sufficiently consistent predictive
            evidence to drive the frozen router.
            """
        )
    
    route_name = get_route_name(
        rhr_enabled,
        temperature_enabled,
    )

    with route_column:
        st.subheader("Active route")

        if route_name == "Equal fusion":
            st.success(route_name)
        elif route_name == "Neither sensor":
            st.error(route_name)
        else:
            st.warning(route_name)

        st.write(
            get_route_description(
                route_name
            )
        )

        benchmark_row = route_benchmark_row(
            route_table,
            route_name,
        )

        if benchmark_row is not None:
            route_metrics = st.columns(3)

            route_metrics[0].metric(
                "Held-out days",
                int(
                    benchmark_row[
                        "Participant-days"
                    ]
                ),
            )

            route_metrics[1].metric(
                "Decision coverage",
                percent(
                    benchmark_row[
                        "Decision coverage"
                    ]
                ),
            )

            route_metrics[2].metric(
                "Set coverage",
                percent(
                    benchmark_row[
                        "Pair-set coverage"
                    ]
                ),
            )
    render_routing_diagram(
        rhr_enabled=rhr_enabled,
        temperature_enabled=temperature_enabled,
        route_name=route_name,
    )
    
    st.divider()
    
    journey = build_synthetic_journey(
        probability_temperature=float(
            model[
                "probability_temperature"
            ]
        ),
        threshold=float(
            model[
                "top_two_threshold"
            ]
        ),
    )
    
    selected_journey_row = (
        render_synthetic_journey(
            journey
        )
    )
    
    journey_probabilities = (
        selected_journey_row[
            PHASES
        ]
        .to_numpy(
            dtype=float
        )
    )
    
    journey_top_two_phases = [
        str(
            selected_journey_row[
                "Primary phase"
            ]
        ),
        str(
            selected_journey_row[
                "Secondary phase"
            ]
        ),
    ]
    
    journey_decision_made = bool(
        selected_journey_row[
            "Decision"
        ]
    )
    
    journey_rhr_enabled = bool(
        selected_journey_row[
            "RHR available"
        ]
    )
    
    journey_temperature_enabled = bool(
        selected_journey_row[
            "Temperature available"
        ]
    )
    
    journey_route_name = str(
        selected_journey_row[
            "Route"
        ]
    )
    
    journey_top_two_probability = float(
        selected_journey_row[
            "Top-two evidence"
        ]
    )
    
    journey_threshold = float(
        model[
            "top_two_threshold"
        ]
    )
    
    st.markdown(
        "#### Selected-day MARA interpretation"
    )
    
    journey_wheel_column, journey_explanation_column = (
        st.columns(
            [
                1,
                1.15,
            ]
        )
    )
    
    with journey_wheel_column:
        render_phase_wheel(
            probabilities=journey_probabilities,
            top_two_phases=journey_top_two_phases,
            decision_made=journey_decision_made,
        )
    
    with journey_explanation_column:
        render_decision_explanation(
            route_name=journey_route_name,
            top_two_phases=journey_top_two_phases,
            top_two_probability=(
                journey_top_two_probability
            ),
            threshold=journey_threshold,
            decision_made=journey_decision_made,
            rhr_enabled=journey_rhr_enabled,
            temperature_enabled=(
                journey_temperature_enabled
            ),
        )
    
    with st.expander(
        "Inspect selected-day routing path"
    ):
        render_routing_diagram(
            rhr_enabled=journey_rhr_enabled,
            temperature_enabled=(
                journey_temperature_enabled
            ),
            route_name=journey_route_name,
        )
    
    st.divider()
    
    st.subheader(
        "Interactive probability sandbox"
    )

    st.info(
        """
        Adjust hypothetical expert probabilities to see
        how equal fusion, probability temperature scaling
        and the frozen top-two threshold affect the output.
        """
    )

    preset_name = st.selectbox(
        "Example scenario",
        [
            "Expert agreement",
            "Expert disagreement",
            "Low evidence",
            "Custom starting point",
        ],
    )

    presets = {
        "Expert agreement": {
            "rhr": {
                "Menstrual": 0.10,
                "Follicular": 0.20,
                "Fertility": 0.55,
                "Luteal": 0.15,
            },
            "temperature": {
                "Menstrual": 0.12,
                "Follicular": 0.18,
                "Fertility": 0.50,
                "Luteal": 0.20,
            },
        },
        "Expert disagreement": {
            "rhr": {
                "Menstrual": 0.10,
                "Follicular": 0.55,
                "Fertility": 0.20,
                "Luteal": 0.15,
            },
            "temperature": {
                "Menstrual": 0.10,
                "Follicular": 0.20,
                "Fertility": 0.55,
                "Luteal": 0.15,
            },
        },
        "Low evidence": {
            "rhr": {
                "Menstrual": 0.26,
                "Follicular": 0.25,
                "Fertility": 0.24,
                "Luteal": 0.25,
            },
            "temperature": {
                "Menstrual": 0.24,
                "Follicular": 0.26,
                "Fertility": 0.25,
                "Luteal": 0.25,
            },
        },
        "Custom starting point": {
            "rhr": {
                "Menstrual": 0.25,
                "Follicular": 0.25,
                "Fertility": 0.25,
                "Luteal": 0.25,
            },
            "temperature": {
                "Menstrual": 0.25,
                "Follicular": 0.25,
                "Fertility": 0.25,
                "Luteal": 0.25,
            },
        },
    }

    selected_preset = presets[preset_name]

    active_probabilities = []

    if rhr_enabled:
        rhr_probabilities = probability_editor(
            "RHR expert probabilities",
            selected_preset["rhr"],
            (
                "rhr_"
                + preset_name
                .lower()
                .replace(" ", "_")
            ),
        )

        active_probabilities.append(
            rhr_probabilities
        )

    if temperature_enabled:
        temperature_probabilities = (
            probability_editor(
                "Temperature expert probabilities",
                selected_preset[
                    "temperature"
                ],
                (
                    "temperature_"
                    + preset_name
                    .lower()
                    .replace(" ", "_")
                ),
            )
        )

        active_probabilities.append(
            temperature_probabilities
        )

    if not active_probabilities:
        st.error(
            """
            Final output: NO-CALL
    
            Neither predictive sensor is available.
            """
        )
    
        render_sensor_lab_report(
            preset_name=preset_name,
            route_name=route_name,
            rhr_enabled=rhr_enabled,
            temperature_enabled=(
                temperature_enabled
            ),
            calibrated_probabilities=None,
            top_phase=None,
            top_two_phases=[],
            top_two_probability=None,
            threshold=float(
                model[
                    "top_two_threshold"
                ]
            ),
            decision_made=False,
        )
    
        return

    if len(active_probabilities) == 2:
        routed_probabilities = (
            0.5 * active_probabilities[0]
            + 0.5 * active_probabilities[1]
        )
    else:
        routed_probabilities = (
            active_probabilities[0]
        )

    calibrated_probabilities = (
        temperature_scale(
            routed_probabilities,
            float(
                model[
                    "probability_temperature"
                ]
            ),
        )
    )

    order = np.argsort(
        -calibrated_probabilities
    )

    top_phase = PHASES[
        int(order[0])
    ]

    top_two_phases = [
        PHASES[int(order[0])],
        PHASES[int(order[1])],
    ]

    top_two_probability = float(
        calibrated_probabilities[
            order[:2]
        ].sum()
    )

    threshold = float(
        model["top_two_threshold"]
    )

    decision_made = (
        top_two_probability >= threshold
    )
    st.subheader("How MARA reached this output")

    wheel_column, explanation_column = (
        st.columns([1, 1.15])
    )

    with wheel_column:
        render_phase_wheel(
            probabilities=calibrated_probabilities,
            top_two_phases=top_two_phases,
            decision_made=decision_made,
        )

    with explanation_column:
        render_decision_explanation(
            route_name=route_name,
            top_two_phases=top_two_phases,
            top_two_probability=top_two_probability,
            threshold=threshold,
            decision_made=decision_made,
            rhr_enabled=rhr_enabled,
            temperature_enabled=temperature_enabled,
        )

    result_columns = st.columns(3)

    result_columns[0].metric(
        "Top-ranked phase",
        top_phase,
        "Context only",
    )

    result_columns[1].metric(
        "Top-two evidence",
        percent(
            top_two_probability
        ),
    )

    result_columns[2].metric(
        "Decision threshold",
        percent(threshold),
    )

    render_sensor_lab_report(
        preset_name=preset_name,
        route_name=route_name,
        rhr_enabled=rhr_enabled,
        temperature_enabled=(
            temperature_enabled
        ),
        calibrated_probabilities=(
            calibrated_probabilities
        ),
        top_phase=top_phase,
        top_two_phases=top_two_phases,
        top_two_probability=(
            top_two_probability
        ),
        threshold=threshold,
        decision_made=decision_made,
    )


def render_robustness(
    route_table: pd.DataFrame,
    interval_table: pd.DataFrame,
    phase_table: pd.DataFrame,
) -> None:
    st.header("Robustness audit")

    st.write(
        """
        Aggregate results are separated by study interval
        and true phase to expose distribution shifts and
        uneven performance rather than hiding them inside
        a single overall score.
        """
    )

    render_robustness_dashboard(
        route_table=route_table,
        phase_table=phase_table,
    )
    
    st.divider()
    
    st.subheader(
        "Detailed aggregate comparisons"
    )

    st.subheader("Study-interval comparison")

    interval_chart = (
        interval_table
        .set_index("Study interval")[
            [
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
            ]
        ]
        * 100
    )

    st.bar_chart(
        interval_chart,
        y=[
            "Decision coverage",
            "Pair-set coverage",
            "No-call rate",
        ],
        color=[
            MARA_PINK,
            MARA_TEAL,
            MARA_AMBER,
        ],
        stack=False,
        height=380,
    )

    displayed_intervals = (
        format_table_percentages(
            interval_table,
            [
                "Sensor coverage",
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
                (
                    "Unrestricted top-1 "
                    "balanced accuracy"
                ),
            ],
        )
    )

    st.dataframe(
        displayed_intervals,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        """
        The 2024 interval produced fewer decisions but
        higher pair-set coverage than the 2022 interval
        on the held-out participants.
        """
    )

    st.subheader("Phase-level comparison")

    phase_chart = (
        phase_table
        .set_index("True phase")[
            [
                "Decision coverage",
                "Pair-set coverage",
            ]
        ]
        * 100
    )

    st.bar_chart(
        phase_chart,
        y=[
            "Decision coverage",
            "Pair-set coverage",
        ],
        color=[
            MARA_PINK,
            MARA_TEAL,
        ],
        stack=False,
        height=380,
    )

    displayed_phases = (
        format_table_percentages(
            phase_table,
            [
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
            ],
        )
    )

    st.dataframe(
        displayed_phases,
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        """
        Pair-set coverage varies substantially by phase.
        The prototype must not be presented as equally
        reliable across the menstrual cycle.
        """
    )


def main() -> None:
    try:
        (
            summary,
            route_table,
            interval_table,
            phase_table,
        ) = load_public_assets()

    except Exception as error:
        st.error(
            "The public application assets "
            "could not be loaded."
        )

        st.exception(error)
        st.stop()

    # Persistent MARA header on every page.
    render_mara_hero(
        summary
    )

    render_mara_metric_strip(
        summary
    )

    if NAVIGATION_KEY not in st.session_state:
        st.session_state[
            NAVIGATION_KEY
        ] = "Product Overview"
    
    page = st.session_state[
        NAVIGATION_KEY
    ]

    st.warning(
        summary["medical_disclaimer"],
        icon="⚠️",
    )

    render_page_shortcuts(
        current_page=page,
    )

    st.divider()

    if page == "Product Overview":
        render_overview(
            summary
        )

    elif page == "Held-out Benchmark":
        render_benchmark(
            summary,
            route_table,
        )

    elif page == "Sensor-Routing Lab":
        render_sensor_lab(
            summary,
            route_table,
        )

    elif page == "Robustness Audit":
        render_robustness(
            route_table,
            interval_table,
            phase_table,
        )

    elif page == "MARA AI Guide":
        render_ai_mara_chat(
            summary=summary,
            route_table=route_table,
            interval_table=interval_table,
            phase_table=phase_table,
        )

    render_mara_footer()


if __name__ == "__main__":
    main()