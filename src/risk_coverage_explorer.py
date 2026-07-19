from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_ASSETS_DIR = PROJECT_DIR / "app_assets"

CURVE_PATH = (
    APP_ASSETS_DIR
    / "validation_risk_coverage.csv"
)

METADATA_PATH = (
    APP_ASSETS_DIR
    / "validation_risk_coverage_metadata.json"
)

MARA_PINK = "#D65D8B"
MARA_TEAL = "#4F8F88"
MARA_LAVENDER = "#8B7BBE"
MARA_AMBER = "#D99A45"
MARA_BERRY = "#71324F"
MARA_INK = "#352631"
MARA_MUTED = "#6F5964"
MARA_GRID = "#F1E6EB"


def percentage(
    value: float | int,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{100.0 * float(value):.2f}%"


@st.cache_data
def load_validation_risk_coverage() -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    if not CURVE_PATH.exists():
        raise FileNotFoundError(
            "Validation risk-coverage asset "
            f"not found: {CURVE_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Validation risk-coverage metadata "
            f"not found: {METADATA_PATH}"
        )

    curve = pd.read_csv(
        CURVE_PATH
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_columns = {
        "Threshold",
        "Total validation days",
        "Sensor-available days",
        "Decided days",
        "Missing-sensor no-call days",
        "Insufficient-evidence no-call days",
        "Decision coverage",
        "Pair-set coverage",
        "Selective risk",
        "No-call rate",
        "Missing-sensor no-call rate",
        "Insufficient-evidence no-call rate",
        "Frozen policy",
    }

    missing_columns = (
        required_columns
        - set(curve.columns)
    )

    if missing_columns:
        formatted = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Validation risk-coverage asset "
            "is missing columns: "
            f"{formatted}"
        )

    curve["Threshold"] = pd.to_numeric(
        curve["Threshold"],
        errors="raise",
    ).round(2)

    curve = curve.sort_values(
        "Threshold"
    ).reset_index(
        drop=True
    )

    return curve, metadata


def selected_threshold_row(
    curve: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    matches = curve.loc[
        np.isclose(
            curve["Threshold"],
            threshold,
        )
    ]

    if matches.empty:
        raise ValueError(
            "Selected threshold is not "
            "present in the aggregate curve."
        )

    return matches.iloc[0]


def render_validation_risk_coverage_explorer() -> None:
    try:
        curve, metadata = (
            load_validation_risk_coverage()
        )

    except Exception as error:
        st.warning(
            "The validation risk-coverage explorer "
            "could not be loaded."
        )

        with st.expander(
            "Technical details"
        ):
            st.exception(error)

        return

    frozen_threshold = round(
        float(
            metadata[
                "frozen_threshold"
            ]
        ),
        2,
    )

    st.html(
        """
<section style="
    margin:1.2rem 0 1.4rem;
    padding:1.4rem 1.5rem;
    border:1px solid #E8D3DD;
    border-radius:22px;
    background:
        linear-gradient(
            120deg,
            rgba(255,241,246,0.97),
            rgba(244,241,253,0.97) 55%,
            rgba(237,247,245,0.97)
        );
">
    <div style="
        color:#71324F;
        font-size:0.78rem;
        font-weight:750;
        letter-spacing:0.06em;
        text-transform:uppercase;
    ">
        Validation-only policy explorer
    </div>

    <div style="
        margin-top:0.35rem;
        color:#352631;
        font-size:1.35rem;
        font-weight:780;
        letter-spacing:-0.025em;
    ">
        Explore MARA's coverage–reliability trade-off
    </div>

    <div style="
        max-width:900px;
        margin-top:0.45rem;
        color:#6F5964;
        font-size:0.92rem;
        line-height:1.6;
    ">
        Raising the evidence threshold causes MARA to
        abstain more often. The explorer below is derived
        only from validation predictions. The held-out test
        set was not used to construct or tune this curve.
    </div>
</section>
"""
    )

    threshold_options = (
        curve["Threshold"]
        .astype(float)
        .tolist()
    )

    selected_threshold = st.select_slider(
        "Exploratory top-two evidence threshold",
        options=threshold_options,
        value=frozen_threshold,
        format_func=(
            lambda value: f"{value:.2f}"
        ),
        key="validation_threshold_explorer",
    )

    selected_row = selected_threshold_row(
        curve,
        float(selected_threshold),
    )

    is_frozen_threshold = bool(
        np.isclose(
            selected_threshold,
            frozen_threshold,
        )
    )

    if is_frozen_threshold:
        st.success(
            """
            You are viewing MARA's frozen policy threshold.
            The held-out test benchmark was evaluated using
            this threshold without later adjustment.
            """
        )

    else:
        st.warning(
            f"""
            Exploratory validation setting: {selected_threshold:.2f}

            MARA's frozen policy remains {frozen_threshold:.2f}.
            Moving this control does not change the model,
            public benchmark, or final test policy.
            """
        )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Selected threshold",
        f"{selected_threshold:.2f}",
        (
            "Frozen policy"
            if is_frozen_threshold
            else "Validation exploration"
        ),
    )

    metric_columns[1].metric(
        "Decision coverage",
        percentage(
            selected_row[
                "Decision coverage"
            ]
        ),
        "All validation days",
    )

    metric_columns[2].metric(
        "Pair-set coverage",
        percentage(
            selected_row[
                "Pair-set coverage"
            ]
        ),
        "Among decided days",
    )

    metric_columns[3].metric(
        "No-call rate",
        percentage(
            selected_row[
                "No-call rate"
            ]
        ),
        "Missing or insufficient evidence",
    )

    metric_columns[4].metric(
        "Decided days",
        int(
            selected_row[
                "Decided days"
            ]
        ),
        (
            f"of "
            f"{int(selected_row['Total validation days'])}"
        ),
    )

    chart_frame = curve[
        [
            "Threshold",
            "Decision coverage",
            "Pair-set coverage",
            "No-call rate",
        ]
    ].copy()

    chart_frame[
        [
            "Decision coverage",
            "Pair-set coverage",
            "No-call rate",
        ]
    ] = (
        chart_frame[
            [
                "Decision coverage",
                "Pair-set coverage",
                "No-call rate",
            ]
        ]
        * 100.0
    )

    long_chart = chart_frame.melt(
        id_vars="Threshold",
        var_name="Metric",
        value_name="Percent",
    )

    metric_domain = [
        "Decision coverage",
        "Pair-set coverage",
        "No-call rate",
    ]

    metric_range = [
        MARA_PINK,
        MARA_TEAL,
        MARA_AMBER,
    ]

    base_chart = (
        alt.Chart(long_chart)
        .mark_line(
            strokeWidth=3,
            interpolate="monotone",
            point={
                "filled": True,
                "size": 28,
            },
        )
        .encode(
            x=alt.X(
                "Threshold:Q",
                title="Top-two evidence threshold",
                scale=alt.Scale(
                    domain=[
                        float(
                            curve[
                                "Threshold"
                            ].min()
                        ),
                        float(
                            curve[
                                "Threshold"
                            ].max()
                        ),
                    ]
                ),
                axis=alt.Axis(
                    format=".2f",
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                    tickColor="#DCC6D0",
                    domainColor="#DCC6D0",
                    gridColor=MARA_GRID,
                ),
            ),
            y=alt.Y(
                "Percent:Q",
                title="Validation days (%)",
                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
                axis=alt.Axis(
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                    gridColor=MARA_GRID,
                    domain=False,
                ),
            ),
            color=alt.Color(
                "Metric:N",
                title=None,
                scale=alt.Scale(
                    domain=metric_domain,
                    range=metric_range,
                ),
                legend=alt.Legend(
                    orient="bottom",
                    labelColor=MARA_MUTED,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    format=".2f",
                    title="Threshold",
                ),
                alt.Tooltip(
                    "Metric:N",
                    title="Metric",
                ),
                alt.Tooltip(
                    "Percent:Q",
                    format=".2f",
                    title="Percent",
                ),
            ],
        )
    )

    frozen_rule_data = pd.DataFrame(
        {
            "Threshold": [
                frozen_threshold
            ],
            "Label": [
                (
                    "Frozen policy "
                    f"{frozen_threshold:.2f}"
                )
            ],
        }
    )

    frozen_rule = (
        alt.Chart(
            frozen_rule_data
        )
        .mark_rule(
            color=MARA_BERRY,
            strokeWidth=3,
        )
        .encode(
            x="Threshold:Q",
            tooltip=[
                alt.Tooltip(
                    "Label:N",
                    title=None,
                )
            ],
        )
    )

    selected_rule_data = pd.DataFrame(
        {
            "Threshold": [
                selected_threshold
            ],
            "Label": [
                (
                    "Selected "
                    f"{selected_threshold:.2f}"
                )
            ],
        }
    )

    selected_rule = (
        alt.Chart(
            selected_rule_data
        )
        .mark_rule(
            color=MARA_LAVENDER,
            strokeWidth=2,
            strokeDash=[
                6,
                5,
            ],
        )
        .encode(
            x="Threshold:Q",
            tooltip=[
                alt.Tooltip(
                    "Label:N",
                    title=None,
                )
            ],
        )
    )

    threshold_chart = (
        base_chart
        + frozen_rule
        + selected_rule
    ).properties(
        height=410,
        title={
            "text": (
                "Validation behaviour across "
                "evidence thresholds"
            ),
            "subtitle": (
                "The solid berry line marks the "
                "frozen 0.62 policy."
            ),
            "color": MARA_INK,
            "subtitleColor": MARA_MUTED,
            "fontSize": 17,
            "subtitleFontSize": 12,
            "anchor": "start",
        },
    ).configure_view(
        stroke=None
    )

    st.altair_chart(
        threshold_chart,
        width="stretch",
        theme=None,
    )

    st.subheader(
        "Coverage–reliability frontier"
    )

    risk_curve = curve.loc[
        curve[
            "Pair-set coverage"
        ].notna()
    ].copy()

    risk_curve[
        "Decision coverage (%)"
    ] = (
        risk_curve[
            "Decision coverage"
        ]
        * 100.0
    )

    risk_curve[
        "Pair-set coverage (%)"
    ] = (
        risk_curve[
            "Pair-set coverage"
        ]
        * 100.0
    )

    frontier = (
        alt.Chart(risk_curve)
        .mark_line(
            color=MARA_LAVENDER,
            strokeWidth=3,
            interpolate="monotone",
            point={
                "filled": True,
                "fill": MARA_LAVENDER,
                "size": 44,
            },
        )
        .encode(
            x=alt.X(
                "Decision coverage (%):Q",
                title="Decision coverage (%)",
                scale=alt.Scale(
                    zero=False
                ),
                axis=alt.Axis(
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                    gridColor=MARA_GRID,
                    domain=False,
                ),
            ),
            y=alt.Y(
                "Pair-set coverage (%):Q",
                title=(
                    "Pair-set coverage "
                    "among decided days (%)"
                ),
                scale=alt.Scale(
                    zero=False
                ),
                axis=alt.Axis(
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                    gridColor=MARA_GRID,
                    domain=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    format=".2f",
                    title="Threshold",
                ),
                alt.Tooltip(
                    "Decision coverage (%):Q",
                    format=".2f",
                    title="Decision coverage",
                ),
                alt.Tooltip(
                    "Pair-set coverage (%):Q",
                    format=".2f",
                    title="Pair-set coverage",
                ),
                alt.Tooltip(
                    "Decided days:Q",
                    format=".0f",
                    title="Decided days",
                ),
            ],
        )
    )

    frozen_frontier_row = (
        risk_curve.loc[
            np.isclose(
                risk_curve["Threshold"],
                frozen_threshold,
            )
        ]
    )

    frozen_point = (
        alt.Chart(
            frozen_frontier_row
        )
        .mark_circle(
            color=MARA_PINK,
            stroke="#FFFFFF",
            strokeWidth=3,
            size=230,
        )
        .encode(
            x="Decision coverage (%):Q",
            y="Pair-set coverage (%):Q",
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    format=".2f",
                    title="Frozen threshold",
                ),
                alt.Tooltip(
                    "Decision coverage (%):Q",
                    format=".2f",
                    title="Decision coverage",
                ),
                alt.Tooltip(
                    "Pair-set coverage (%):Q",
                    format=".2f",
                    title="Pair-set coverage",
                ),
            ],
        )
    )

    selected_frontier_row = (
        risk_curve.loc[
            np.isclose(
                risk_curve["Threshold"],
                selected_threshold,
            )
        ]
    )

    selected_point = (
        alt.Chart(
            selected_frontier_row
        )
        .mark_circle(
            fill="#FFFFFF",
            stroke=MARA_BERRY,
            strokeWidth=4,
            size=180,
        )
        .encode(
            x="Decision coverage (%):Q",
            y="Pair-set coverage (%):Q",
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    format=".2f",
                    title="Selected threshold",
                )
            ],
        )
    )

    frontier_chart = (
        frontier
        + frozen_point
        + selected_point
    ).properties(
        height=400,
        title={
            "text": (
                "Validation coverage–reliability frontier"
            ),
            "subtitle": (
                "Moving toward higher pair-set coverage "
                "generally reduces the number of days "
                "receiving an output."
            ),
            "color": MARA_INK,
            "subtitleColor": MARA_MUTED,
            "fontSize": 17,
            "subtitleFontSize": 12,
            "anchor": "start",
        },
    ).configure_view(
        stroke=None
    )

    st.altair_chart(
        frontier_chart,
        width="stretch",
        theme=None,
    )

    st.subheader(
        "Why MARA returned no-call"
    )

    reason_columns = st.columns(2)

    reason_columns[0].metric(
        "Missing-sensor no-calls",
        int(
            selected_row[
                "Missing-sensor no-call days"
            ]
        ),
        percentage(
            selected_row[
                "Missing-sensor no-call rate"
            ]
        ),
    )

    reason_columns[1].metric(
        "Insufficient-evidence no-calls",
        int(
            selected_row[
                (
                    "Insufficient-evidence "
                    "no-call days"
                )
            ]
        ),
        percentage(
            selected_row[
                (
                    "Insufficient-evidence "
                    "no-call rate"
                )
            ]
        ),
    )

    if (
        selected_row[
            "Insufficient-evidence no-call days"
        ]
        > selected_row[
            "Missing-sensor no-call days"
        ]
    ):
        st.info(
            """
            At this threshold, most abstentions occur
            because the calibrated top-two evidence is
            below the required level—not because both
            wearable sensors are missing.
            """
        )

    else:
        st.info(
            """
            At this threshold, missing-sensor days form
            a substantial portion of the no-call outputs.
            """
        )

    with st.expander(
        "View aggregate threshold table"
    ):
        displayed = curve[
            [
                "Threshold",
                "Decided days",
                "Decision coverage",
                "Pair-set coverage",
                "Selective risk",
                "No-call rate",
                (
                    "Missing-sensor "
                    "no-call days"
                ),
                (
                    "Insufficient-evidence "
                    "no-call days"
                ),
                "Frozen policy",
            ]
        ].copy()

        for column in [
            "Decision coverage",
            "Pair-set coverage",
            "Selective risk",
            "No-call rate",
        ]:
            displayed[column] = (
                displayed[column]
                .apply(percentage)
            )

        st.dataframe(
            displayed,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        """
        This explorer uses aggregate validation results.
        It contains no participant identifiers or daily
        prediction rows. Alternative thresholds shown here
        are exploratory and do not replace MARA's frozen
        0.62 test policy.
        """
    )