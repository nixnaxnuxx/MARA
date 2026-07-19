from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

ROUTE_COLORS = {
    "Equal fusion": "#D65D8B",
    "RHR expert": "#4F8F88",
    "Temperature expert": "#8B7BBE",
    "Neither sensor": "#D99A45",
}


@st.cache_data
def build_synthetic_journey(
    probability_temperature: float,
    threshold: float,
) -> pd.DataFrame:
    """
    Build a deterministic synthetic demonstration.

    This is not participant data and is not intended
    to represent a typical or clinically valid cycle.
    """

    days = np.arange(1, 29)

    synthetic_rhr = (
        62.4
        + 1.35
        * np.sin(
            2.0
            * np.pi
            * (days - 16)
            / 28.0
        )
        + 0.35
        * np.cos(
            2.0
            * np.pi
            * days
            / 7.0
        )
    )

    synthetic_temperature = (
        36.22
        + 0.18
        * np.sin(
            2.0
            * np.pi
            * (days - 14)
            / 28.0
        )
        + 0.03
        * np.cos(
            2.0
            * np.pi
            * days
            / 5.0
        )
    )

    # Deliberately selected missingness patterns.
    rhr_missing_days = {
        13,
        14,
        24,
        27,
    }

    temperature_missing_days = {
        8,
        9,
        18,
        19,
        20,
        27,
    }

    # Synthetic phase-profile centres and widths.
    phase_centres = np.array(
        [
            3.0,
            9.0,
            14.5,
            22.0,
        ]
    )

    phase_widths = np.array(
        [
            2.8,
            4.2,
            2.4,
            5.5,
        ]
    )

    # Lower synthetic evidence around selected transition days.
    quality_by_day = {
        6: 0.15,
        7: 0.35,
        12: 0.20,
        13: 0.35,
        16: 0.20,
        17: 0.35,
        24: 0.25,
        25: 0.40,
    }

    rows: list[dict[str, object]] = []

    for index, day in enumerate(days):
        day_number = int(day)

        rhr_available = (
            day_number
            not in rhr_missing_days
        )

        temperature_available = (
            day_number
            not in temperature_missing_days
        )

        if (
            rhr_available
            and temperature_available
        ):
            route = "Equal fusion"
            route_strength = 0.95

        elif rhr_available:
            route = "RHR expert"
            route_strength = 0.82

        elif temperature_available:
            route = "Temperature expert"
            route_strength = 0.76

        else:
            route = "Neither sensor"
            route_strength = 0.0

        phase_scores = (
            np.exp(
                -0.5
                * (
                    (
                        day_number
                        - phase_centres
                    )
                    / phase_widths
                )
                ** 2
            )
            + 0.02
        )

        base_probabilities = (
            phase_scores
            / phase_scores.sum()
        )

        synthetic_quality = (
            quality_by_day.get(
                day_number,
                1.0,
            )
        )

        mixture_strength = (
            route_strength
            * synthetic_quality
        )

        uniform_probabilities = np.full(
            len(PHASES),
            1.0 / len(PHASES),
            dtype=float,
        )

        probabilities = (
            mixture_strength
            * base_probabilities
            + (
                1.0
                - mixture_strength
            )
            * uniform_probabilities
        )

        clipped = np.clip(
            probabilities,
            1e-12,
            1.0,
        )

        calibrated_probabilities = (
            np.power(
                clipped,
                1.0
                / probability_temperature,
            )
        )

        calibrated_probabilities = (
            calibrated_probabilities
            / calibrated_probabilities.sum()
        )

        probability_order = np.argsort(
            -calibrated_probabilities
        )

        primary_phase = PHASES[
            int(probability_order[0])
        ]

        secondary_phase = PHASES[
            int(probability_order[1])
        ]

        top_two_evidence = float(
            calibrated_probabilities[
                probability_order[:2]
            ].sum()
        )

        decision_made = bool(
            route != "Neither sensor"
            and top_two_evidence
            >= threshold
        )

        row: dict[str, object] = {
            "Day": day_number,
            "Day start": (
                day_number - 0.45
            ),
            "Day end": (
                day_number + 0.45
            ),
            "RHR": (
                round(
                    float(
                        synthetic_rhr[index]
                    ),
                    2,
                )
                if rhr_available
                else np.nan
            ),
            "Temperature": (
                round(
                    float(
                        synthetic_temperature[
                            index
                        ]
                    ),
                    3,
                )
                if temperature_available
                else np.nan
            ),
            "RHR available": (
                rhr_available
            ),
            "Temperature available": (
                temperature_available
            ),
            "Route": route,
            "Primary phase": (
                primary_phase
            ),
            "Secondary phase": (
                secondary_phase
            ),
            "Top-two evidence": (
                top_two_evidence
            ),
            "Decision": decision_made,
            "Output": (
                "Phase pair"
                if decision_made
                else "No-call"
            ),
        }

        for phase_index, phase in enumerate(
            PHASES
        ):
            row[phase] = float(
                calibrated_probabilities[
                    phase_index
                ]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _format_sensor_value(
    value: object,
    unit: str,
) -> str:
    if pd.isna(value):
        return "Missing"

    if unit == "bpm":
        return f"{float(value):.1f} bpm"

    return f"{float(value):.2f} °C"


def render_synthetic_journey(
    journey: pd.DataFrame,
) -> pd.Series:
    st.html(
        """
<section style="
    margin:1rem 0 1.3rem;
    padding:1.35rem 1.45rem;
    border:1px solid #e8d3dd;
    border-radius:22px;
    background:
        linear-gradient(
            120deg,
            rgba(255,241,246,0.96),
            rgba(237,247,245,0.96)
        );
">
    <div style="
        color:#71324f;
        font-size:0.78rem;
        font-weight:750;
        letter-spacing:0.06em;
        text-transform:uppercase;
    ">
        Interactive product demonstration
    </div>

    <div style="
        margin-top:0.35rem;
        color:#352631;
        font-size:1.35rem;
        font-weight:780;
        letter-spacing:-0.025em;
    ">
        Synthetic 28-day MARA journey
    </div>

    <div style="
        max-width:850px;
        margin-top:0.45rem;
        color:#6f5964;
        font-size:0.92rem;
        line-height:1.6;
    ">
        Explore how changing signal patterns and missing
        wearable modalities alter the route, evidence
        strength and selective output. All values in this
        demonstration are synthetic.
    </div>
</section>
"""
    )

    selected_day = st.select_slider(
        "Explore synthetic day",
        options=journey[
            "Day"
        ].astype(int).tolist(),
        value=14,
        format_func=(
            lambda day: f"Day {day}"
        ),
        key="mara_synthetic_day",
    )

    selected_row = journey.loc[
        journey["Day"] == selected_day
    ].iloc[0]

    route_display_names = {
        "Equal fusion": "Both sensors",
        "RHR expert": "RHR only",
        "Temperature expert": "Temperature only",
        "Neither sensor": "No sensors",
    }

    route_display = route_display_names.get(
        str(selected_row["Route"]),
        str(selected_row["Route"]),
    )

    with st.container(
        key="synthetic_journey_metrics"
    ):
        metric_columns = st.columns(5)

        metric_columns[0].metric(
            label="Selected day",
            value=f"Day {int(selected_day)}",
        )

        metric_columns[1].metric(
            label="Active route",
            value=route_display,
        )

        metric_columns[2].metric(
            label="Resting heart rate",
            value=_format_sensor_value(
                selected_row["RHR"],
                "bpm",
            ),
        )

        metric_columns[3].metric(
            label="Wrist temperature",
            value=_format_sensor_value(
                selected_row["Temperature"],
                "°C",
            ),
        )

        metric_columns[4].metric(
            label="Selective output",
            value=str(
                selected_row["Output"]
            ),
        )

    selected_day_frame = pd.DataFrame(
        {
            "Day": [
                selected_day
            ]
        }
    )

    x_encoding = alt.X(
        "Day:Q",
        title=None,
        scale=alt.Scale(
            domain=[
                1,
                28,
            ]
        ),
        axis=alt.Axis(
            values=list(
                range(
                    1,
                    29,
                    2,
                )
            ),
            labelColor="#6F5964",
            tickColor="#DCC6D0",
            domainColor="#DCC6D0",
            gridColor="#F1E6EB",
        ),
    )

    selection_rule = (
        alt.Chart(
            selected_day_frame
        )
        .mark_rule(
            color="#352631",
            strokeWidth=2,
            strokeDash=[
                5,
                4,
            ],
        )
        .encode(
            x="Day:Q"
        )
    )

    rhr_line = (
        alt.Chart(journey)
        .mark_line(
            color="#4F8F88",
            strokeWidth=3,
            interpolate="monotone",
        )
        .encode(
            x=x_encoding,
            y=alt.Y(
                "RHR:Q",
                title="RHR (bpm)",
                scale=alt.Scale(
                    zero=False
                ),
                axis=alt.Axis(
                    labelColor="#6F5964",
                    titleColor="#4F8F88",
                    gridColor="#F1E6EB",
                    domain=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Day:Q",
                    format=".0f",
                    title="Day",
                ),
                alt.Tooltip(
                    "RHR:Q",
                    format=".1f",
                    title="RHR (bpm)",
                ),
                alt.Tooltip(
                    "Route:N",
                    title="Route",
                ),
            ],
        )
    )

    rhr_points = (
        alt.Chart(journey)
        .mark_circle(
            color="#4F8F88",
            size=48,
            opacity=0.72,
        )
        .encode(
            x="Day:Q",
            y="RHR:Q",
        )
    )

    selected_rhr_point = (
        alt.Chart(
            journey.loc[
                journey["Day"]
                == selected_day
            ]
        )
        .mark_circle(
            color="#FFFFFF",
            stroke="#4F8F88",
            strokeWidth=4,
            size=150,
        )
        .encode(
            x="Day:Q",
            y="RHR:Q",
        )
    )

    rhr_chart = (
        rhr_line
        + rhr_points
        + selection_rule
        + selected_rhr_point
    ).properties(
        height=155,
        title={
            "text": (
                "Synthetic resting-heart-rate signal"
            ),
            "color": "#352631",
            "fontSize": 15,
            "anchor": "start",
        },
    )

    temperature_line = (
        alt.Chart(journey)
        .mark_line(
            color="#F08A7C",
            strokeWidth=3,
            interpolate="monotone",
        )
        .encode(
            x=x_encoding,
            y=alt.Y(
                "Temperature:Q",
                title="Temperature (°C)",
                scale=alt.Scale(
                    zero=False
                ),
                axis=alt.Axis(
                    labelColor="#6F5964",
                    titleColor="#F08A7C",
                    gridColor="#F1E6EB",
                    domain=False,
                    format=".2f",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Day:Q",
                    format=".0f",
                    title="Day",
                ),
                alt.Tooltip(
                    "Temperature:Q",
                    format=".2f",
                    title="Temperature (°C)",
                ),
                alt.Tooltip(
                    "Route:N",
                    title="Route",
                ),
            ],
        )
    )

    temperature_points = (
        alt.Chart(journey)
        .mark_circle(
            color="#F08A7C",
            size=48,
            opacity=0.72,
        )
        .encode(
            x="Day:Q",
            y="Temperature:Q",
        )
    )

    selected_temperature_point = (
        alt.Chart(
            journey.loc[
                journey["Day"]
                == selected_day
            ]
        )
        .mark_circle(
            color="#FFFFFF",
            stroke="#F08A7C",
            strokeWidth=4,
            size=150,
        )
        .encode(
            x="Day:Q",
            y="Temperature:Q",
        )
    )

    temperature_chart = (
        temperature_line
        + temperature_points
        + selection_rule
        + selected_temperature_point
    ).properties(
        height=155,
        title={
            "text": (
                "Synthetic wrist-temperature signal"
            ),
            "color": "#352631",
            "fontSize": 15,
            "anchor": "start",
        },
    )

    route_domain = list(
        ROUTE_COLORS.keys()
    )

    route_range = [
        ROUTE_COLORS[
            route
        ]
        for route in route_domain
    ]

    route_strip = (
        alt.Chart(journey)
        .mark_rect(
            cornerRadius=4,
        )
        .encode(
            x=alt.X(
                "Day start:Q",
                title="Synthetic cycle day",
                scale=alt.Scale(
                    domain=[
                        0.5,
                        28.5,
                    ]
                ),
                axis=alt.Axis(
                    values=list(
                        range(
                            1,
                            29,
                            2,
                        )
                    ),
                    labelColor="#6F5964",
                    titleColor="#6F5964",
                    tickColor="#DCC6D0",
                    domainColor="#DCC6D0",
                ),
            ),
            x2="Day end:Q",
            color=alt.Color(
                "Route:N",
                title=None,
                scale=alt.Scale(
                    domain=route_domain,
                    range=route_range,
                ),
                legend=alt.Legend(
                    orient="bottom",
                    columns=2,
                    labelColor="#6F5964",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Day:Q",
                    format=".0f",
                    title="Day",
                ),
                alt.Tooltip(
                    "Route:N",
                    title="Active route",
                ),
                alt.Tooltip(
                    "Output:N",
                    title="Output",
                ),
                alt.Tooltip(
                    "Top-two evidence:Q",
                    format=".1%",
                    title="Top-two evidence",
                ),
            ],
        )
        .properties(
            height=42,
            title={
                "text": (
                    "Daily missing-modality route"
                ),
                "color": "#352631",
                "fontSize": 15,
                "anchor": "start",
            },
        )
    )

    selected_route_outline = (
        alt.Chart(
            journey.loc[
                journey["Day"]
                == selected_day
            ]
        )
        .mark_rect(
            fillOpacity=0,
            stroke="#352631",
            strokeWidth=3,
            cornerRadius=4,
        )
        .encode(
            x="Day start:Q",
            x2="Day end:Q",
        )
    )

    combined_chart = alt.vconcat(
        rhr_chart,
        temperature_chart,
        (
            route_strip
            + selected_route_outline
        ),
        spacing=14,
    ).configure_view(
        stroke=None
    ).configure_title(
        font="sans-serif"
    )

    st.altair_chart(
        combined_chart,
        width="stretch",
        theme=None,
    )

    st.caption(
        """
        Gaps in a signal line represent deliberately
        missing synthetic measurements. The coloured strip
        shows the route selected from the available sensors.
        This demonstration is not generated from personal
        measurements or restricted participant data.
        """
    )

    return selected_row