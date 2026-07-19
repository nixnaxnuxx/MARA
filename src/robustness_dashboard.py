from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


MARA_PINK = "#D65D8B"
MARA_TEAL = "#4F8F88"
MARA_LAVENDER = "#8B7BBE"
MARA_CORAL = "#F08A7C"
MARA_AMBER = "#D99A45"
MARA_BERRY = "#71324F"
MARA_INK = "#352631"
MARA_MUTED = "#6F5964"
MARA_GRID = "#F1E6EB"

ROUTE_ORDER = [
    "Equal fusion",
    "RHR expert",
    "Temperature expert",
    "Neither sensor",
]

ROUTE_COLORS = {
    "Equal fusion": MARA_PINK,
    "RHR expert": MARA_TEAL,
    "Temperature expert": MARA_LAVENDER,
    "Neither sensor": MARA_AMBER,
}

ROBUSTNESS_METRICS = [
    "Decision coverage",
    "Pair-set coverage",
    "No-call rate",
]


def percentage(
    value: float | int | None,
) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{100.0 * float(value):.2f}%"


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        formatted = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"{table_name} is missing "
            f"required columns: {formatted}"
        )


def _prepare_heatmap_data(
    dataframe: pd.DataFrame,
    category_column: str,
) -> pd.DataFrame:
    required_columns = {
        category_column,
        *ROBUSTNESS_METRICS,
    }

    _require_columns(
        dataframe,
        required_columns,
        category_column,
    )

    selected = dataframe[
        [
            category_column,
            *ROBUSTNESS_METRICS,
        ]
    ].copy()

    for column in ROBUSTNESS_METRICS:
        selected[column] = pd.to_numeric(
            selected[column],
            errors="coerce",
        )

    long_table = selected.melt(
        id_vars=category_column,
        value_vars=ROBUSTNESS_METRICS,
        var_name="Metric",
        value_name="Rate",
    )

    long_table["Percent"] = (
        long_table["Rate"]
        * 100.0
    )

    return long_table


def render_robustness_heatmap(
    dataframe: pd.DataFrame,
    category_column: str,
    category_order: list[str] | None,
    title: str,
    subtitle: str,
) -> None:
    long_table = _prepare_heatmap_data(
        dataframe,
        category_column,
    )

    heatmap = (
        alt.Chart(long_table)
        .mark_rect(
            stroke="#FFFFFF",
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "Metric:N",
                title=None,
                sort=ROBUSTNESS_METRICS,
                axis=alt.Axis(
                    labelAngle=0,
                    labelColor=MARA_MUTED,
                    labelFontSize=12,
                    labelPadding=10,
                    domain=False,
                    ticks=False,
                ),
            ),
            y=alt.Y(
                f"{category_column}:N",
                title=None,
                sort=category_order,
                axis=alt.Axis(
                    labelColor=MARA_INK,
                    labelFontSize=12,
                    labelPadding=8,
                    domain=False,
                    ticks=False,
                ),
            ),
            color=alt.Color(
                "Percent:Q",
                title="Magnitude (%)",
                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ],
                    range=[
                        "#FFF8FB",
                        "#F5DCE7",
                        "#DABFDD",
                        "#9F9BC6",
                        "#4F8F88",
                    ],
                ),
                legend=alt.Legend(
                    orient="bottom",
                    gradientLength=280,
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    f"{category_column}:N",
                    title=category_column,
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

    labels = (
        alt.Chart(long_table)
        .mark_text(
            fontSize=13,
            fontWeight=700,
        )
        .encode(
            x=alt.X(
                "Metric:N",
                sort=ROBUSTNESS_METRICS,
            ),
            y=alt.Y(
                f"{category_column}:N",
                sort=category_order,
            ),
            text=alt.Text(
                "Percent:Q",
                format=".1f",
            ),
            color=alt.condition(
                "datum.Percent >= 62",
                alt.value("#FFFFFF"),
                alt.value(MARA_INK),
            ),
        )
    )

    chart = (
        heatmap
        + labels
    ).properties(
        height=max(
            230,
            62
            * dataframe[
                category_column
            ].nunique(),
        ),
        title={
            "text": title,
            "subtitle": subtitle,
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
        chart,
        width="stretch",
        theme=None,
    )

    st.caption(
        """
        Colour intensity represents metric magnitude,
        not whether a value is inherently desirable.
        For example, higher pair-set coverage is desirable,
        while a higher no-call rate means more abstention.
        """
    )


def _sample_size_label(
    participant_days: float,
) -> str:
    if participant_days < 50:
        return "Very small route sample"

    if participant_days < 100:
        return "Small route sample"

    if participant_days < 250:
        return "Moderate route sample"

    return "Larger route sample"


def render_route_sample_warning(
    route_table: pd.DataFrame,
) -> None:
    required_columns = {
        "Sensor route",
        "Participant-days",
    }

    _require_columns(
        route_table,
        required_columns,
        "Route benchmark",
    )

    displayed = route_table[
        [
            "Sensor route",
            "Participant-days",
        ]
    ].copy()

    displayed["Participant-days"] = (
        pd.to_numeric(
            displayed[
                "Participant-days"
            ],
            errors="coerce",
        )
    )

    displayed[
        "Interpretation caution"
    ] = displayed[
        "Participant-days"
    ].apply(
        _sample_size_label
    )

    with st.expander(
        "Inspect route sample sizes"
    ):
        st.dataframe(
            displayed,
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            """
            Route-specific percentages can be unstable
            when the number of participant-days is small.
            The heatmap describes this held-out cohort and
            should not be treated as a clinical guarantee.
            """
        )


def _observed_route_shares(
    route_table: pd.DataFrame,
) -> dict[str, float]:
    required_columns = {
        "Sensor route",
        "Participant-days",
    }

    _require_columns(
        route_table,
        required_columns,
        "Route benchmark",
    )

    working = route_table[
        [
            "Sensor route",
            "Participant-days",
        ]
    ].copy()

    working["Participant-days"] = (
        pd.to_numeric(
            working[
                "Participant-days"
            ],
            errors="coerce",
        ).fillna(0.0)
    )

    total_days = float(
        working[
            "Participant-days"
        ].sum()
    )

    if total_days <= 0:
        raise ValueError(
            "Route benchmark contains "
            "no participant-days."
        )

    shares = {
        route: 0.0
        for route in ROUTE_ORDER
    }

    for _, row in working.iterrows():
        route = str(
            row["Sensor route"]
        )

        if route in shares:
            shares[route] = (
                float(
                    row[
                        "Participant-days"
                    ]
                )
                / total_days
            )

    return shares


def _simulated_route_shares(
    rhr_missing_rate: float,
    temperature_missing_rate: float,
) -> dict[str, float]:
    rhr_missing_rate = float(
        np.clip(
            rhr_missing_rate,
            0.0,
            1.0,
        )
    )

    temperature_missing_rate = float(
        np.clip(
            temperature_missing_rate,
            0.0,
            1.0,
        )
    )

    return {
        "Equal fusion": (
            (
                1.0
                - rhr_missing_rate
            )
            * (
                1.0
                - temperature_missing_rate
            )
        ),
        "RHR expert": (
            (
                1.0
                - rhr_missing_rate
            )
            * temperature_missing_rate
        ),
        "Temperature expert": (
            rhr_missing_rate
            * (
                1.0
                - temperature_missing_rate
            )
        ),
        "Neither sensor": (
            rhr_missing_rate
            * temperature_missing_rate
        ),
    }


def _reweighted_metrics(
    route_table: pd.DataFrame,
    route_shares: dict[str, float],
) -> dict[str, float]:
    required_columns = {
        "Sensor route",
        "Decision coverage",
        "Pair-set coverage",
    }

    _require_columns(
        route_table,
        required_columns,
        "Route benchmark",
    )

    working = route_table.copy()

    working[
        "Decision coverage"
    ] = pd.to_numeric(
        working[
            "Decision coverage"
        ],
        errors="coerce",
    ).fillna(0.0)

    working[
        "Pair-set coverage"
    ] = pd.to_numeric(
        working[
            "Pair-set coverage"
        ],
        errors="coerce",
    )

    route_lookup = (
        working
        .set_index(
            "Sensor route"
        )
    )

    expected_decision_coverage = 0.0
    expected_correct_decisions = 0.0

    for route in ROUTE_ORDER:
        route_share = float(
            route_shares.get(
                route,
                0.0,
            )
        )

        if route not in route_lookup.index:
            continue

        route_row = route_lookup.loc[
            route
        ]

        decision_coverage = float(
            route_row[
                "Decision coverage"
            ]
        )

        pair_set_coverage = (
            route_row[
                "Pair-set coverage"
            ]
        )

        if pd.isna(
            pair_set_coverage
        ):
            pair_set_coverage = 0.0

        pair_set_coverage = float(
            pair_set_coverage
        )

        expected_decision_coverage += (
            route_share
            * decision_coverage
        )

        expected_correct_decisions += (
            route_share
            * decision_coverage
            * pair_set_coverage
        )

    if expected_decision_coverage > 0:
        expected_pair_set_coverage = (
            expected_correct_decisions
            / expected_decision_coverage
        )
    else:
        expected_pair_set_coverage = np.nan

    return {
        "Decision coverage": (
            expected_decision_coverage
        ),
        "Pair-set coverage": (
            expected_pair_set_coverage
        ),
        "No-call rate": (
            1.0
            - expected_decision_coverage
        ),
    }


def _route_comparison_frame(
    observed_shares: dict[str, float],
    simulated_shares: dict[str, float],
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for route in ROUTE_ORDER:
        rows.append(
            {
                "Scenario": (
                    "Observed held-out route mix"
                ),
                "Route": route,
                "Share": observed_shares[
                    route
                ],
            }
        )

        rows.append(
            {
                "Scenario": (
                    "Simulated missingness"
                ),
                "Route": route,
                "Share": simulated_shares[
                    route
                ],
            }
        )

    return pd.DataFrame(rows)


def render_missing_modality_stress_test(
    route_table: pd.DataFrame,
) -> None:
    observed_shares = (
        _observed_route_shares(
            route_table
        )
    )

    observed_rhr_missing = (
        observed_shares[
            "Temperature expert"
        ]
        + observed_shares[
            "Neither sensor"
        ]
    )

    observed_temperature_missing = (
        observed_shares[
            "RHR expert"
        ]
        + observed_shares[
            "Neither sensor"
        ]
    )

    st.html(
        """
<section style="
    margin:1.3rem 0 1.4rem;
    padding:1.4rem 1.5rem;
    border:1px solid #E8D3DD;
    border-radius:22px;
    background:
        linear-gradient(
            120deg,
            rgba(237,247,245,0.98),
            rgba(244,241,253,0.98) 52%,
            rgba(255,247,235,0.98)
        );
">
    <div style="
        color:#71324F;
        font-size:0.78rem;
        font-weight:750;
        letter-spacing:0.06em;
        text-transform:uppercase;
    ">
        Missing-modality stress simulation
    </div>

    <div style="
        margin-top:0.35rem;
        color:#352631;
        font-size:1.35rem;
        font-weight:780;
        letter-spacing:-0.025em;
    ">
        What happens as wearable signals disappear?
    </div>

    <div style="
        max-width:920px;
        margin-top:0.45rem;
        color:#6F5964;
        font-size:0.92rem;
        line-height:1.6;
    ">
        Adjust hypothetical sensor-missingness rates to
        explore how often each MARA route would be used.
        The simulation assumes RHR and temperature
        missingness occur independently.
    </div>
</section>
"""
    )

    control_columns = st.columns(2)

    with control_columns[0]:
        rhr_missing_percent = st.slider(
            "Hypothetical RHR missingness",
            min_value=0,
            max_value=100,
            value=int(
                round(
                    100.0
                    * observed_rhr_missing
                )
            ),
            step=1,
            format="%d%%",
            key="stress_rhr_missingness",
        )

    with control_columns[1]:
        temperature_missing_percent = (
            st.slider(
                (
                    "Hypothetical wrist-temperature "
                    "missingness"
                ),
                min_value=0,
                max_value=100,
                value=int(
                    round(
                        100.0
                        * observed_temperature_missing
                    )
                ),
                step=1,
                format="%d%%",
                key=(
                    "stress_temperature_"
                    "missingness"
                ),
            )
        )

    simulated_shares = (
        _simulated_route_shares(
            rhr_missing_rate=(
                rhr_missing_percent
                / 100.0
            ),
            temperature_missing_rate=(
                temperature_missing_percent
                / 100.0
            ),
        )
    )

    comparison = (
        _route_comparison_frame(
            observed_shares,
            simulated_shares,
        )
    )

    route_color_range = [
        ROUTE_COLORS[
            route
        ]
        for route in ROUTE_ORDER
    ]

    route_chart = (
        alt.Chart(comparison)
        .mark_bar(
            size=46,
        )
        .encode(
            y=alt.Y(
                "Scenario:N",
                title=None,
                sort=[
                    (
                        "Observed held-out "
                        "route mix"
                    ),
                    (
                        "Simulated "
                        "missingness"
                    ),
                ],
                axis=alt.Axis(
                    labelColor=MARA_INK,
                    labelFontSize=12,
                    labelPadding=10,
                    domain=False,
                    ticks=False,
                ),
            ),
            x=alt.X(
                "Share:Q",
                title="Share of days",
                stack="normalize",
                axis=alt.Axis(
                    format=".0%",
                    labelColor=MARA_MUTED,
                    titleColor=MARA_MUTED,
                    gridColor=MARA_GRID,
                    domain=False,
                ),
            ),
            color=alt.Color(
                "Route:N",
                title=None,
                scale=alt.Scale(
                    domain=ROUTE_ORDER,
                    range=route_color_range,
                ),
                legend=alt.Legend(
                    orient="bottom",
                    columns=2,
                    labelColor=MARA_MUTED,
                ),
            ),
            order=alt.Order(
                "Route:N",
                sort="ascending",
            ),
            tooltip=[
                alt.Tooltip(
                    "Scenario:N",
                    title="Scenario",
                ),
                alt.Tooltip(
                    "Route:N",
                    title="Route",
                ),
                alt.Tooltip(
                    "Share:Q",
                    format=".2%",
                    title="Expected share",
                ),
            ],
        )
        .properties(
            height=180,
            title={
                "text": (
                    "Observed and simulated "
                    "sensor-route distribution"
                ),
                "subtitle": (
                    "The simulated row changes "
                    "with the two missingness controls."
                ),
                "color": MARA_INK,
                "subtitleColor": MARA_MUTED,
                "fontSize": 17,
                "subtitleFontSize": 12,
                "anchor": "start",
            },
        )
        .configure_view(
            stroke=None
        )
    )

    st.altair_chart(
        route_chart,
        width="stretch",
        theme=None,
    )

    route_metric_columns = st.columns(4)

    for column, route in zip(
        route_metric_columns,
        ROUTE_ORDER,
    ):
        column.metric(
            route,
            percentage(
                simulated_shares[
                    route
                ]
            ),
        )

    reweighted = _reweighted_metrics(
        route_table,
        simulated_shares,
    )

    st.markdown(
        "#### Aggregate route-mix reweighting"
    )

    projected_columns = st.columns(3)

    projected_columns[0].metric(
        "Reweighted decision coverage",
        percentage(
            reweighted[
                "Decision coverage"
            ]
        ),
    )

    projected_columns[1].metric(
        "Reweighted pair-set coverage",
        percentage(
            reweighted[
                "Pair-set coverage"
            ]
        ),
        "Among estimated decisions",
    )

    projected_columns[2].metric(
        "Reweighted no-call rate",
        percentage(
            reweighted[
                "No-call rate"
            ]
        ),
    )

    route_sizes = (
        route_table[
            [
                "Sensor route",
                "Participant-days",
            ]
        ]
        .copy()
    )

    route_sizes[
        "Participant-days"
    ] = pd.to_numeric(
        route_sizes[
            "Participant-days"
        ],
        errors="coerce",
    ).fillna(0.0)

    observed_lookup = (
        observed_shares
    )

    sparse_routes: list[str] = []

    for _, row in route_sizes.iterrows():
        route = str(
            row["Sensor route"]
        )

        participant_days = float(
            row[
                "Participant-days"
            ]
        )

        simulated_share = (
            simulated_shares.get(
                route,
                0.0,
            )
        )

        observed_share = (
            observed_lookup.get(
                route,
                0.0,
            )
        )

        if (
            participant_days < 100
            and simulated_share
            > observed_share + 0.05
        ):
            sparse_routes.append(
                route
            )

    if sparse_routes:
        route_names = ", ".join(
            sparse_routes
        )

        st.warning(
            f"""
            This scenario increases reliance on route(s)
            with fewer than 100 held-out participant-days:
            {route_names}.

            The reweighted metrics may therefore be
            particularly unstable.
            """
        )

    st.info(
        """
        The reweighted metrics are a descriptive stress
        estimate. They assume each route keeps the same
        aggregate held-out performance while only the route
        mixture changes. Real performance may change under
        different devices, populations, missingness patterns,
        or signal quality.
        """
    )

    with st.expander(
        "How the stress simulation is calculated"
    ):
        st.markdown(
            """
            Under the independence assumption:

            - **Equal fusion**  
              `(1 − RHR missingness) × (1 − temperature missingness)`

            - **RHR expert**  
              `(1 − RHR missingness) × temperature missingness`

            - **Temperature expert**  
              `RHR missingness × (1 − temperature missingness)`

            - **Neither sensor**  
              `RHR missingness × temperature missingness`

            Route-specific aggregate benchmark results are
            then reweighted using the simulated route shares.
            The frozen model and threshold are not changed.
            """
        )


def render_robustness_dashboard(
    route_table: pd.DataFrame,
    phase_table: pd.DataFrame,
) -> None:
    st.subheader(
        "Robustness heatmaps"
    )

    st.write(
        """
        Compare decision frequency, phase-pair inclusion
        and abstention across true phases and sensor routes.
        """
    )

    phase_tab, route_tab = st.tabs(
        [
            "By true phase",
            "By sensor route",
        ]
    )

    with phase_tab:
        render_robustness_heatmap(
            dataframe=phase_table,
            category_column="True phase",
            category_order=[
                "Menstrual",
                "Follicular",
                "Fertility",
                "Luteal",
            ],
            title=(
                "Held-out robustness by true phase"
            ),
            subtitle=(
                "Performance is not uniform "
                "across the four phase labels."
            ),
        )

    with route_tab:
        render_robustness_heatmap(
            dataframe=route_table,
            category_column="Sensor route",
            category_order=ROUTE_ORDER,
            title=(
                "Held-out robustness by sensor route"
            ),
            subtitle=(
                "Missing modalities change both "
                "route frequency and reliability."
            ),
        )

        render_route_sample_warning(
            route_table
        )

    st.divider()

    render_missing_modality_stress_test(
        route_table
    )