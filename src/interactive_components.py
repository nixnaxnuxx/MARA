from __future__ import annotations

import html
import math
from typing import Sequence

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PHASE_COLORS = {
    "Menstrual": "#D65D8B",
    "Follicular": "#8B7BBE",
    "Fertility": "#F08A7C",
    "Luteal": "#4F8F88",
}


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def apply_interactive_styles() -> None:
    st.html(
        """
        <style>
        .mara-live-panel {
            margin: 1.2rem 0 1.8rem;
            padding: 1.4rem;
            border: 1px solid #edd5df;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.94);
        }

        .mara-live-panel-title {
            margin-bottom: 1rem;
            color: #402432;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .mara-route-grid {
            display: grid;
            grid-template-columns:
                minmax(180px, 1fr)
                50px
                minmax(180px, 1fr)
                50px
                minmax(180px, 1fr);
            gap: 0.7rem;
            align-items: center;
        }

        .mara-expert-stack {
            display: grid;
            gap: 0.75rem;
        }

        .mara-route-node.active {
            border-color: #4f8f88;
            background:
                linear-gradient(
                    135deg,
                    #f3fbf9,
                    #e8f6f3
                );
            box-shadow:
                0 9px 25px rgba(79, 143, 136, 0.12);
        }

        .mara-route-node.active {
            border-color: #d65d8b;
            background: #fff1f6;
            box-shadow: 0 9px 25px rgba(113, 50, 79, 0.08);
        }

        .mara-route-node.inactive {
            opacity: 0.38;
            filter: grayscale(0.35);
        }

        .mara-route-node.output {
            border-color: #d65d8b;
            background:
                linear-gradient(
                    135deg,
                    #fff1f6,
                    #fce4ed
                );
            box-shadow:
                0 9px 25px rgba(214, 93, 139, 0.12);
        }

        .mara-route-node.no-call {
            border-style: dashed;
            border-color: #d99a45;
            background: #fff8ec;
            box-shadow:
                0 9px 25px rgba(217, 154, 69, 0.10);
        }

        .mara-route-label {
            color: #71324f;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.045em;
            text-transform: uppercase;
        }

        .mara-route-value {
            margin-top: 0.3rem;
            color: #402432;
            font-size: 1.02rem;
            font-weight: 700;
        }

        .mara-route-detail {
            margin-top: 0.35rem;
            color: #7d6570;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .mara-route-arrow {
            color: #d65d8b;
            font-size: 1.65rem;
            font-weight: 700;
            text-align: center;
        }

        .mara-decision-panel {
            height: 100%;
            padding: 1.35rem;
            border: 1px solid #edd5df;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.96);
        }

        .mara-result-badge {
            display: inline-flex;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: #fce4ed;
            color: #71324f;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .mara-result-badge.no-call {
            background: #fff0d7;
            color: #8a5a19;
        }

        .mara-decision-title {
            margin-top: 0.8rem;
            color: #402432;
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.025em;
        }

        .mara-decision-copy {
            margin-top: 0.6rem;
            color: #65505a;
            font-size: 0.93rem;
            line-height: 1.6;
        }

        .mara-evidence-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 1.1rem;
            color: #65505a;
            font-size: 0.84rem;
        }

        .mara-evidence-track {
            width: 100%;
            height: 12px;
            margin-top: 0.5rem;
            overflow: hidden;
            border-radius: 999px;
            background: #f0e3e8;
        }

        .mara-evidence-fill {
            height: 100%;
            border-radius: inherit;
            background:
                linear-gradient(
                    90deg,
                    #4f8f88,
                    #8b7bbe 48%,
                    #d65d8b 78%,
                    #71324f
                );
            transition: width 260ms ease;
        }

        .mara-threshold-note {
            margin-top: 0.5rem;
            color: #7d6570;
            font-size: 0.8rem;
        }

        .mara-explanation-list {
            margin: 1rem 0 0;
            padding-left: 1.15rem;
            color: #65505a;
            font-size: 0.88rem;
            line-height: 1.65;
        }

        .mara-wheel-shell {
            padding: 1rem;
            border: 1px solid #edd5df;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.96);
            text-align: center;
        }

        .mara-wheel-caption {
            margin-top: 0.5rem;
            color: #7d6570;
            font-size: 0.8rem;
            line-height: 1.45;
        }

        @media (max-width: 850px) {
            .mara-route-grid {
                grid-template-columns: 1fr;
            }

        .mara-route-arrow {
            color: #8b7bbe;
            }
        }
        </style>
        """
    )


def render_routing_diagram(
    rhr_enabled: bool,
    temperature_enabled: bool,
    route_name: str,
) -> None:
    rhr_class = (
        "active"
        if rhr_enabled
        else "inactive"
    )

    temperature_class = (
        "active"
        if temperature_enabled
        else "inactive"
    )

    if route_name == "Equal fusion":
        router_detail = (
            "Both expert probability vectors are "
            "averaged equally."
        )
        output_label = "Fused evidence"
        output_class = "output"

    elif route_name == "RHR expert":
        router_detail = (
            "The temperature branch is unavailable. "
            "The RHR expert is used alone."
        )
        output_label = "RHR evidence"
        output_class = "output"

    elif route_name == "Temperature expert":
        router_detail = (
            "The RHR branch is unavailable. "
            "The temperature expert is used alone."
        )
        output_label = "Temperature evidence"
        output_class = "output"

    else:
        router_detail = (
            "No predictive branch is available."
        )
        output_label = "Immediate no-call"
        output_class = "no-call"

    route_name_safe = html.escape(route_name)
    router_detail_safe = html.escape(router_detail)
    output_label_safe = html.escape(output_label)

    diagram_html = f"""
<section class="mara-live-panel">
    <div class="mara-live-panel-title">
        Live missing-sensor routing path
    </div>

    <div class="mara-route-grid">
        <div class="mara-expert-stack">
            <div class="mara-route-node {rhr_class}">
                <div class="mara-route-label">
                    Wearable modality
                </div>
                <div class="mara-route-value">
                    Resting heart rate
                </div>
                <div class="mara-route-detail">
                    RHR expert and past-only temporal features
                </div>
            </div>

            <div class="mara-route-node {temperature_class}">
                <div class="mara-route-label">
                    Wearable modality
                </div>
                <div class="mara-route-value">
                    Wrist temperature
                </div>
                <div class="mara-route-detail">
                    Temperature expert and participant-relative history
                </div>
            </div>
        </div>

        <div class="mara-route-arrow" aria-hidden="true">
            →
        </div>

        <div class="mara-route-node active">
            <div class="mara-route-label">
                Active routing rule
            </div>
            <div class="mara-route-value">
                {route_name_safe}
            </div>
            <div class="mara-route-detail">
                {router_detail_safe}
            </div>
        </div>

        <div class="mara-route-arrow" aria-hidden="true">
            →
        </div>

        <div class="mara-route-node {output_class}">
            <div class="mara-route-label">
                Routed model evidence
            </div>
            <div class="mara-route-value">
                {output_label_safe}
            </div>
            <div class="mara-route-detail">
                Evidence is calibrated before applying
                the frozen selective-output threshold.
            </div>
        </div>
    </div>
</section>
"""

    st.html(diagram_html)


def _polar_point(
    center_x: float,
    center_y: float,
    radius: float,
    angle_degrees: float,
) -> tuple[float, float]:
    angle_radians = math.radians(
        angle_degrees
    )

    return (
        center_x
        + radius
        * math.cos(angle_radians),
        center_y
        + radius
        * math.sin(angle_radians),
    )


def _ring_segment_path(
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float,
    start_angle: float,
    end_angle: float,
) -> str:
    outer_start = _polar_point(
        center_x,
        center_y,
        outer_radius,
        start_angle,
    )

    outer_end = _polar_point(
        center_x,
        center_y,
        outer_radius,
        end_angle,
    )

    inner_end = _polar_point(
        center_x,
        center_y,
        inner_radius,
        end_angle,
    )

    inner_start = _polar_point(
        center_x,
        center_y,
        inner_radius,
        start_angle,
    )

    large_arc = (
        1
        if end_angle - start_angle > 180
        else 0
    )

    return (
        f"M {outer_start[0]:.3f} "
        f"{outer_start[1]:.3f} "
        f"A {outer_radius} {outer_radius} "
        f"0 {large_arc} 1 "
        f"{outer_end[0]:.3f} "
        f"{outer_end[1]:.3f} "
        f"L {inner_end[0]:.3f} "
        f"{inner_end[1]:.3f} "
        f"A {inner_radius} {inner_radius} "
        f"0 {large_arc} 0 "
        f"{inner_start[0]:.3f} "
        f"{inner_start[1]:.3f} Z"
    )


def render_phase_wheel(
    probabilities: np.ndarray,
    top_two_phases: Sequence[str],
    decision_made: bool,
) -> None:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.shape != (4,):
        raise ValueError(
            "The phase wheel expects exactly "
            "four probabilities."
        )

    maximum_probability = float(
        probabilities.max()
    )

    if maximum_probability <= 0:
        maximum_probability = 1.0

    selected_phases = set(
        top_two_phases
    )

    paths: list[str] = []
    labels: list[str] = []

    center_x = 160.0
    center_y = 160.0
    outer_radius = 126.0
    inner_radius = 76.0

    for index, phase in enumerate(PHASES):
        start_angle = (
            -90.0
            + index * 90.0
            + 2.5
        )

        end_angle = (
            -90.0
            + (index + 1) * 90.0
            - 2.5
        )

        path = _ring_segment_path(
            center_x=center_x,
            center_y=center_y,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        relative_probability = (
            float(probabilities[index])
            / maximum_probability
        )

        opacity = (
            0.36
            + 0.64
            * relative_probability
        )

        is_selected = (
            phase in selected_phases
        )

        if is_selected and decision_made:
            stroke = "#71324F"
            stroke_width = "6"
            dash_array = ""
        elif is_selected:
            stroke = "#8E7280"
            stroke_width = "4"
            dash_array = 'stroke-dasharray="7 6"'
        else:
            stroke = "#FFFFFF"
            stroke_width = "2"
            dash_array = ""

        phase_safe = html.escape(phase)
        probability_label = percent(
            probabilities[index]
        )

        paths.append(
            f"""
            <path
                d="{path}"
                fill="{PHASE_COLORS[phase]}"
                fill-opacity="{opacity:.3f}"
                stroke="{stroke}"
                stroke-width="{stroke_width}"
                {dash_array}
            >
                <title>
                    {phase_safe}: {probability_label}
                </title>
            </path>
            """
        )

        middle_angle = (
            start_angle + end_angle
        ) / 2.0

        label_x, label_y = _polar_point(
            center_x,
            center_y,
            103.0,
            middle_angle,
        )

        labels.append(
            f"""
            <text
                x="{label_x:.2f}"
                y="{label_y:.2f}"
                text-anchor="middle"
                dominant-baseline="middle"
                fill="#402432"
                font-size="12"
                font-weight="700"
            >
                {phase_safe}
            </text>
            """
        )

    if decision_made:
        center_title = "Phase pair"
        center_line_one = html.escape(
            str(top_two_phases[0])
        )
        center_line_two = html.escape(
            str(top_two_phases[1])
        )
    else:
        center_title = "No-call"
        center_line_one = "Evidence below"
        center_line_two = "frozen threshold"

    components.html(
        f"""
<style>
    html,
    body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: transparent;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    .wheel-shell {{
        padding: 14px;
        border: 1px solid #edd5df;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.96);
        text-align: center;
        box-sizing: border-box;
    }}

    .wheel-caption {{
        margin-top: 6px;
        color: #7d6570;
        font-size: 13px;
        line-height: 1.45;
    }}

    svg {{
        display: block;
        width: 100%;
        max-width: 360px;
        margin: auto;
    }}
</style>

<div class="wheel-shell">
    <svg
        viewBox="0 0 320 320"
        role="img"
        aria-label="Four-phase probability wheel"
    >
        {''.join(paths)}
        {''.join(labels)}

        <circle
            cx="160"
            cy="160"
            r="68"
            fill="#FFF9FB"
            stroke="#EDD5DF"
            stroke-width="2"
        />

        <text
            x="160"
            y="137"
            text-anchor="middle"
            fill="#71324F"
            font-size="13"
            font-weight="700"
        >
            {center_title}
        </text>

        <text
            x="160"
            y="161"
            text-anchor="middle"
            fill="#402432"
            font-size="14"
            font-weight="700"
        >
            {center_line_one}
        </text>

        <text
            x="160"
            y="181"
            text-anchor="middle"
            fill="#402432"
            font-size="14"
            font-weight="700"
        >
            {center_line_two}
        </text>
    </svg>

    <div class="wheel-caption">
        Segment intensity represents relative calibrated
        probability. Highlighted segments show the two
        leading phases.
    </div>
</div>
""",
        height=430,
        scrolling=False,
    )


def render_decision_explanation(
    route_name: str,
    top_two_phases: Sequence[str],
    top_two_probability: float,
    threshold: float,
    decision_made: bool,
    rhr_enabled: bool,
    temperature_enabled: bool,
) -> None:
    route_name_safe = html.escape(
        route_name
    )

    top_two_probability = float(
        top_two_probability
    )

    threshold = float(threshold)

    evidence_width = min(
        max(
            100.0 * top_two_probability,
            0.0,
        ),
        100.0,
    )

    if rhr_enabled and temperature_enabled:
        sensor_text = (
            "RHR and temperature evidence were "
            "combined using equal fusion."
        )
    elif rhr_enabled:
        sensor_text = (
            "Only the RHR expert was available."
        )
    elif temperature_enabled:
        sensor_text = (
            "Only the temperature expert was available."
        )
    else:
        sensor_text = (
            "Neither predictive sensor was available."
        )

    if decision_made:
        badge_class = ""
        badge_text = "Two-phase output"

        title = (
            f"{top_two_phases[0]} or "
            f"{top_two_phases[1]}"
        )

        margin = (
            top_two_probability
            - threshold
        )

        decision_copy = (
            "The two leading phases accumulated "
            f"{percent(top_two_probability)} calibrated "
            "evidence, exceeding the frozen "
            f"{percent(threshold)} threshold."
        )

        threshold_note = (
            f"Evidence margin: +{percent(margin)}"
        )
    else:
        badge_class = "no-call"
        badge_text = "Selective abstention"
        title = "MARA returned no-call"

        shortfall = (
            threshold
            - top_two_probability
        )

        decision_copy = (
            "The leading phase pair did not accumulate "
            "enough calibrated evidence for the frozen "
            "selective policy to issue a compact result."
        )

        threshold_note = (
            f"Evidence shortfall: {percent(shortfall)}"
        )

    title_safe = html.escape(title)
    decision_copy_safe = html.escape(
        decision_copy
    )
    sensor_text_safe = html.escape(
        sensor_text
    )
    threshold_note_safe = html.escape(
        threshold_note
    )

    explanation_html = f"""
<section class="mara-decision-panel">
    <div class="mara-result-badge {badge_class}">
        {badge_text}
    </div>

    <div class="mara-decision-title">
        {title_safe}
    </div>

    <div class="mara-decision-copy">
        {decision_copy_safe}
    </div>

    <div class="mara-evidence-row">
        <span>Combined top-two evidence</span>
        <strong>{percent(top_two_probability)}</strong>
    </div>

    <div
        class="mara-evidence-track"
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow="{evidence_width:.2f}"
    >
        <div
            class="mara-evidence-fill"
            style="width:{evidence_width:.2f}%;"
        ></div>
    </div>

    <div class="mara-threshold-note">
        Frozen threshold: {percent(threshold)}
        · {threshold_note_safe}
    </div>

    <ul class="mara-explanation-list">
        <li>
            Active route: <strong>{route_name_safe}</strong>
        </li>
        <li>
            {sensor_text_safe}
        </li>
        <li>
            Official single-phase outputs are disabled.
        </li>
        <li>
            The result is a research output, not an
            ovulation or fertility determination.
        </li>
    </ul>
</section>
"""

    st.html(explanation_html)