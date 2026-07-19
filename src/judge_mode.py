from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from interactive_components import (
    render_decision_explanation,
    render_phase_wheel,
    render_routing_diagram,
)


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

JUDGE_STEPS = [
    "Signals",
    "Routing",
    "Evidence",
    "Decision",
]

SCENARIOS: dict[str, dict[str, Any]] = {
    "complete_evidence": {
        "number": "01",
        "title": "Complete evidence",
        "short_title": "Both sensors available",
        "icon": "✦",
        "description": (
            "Resting heart rate and wrist temperature "
            "are both available and broadly agree."
        ),
        "rhr_enabled": True,
        "temperature_enabled": True,
        "rhr_probabilities": [
            0.08,
            0.20,
            0.57,
            0.15,
        ],
        "temperature_probabilities": [
            0.10,
            0.18,
            0.55,
            0.17,
        ],
        "takeaway": (
            "Both experts support a similar region of the "
            "phase space, allowing equal fusion to produce "
            "a compact two-phase output."
        ),
    },
    "missing_temperature": {
        "number": "02",
        "title": "Missing temperature",
        "short_title": "RHR-only route",
        "icon": "◌",
        "description": (
            "Wrist temperature is unavailable, so MARA "
            "routes through the RHR expert alone."
        ),
        "rhr_enabled": True,
        "temperature_enabled": False,
        "rhr_probabilities": [
            0.08,
            0.57,
            0.22,
            0.13,
        ],
        "temperature_probabilities": None,
        "takeaway": (
            "The system does not fail when one predictive "
            "sensor is missing. It changes route and applies "
            "the same calibrated selective-output policy."
        ),
    },
    "insufficient_evidence": {
        "number": "03",
        "title": "Insufficient evidence",
        "short_title": "Selective no-call",
        "icon": "◇",
        "description": (
            "Both sensors are available, but the expert "
            "probabilities remain diffuse and uncertain."
        ),
        "rhr_enabled": True,
        "temperature_enabled": True,
        "rhr_probabilities": [
            0.27,
            0.24,
            0.25,
            0.24,
        ],
        "temperature_probabilities": [
            0.24,
            0.26,
            0.25,
            0.25,
        ],
        "takeaway": (
            "Sensor availability alone is not enough. "
            "When calibrated evidence remains diffuse, "
            "MARA abstains rather than forcing a result."
        ),
    },
}


def normalize_probabilities(
    probabilities: list[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        probabilities,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    total = float(
        values.sum()
    )

    if total <= 0:
        return np.full(
            len(PHASES),
            1.0 / len(PHASES),
            dtype=float,
        )

    return values / total


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
        1.0 / float(temperature),
    )

    return (
        transformed
        / transformed.sum()
    )


def get_route_name(
    rhr_enabled: bool,
    temperature_enabled: bool,
) -> str:
    if (
        rhr_enabled
        and temperature_enabled
    ):
        return "Equal fusion"

    if rhr_enabled:
        return "RHR expert"

    if temperature_enabled:
        return "Temperature expert"

    return "Neither sensor"


def calculate_scenario(
    scenario: dict[str, Any],
    probability_temperature: float,
    threshold: float,
) -> dict[str, Any]:
    rhr_enabled = bool(
        scenario["rhr_enabled"]
    )

    temperature_enabled = bool(
        scenario[
            "temperature_enabled"
        ]
    )

    active_probabilities: list[
        np.ndarray
    ] = []

    rhr_probabilities = None
    temperature_probabilities = None

    if rhr_enabled:
        rhr_probabilities = (
            normalize_probabilities(
                scenario[
                    "rhr_probabilities"
                ]
            )
        )

        active_probabilities.append(
            rhr_probabilities
        )

    if temperature_enabled:
        temperature_probabilities = (
            normalize_probabilities(
                scenario[
                    "temperature_probabilities"
                ]
            )
        )

        active_probabilities.append(
            temperature_probabilities
        )

    route_name = get_route_name(
        rhr_enabled,
        temperature_enabled,
    )

    if not active_probabilities:
        routed_probabilities = np.full(
            len(PHASES),
            1.0 / len(PHASES),
            dtype=float,
        )

    elif len(active_probabilities) == 2:
        routed_probabilities = (
            0.5
            * active_probabilities[0]
            + 0.5
            * active_probabilities[1]
        )

    else:
        routed_probabilities = (
            active_probabilities[0]
        )

    calibrated_probabilities = (
        temperature_scale(
            routed_probabilities,
            probability_temperature,
        )
    )

    order = np.argsort(
        -calibrated_probabilities
    )

    top_two_phases = [
        PHASES[int(order[0])],
        PHASES[int(order[1])],
    ]

    top_two_probability = float(
        calibrated_probabilities[
            order[:2]
        ].sum()
    )

    decision_made = bool(
        route_name != "Neither sensor"
        and top_two_probability
        >= threshold
    )

    return {
        "route_name": route_name,
        "rhr_enabled": rhr_enabled,
        "temperature_enabled": (
            temperature_enabled
        ),
        "rhr_probabilities": (
            rhr_probabilities
        ),
        "temperature_probabilities": (
            temperature_probabilities
        ),
        "routed_probabilities": (
            routed_probabilities
        ),
        "calibrated_probabilities": (
            calibrated_probabilities
        ),
        "top_two_phases": (
            top_two_phases
        ),
        "top_two_probability": (
            top_two_probability
        ),
        "decision_made": (
            decision_made
        ),
        "threshold": float(
            threshold
        ),
    }


def _select_scenario(
    scenario_key: str,
) -> None:
    st.session_state[
        "mara_judge_scenario"
    ] = scenario_key

    st.session_state[
        "mara_judge_step"
    ] = 0


def _previous_step() -> None:
    current_step = int(
        st.session_state.get(
            "mara_judge_step",
            0,
        )
    )

    st.session_state[
        "mara_judge_step"
    ] = max(
        current_step - 1,
        0,
    )


def _next_step() -> None:
    current_step = int(
        st.session_state.get(
            "mara_judge_step",
            0,
        )
    )

    st.session_state[
        "mara_judge_step"
    ] = min(
        current_step + 1,
        len(JUDGE_STEPS) - 1,
    )


def _restart_demo() -> None:
    st.session_state[
        "mara_judge_step"
    ] = 0


def _probability_table_html(
    title: str,
    probabilities: np.ndarray | None,
    accent: str,
) -> str:
    if probabilities is None:
        return f"""
<div class="judge-probability-card unavailable">
    <div class="judge-probability-title">
        {title}
    </div>
    <div class="judge-unavailable">
        Sensor unavailable
    </div>
</div>
"""

    rows: list[str] = []

    for phase, probability in zip(
        PHASES,
        probabilities,
    ):
        width = min(
            max(
                100.0
                * float(probability),
                0.0,
            ),
            100.0,
        )

        rows.append(
            f"""
<div class="judge-probability-row">
    <div class="judge-probability-label">
        <span>{phase}</span>
        <strong>{100.0 * float(probability):.1f}%</strong>
    </div>
    <div class="judge-probability-track">
        <div
            class="judge-probability-fill"
            style="
                width:{width:.2f}%;
                background:{accent};
            "
        ></div>
    </div>
</div>
"""
        )

    return f"""
<div class="judge-probability-card">
    <div class="judge-probability-title">
        {title}
    </div>
    {''.join(rows)}
</div>
"""


@st.dialog(
    "Why did MARA do this?",
    width="medium",
    icon="🔎",
)
def show_judge_explanation(
    scenario: dict[str, Any],
    result: dict[str, Any],
) -> None:
    output_text = (
        (
            f"{result['top_two_phases'][0]} "
            f"or {result['top_two_phases'][1]}"
        )
        if result["decision_made"]
        else "No-call"
    )

    margin = (
        float(
            result[
                "top_two_probability"
            ]
        )
        - float(
            result["threshold"]
        )
    )

    st.markdown(
        f"""
### {scenario["title"]}

**1. Sensors checked**

- Resting heart rate: {
    "available"
    if result["rhr_enabled"]
    else "missing"
}
- Wrist temperature: {
    "available"
    if result["temperature_enabled"]
    else "missing"
}

**2. Route selected**

`{result["route_name"]}`

**3. Evidence calibrated**

The two leading phases accumulated
**{100.0 * result["top_two_probability"]:.2f}%**
of the calibrated probability.

**4. Frozen policy applied**

Threshold:
**{100.0 * result["threshold"]:.2f}%**

Evidence margin:
**{margin * 100.0:+.2f} percentage points**

### Output

**{output_text}**
"""
    )

    st.info(
        scenario[
            "takeaway"
        ]
    )

    st.warning(
        """
        This is a hypothetical demonstration of MARA's
        routing and abstention logic. It is not a personal
        menstrual, fertility, or ovulation assessment.
        """
    )


def apply_judge_mode_styles() -> None:
    st.html(
        """
<style>
.judge-mode-shell {
    margin: 2rem 0 1.5rem;
    padding: 1.6rem;
    border: 1px solid #e8d3dd;
    border-radius: 26px;
    background:
        radial-gradient(
            circle at 92% 12%,
            rgba(139,123,190,0.18),
            transparent 18rem
        ),
        linear-gradient(
            120deg,
            rgba(255,241,246,0.98),
            rgba(239,248,246,0.98)
        );
    box-shadow:
        0 18px 45px rgba(113,50,79,0.08);
}

.judge-mode-kicker {
    color: #71324f;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.065em;
    text-transform: uppercase;
}

.judge-mode-title {
    margin-top: 0.4rem;
    color: #352631;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.035em;
}

.judge-mode-copy {
    max-width: 860px;
    margin-top: 0.5rem;
    color: #6f5964;
    font-size: 0.94rem;
    line-height: 1.65;
}

.judge-stepper {
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 1.2rem 0;
}

.judge-step {
    padding: 0.75rem;
    border: 1px solid #e8d3dd;
    border-radius: 14px;
    background: rgba(255,255,255,0.72);
    color: #8a7480;
    font-size: 0.82rem;
    text-align: center;
}

.judge-step.active {
    border-color: #d65d8b;
    background: #fff0f5;
    color: #71324f;
    font-weight: 750;
    box-shadow:
        0 8px 20px rgba(214,93,139,0.10);
}

.judge-step.complete {
    border-color: #4f8f88;
    background: #edf8f5;
    color: #356b66;
    font-weight: 700;
}

.judge-scenario-summary {
    margin: 1rem 0;
    padding: 1.2rem 1.3rem;
    border-left: 5px solid #d65d8b;
    border-radius: 0 18px 18px 0;
    background: rgba(255,255,255,0.86);
}

.judge-scenario-label {
    color: #71324f;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.055em;
    text-transform: uppercase;
}

.judge-scenario-name {
    margin-top: 0.25rem;
    color: #352631;
    font-size: 1.25rem;
    font-weight: 780;
}

.judge-scenario-description {
    margin-top: 0.35rem;
    color: #6f5964;
    font-size: 0.9rem;
    line-height: 1.55;
}

.judge-sensor-grid {
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
}

.judge-sensor-card {
    padding: 1.2rem;
    border: 1px solid #e8d3dd;
    border-radius: 18px;
    background: rgba(255,255,255,0.92);
}

.judge-sensor-card.available {
    border-color: #4f8f88;
    background:
        linear-gradient(
            135deg,
            #f3fbf9,
            #e8f6f3
        );
}

.judge-sensor-card.missing {
    border-style: dashed;
    border-color: #d99a45;
    background: #fff8ec;
}

.judge-sensor-status {
    color: #6f5964;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.045em;
    text-transform: uppercase;
}

.judge-sensor-name {
    margin-top: 0.35rem;
    color: #352631;
    font-size: 1.08rem;
    font-weight: 750;
}

.judge-probability-grid {
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
}

.judge-probability-card {
    padding: 1.15rem;
    border: 1px solid #e8d3dd;
    border-radius: 18px;
    background: rgba(255,255,255,0.94);
}

.judge-probability-card.unavailable {
    display: grid;
    place-items: center;
    min-height: 260px;
    border-style: dashed;
    border-color: #d99a45;
    background: #fff8ec;
}

.judge-probability-title {
    margin-bottom: 0.9rem;
    color: #352631;
    font-size: 1rem;
    font-weight: 750;
}

.judge-probability-row {
    margin-top: 0.75rem;
}

.judge-probability-label {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: #6f5964;
    font-size: 0.82rem;
}

.judge-probability-track {
    height: 9px;
    margin-top: 0.35rem;
    overflow: hidden;
    border-radius: 999px;
    background: #f0e4e9;
}

.judge-probability-fill {
    height: 100%;
    border-radius: inherit;
    transition: width 240ms ease;
}

.judge-unavailable {
    color: #8a5a19;
    font-size: 0.9rem;
    font-weight: 700;
}

.judge-takeaway {
    margin-top: 1rem;
    padding: 1rem 1.15rem;
    border: 1px solid #d7cdec;
    border-radius: 16px;
    background: #f7f3ff;
    color: #574a78;
    font-size: 0.9rem;
    line-height: 1.6;
}

@media (max-width: 760px) {
    .judge-stepper,
    .judge-sensor-grid,
    .judge-probability-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""
    )


def render_stepper(
    current_step: int,
) -> None:
    steps: list[str] = []

    for index, step in enumerate(
        JUDGE_STEPS
    ):
        if index < current_step:
            css_class = (
                "judge-step complete"
            )
            prefix = "✓"

        elif index == current_step:
            css_class = (
                "judge-step active"
            )
            prefix = str(
                index + 1
            )

        else:
            css_class = "judge-step"
            prefix = str(
                index + 1
            )

        steps.append(
            f"""
<div class="{css_class}">
    {prefix} · {step}
</div>
"""
        )

    st.html(
        f"""
<div class="judge-stepper">
    {''.join(steps)}
</div>
"""
    )


def render_signal_step(
    scenario: dict[str, Any],
) -> None:
    rhr_class = (
        "available"
        if scenario["rhr_enabled"]
        else "missing"
    )

    temperature_class = (
        "available"
        if scenario[
            "temperature_enabled"
        ]
        else "missing"
    )

    rhr_status = (
        "Available"
        if scenario["rhr_enabled"]
        else "Missing"
    )

    temperature_status = (
        "Available"
        if scenario[
            "temperature_enabled"
        ]
        else "Missing"
    )

    st.html(
        f"""
<div class="judge-sensor-grid">
    <div class="judge-sensor-card {rhr_class}">
        <div class="judge-sensor-status">
            {rhr_status}
        </div>
        <div class="judge-sensor-name">
            Resting heart rate
        </div>
    </div>

    <div class="judge-sensor-card {temperature_class}">
        <div class="judge-sensor-status">
            {temperature_status}
        </div>
        <div class="judge-sensor-name">
            Wrist temperature
        </div>
    </div>
</div>
"""
    )

    st.info(
        """
        MARA first checks which predictive sensor
        families are available. Sleep and activity are
        not used to select the frozen predictive route.
        """
    )


def render_evidence_step(
    result: dict[str, Any],
) -> None:
    st.html(
        f"""
<div class="judge-probability-grid">
    {
        _probability_table_html(
            "RHR expert probabilities",
            result["rhr_probabilities"],
            "#4F8F88",
        )
    }

    {
        _probability_table_html(
            "Temperature expert probabilities",
            result[
                "temperature_probabilities"
            ],
            "#F08A7C",
        )
    }
</div>
"""
    )

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "Active route",
        result["route_name"],
    )

    metric_columns[1].metric(
        "Top-two evidence",
        (
            f"{100.0 * result['top_two_probability']:.2f}%"
        ),
    )

    metric_columns[2].metric(
        "Frozen threshold",
        (
            f"{100.0 * result['threshold']:.2f}%"
        ),
    )

    render_phase_wheel(
        probabilities=result[
            "calibrated_probabilities"
        ],
        top_two_phases=result[
            "top_two_phases"
        ],
        decision_made=result[
            "decision_made"
        ],
    )


def render_judge_mode(
    summary: dict[str, Any],
) -> None:
    apply_judge_mode_styles()

    if (
        "mara_judge_scenario"
        not in st.session_state
    ):
        st.session_state[
            "mara_judge_scenario"
        ] = "complete_evidence"

    if (
        "mara_judge_step"
        not in st.session_state
    ):
        st.session_state[
            "mara_judge_step"
        ] = 0

    model = summary["model"]

    probability_temperature = float(
        model[
            "probability_temperature"
        ]
    )

    threshold = float(
        model[
            "top_two_threshold"
        ]
    )

    st.html(
        """
<section class="judge-mode-shell">
    <div class="judge-mode-kicker">
        For Demonstration
    </div>

    <div class="judge-mode-title">
        See MARA route, calibrate and abstain
    </div>

    <div class="judge-mode-copy">
        Select one of three hypothetical scenarios, then
        move through the four stages of MARA's frozen
        decision pipeline. This demonstration uses no
        personal or restricted participant measurements.
    </div>
</section>
"""
    )

    scenario_columns = st.columns(3)

    selected_scenario_key = str(
        st.session_state[
            "mara_judge_scenario"
        ]
    )

    for column, (
        scenario_key,
        scenario,
    ) in zip(
        scenario_columns,
        SCENARIOS.items(),
    ):
        with column:
            is_selected = (
                scenario_key
                == selected_scenario_key
            )

            st.button(
                scenario["title"],
                key=(
                    "judge_scenario_"
                    + scenario_key
                ),
                type=(
                    "primary"
                    if is_selected
                    else "secondary"
                ),
                width="stretch",
                on_click=_select_scenario,
                args=(
                    scenario_key,
                ),
            )

            st.caption(
                scenario[
                    "short_title"
                ]
            )

    selected_scenario_key = str(
        st.session_state[
            "mara_judge_scenario"
        ]
    )

    scenario = SCENARIOS[
        selected_scenario_key
    ]

    result = calculate_scenario(
        scenario=scenario,
        probability_temperature=(
            probability_temperature
        ),
        threshold=threshold,
    )

    current_step = int(
        st.session_state[
            "mara_judge_step"
        ]
    )

    st.html(
        f"""
<div class="judge-scenario-summary">
    <div class="judge-scenario-label">
        Selected scenario
    </div>

    <div class="judge-scenario-name">
        {scenario["title"]}
    </div>

    <div class="judge-scenario-description">
        {scenario["description"]}
    </div>
</div>
"""
    )

    render_stepper(
        current_step
    )

    if current_step == 0:
        st.subheader(
            "Step 1 — Check available signals"
        )

        render_signal_step(
            scenario
        )

    elif current_step == 1:
        st.subheader(
            "Step 2 — Select the sensor route"
        )

        render_routing_diagram(
            rhr_enabled=result[
                "rhr_enabled"
            ],
            temperature_enabled=result[
                "temperature_enabled"
            ],
            route_name=result[
                "route_name"
            ],
        )

    elif current_step == 2:
        st.subheader(
            "Step 3 — Calibrate and rank evidence"
        )

        render_evidence_step(
            result
        )

    else:
        st.subheader(
            "Step 4 — Apply the selective policy"
        )

        decision_columns = st.columns(
            [
                1,
                1.15,
            ]
        )

        with decision_columns[0]:
            render_phase_wheel(
                probabilities=result[
                    "calibrated_probabilities"
                ],
                top_two_phases=result[
                    "top_two_phases"
                ],
                decision_made=result[
                    "decision_made"
                ],
            )

        with decision_columns[1]:
            render_decision_explanation(
                route_name=result[
                    "route_name"
                ],
                top_two_phases=result[
                    "top_two_phases"
                ],
                top_two_probability=result[
                    "top_two_probability"
                ],
                threshold=result[
                    "threshold"
                ],
                decision_made=result[
                    "decision_made"
                ],
                rhr_enabled=result[
                    "rhr_enabled"
                ],
                temperature_enabled=result[
                    "temperature_enabled"
                ],
            )

        st.html(
            f"""
<div class="judge-takeaway">
    <strong>Judge takeaway:</strong>
    {scenario["takeaway"]}
</div>
"""
        )

        if st.button(
            "Why did MARA do this?",
            key="judge_explanation_dialog",
            type="primary",
            icon="🔎",
            width="stretch",
        ):
            show_judge_explanation(
                scenario,
                result,
            )

    navigation_columns = st.columns(
        [
            1,
            1,
            2,
        ]
    )

    with navigation_columns[0]:
        st.button(
            "Back",
            key="judge_back",
            disabled=(
                current_step == 0
            ),
            on_click=_previous_step,
            width="stretch",
        )

    with navigation_columns[1]:
        if (
            current_step
            < len(JUDGE_STEPS) - 1
        ):
            st.button(
                "Next step",
                key="judge_next",
                type="primary",
                on_click=_next_step,
                width="stretch",
            )

        else:
            st.button(
                "Restart scenario",
                key="judge_restart",
                on_click=_restart_demo,
                width="stretch",
            )

    with navigation_columns[2]:
        st.caption(
            (
                f"Step {current_step + 1} "
                f"of {len(JUDGE_STEPS)} · "
                "Hypothetical research demonstration"
            )
        )