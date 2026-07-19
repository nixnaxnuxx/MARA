from __future__ import annotations

from typing import Any

import streamlit as st


def percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def apply_mara_theme() -> None:
    st.html(
    """
    <style>
        :root {
            --mara-berry: #71324f;
            --mara-deep: #402432;
            --mara-pink: #d65d8b;
            --mara-rose: #e994b2;
            --mara-soft-rose: #f6cbd9;
            --mara-blush: #fff1f6;
            --mara-cream: #fff9fb;
            --mara-white: #ffffff;
            --mara-muted: #7d6570;
            --mara-border: #edd5df;
            --mara-teal: #477d78;
        }

        .mara-hero,
        .mara-hero *,
        .mara-metric-grid,
        .mara-metric-grid *,
        .method-card {
            box-sizing: border-box;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 92% 5%,
                    rgba(233, 148, 178, 0.17),
                    transparent 28rem
                ),
                radial-gradient(
                    circle at 3% 35%,
                    rgba(246, 203, 217, 0.20),
                    transparent 25rem
                ),
                var(--mara-cream);
            color: var(--mara-deep);
        }

        header[data-testid="stHeader"] {
            background: rgba(255, 249, 251, 0.88);
            backdrop-filter: blur(10px);
        }

        .block-container {
            max-width: 1260px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }
        
        .method-card {
            min-height: 175px;
            padding: 1.25rem;
            border: 1px solid var(--mara-border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 10px 28px rgba(113, 50, 79, 0.06);
            color: var(--mara-deep);
            line-height: 1.55;
        }
        
        .method-card b {
            color: var(--mara-berry);
            font-size: 1.02rem;
        }

        h1,
        h2,
        h3 {
            color: var(--mara-deep);
            letter-spacing: -0.025em;
        }

        p {
            color: var(--mara-deep);
        }

        .mara-hero {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.75fr);
            gap: 2rem;
            align-items: center;
            min-height: 420px;
            padding: 3rem;
            margin-bottom: 1rem;
            border: 1px solid var(--mara-border);
            border-radius: 30px;
            background:
                linear-gradient(
                    115deg,
                    rgba(64, 36, 50, 0.97),
                    rgba(113, 50, 79, 0.93) 52%,
                    rgba(214, 93, 139, 0.86)
                );
            box-shadow:
                0 24px 70px rgba(113, 50, 79, 0.16);
        }

        .mara-hero::after {
            content: "";
            position: absolute;
            width: 420px;
            height: 420px;
            right: -150px;
            top: -210px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
        }

        .mara-hero-copy {
            position: relative;
            z-index: 2;
        }

        .mara-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.8rem;
            margin-bottom: 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.30);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.11);
            color: #ffffff;
            font-size: 0.84rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .mara-hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: clamp(3.5rem, 8vw, 6rem);
            line-height: 0.95;
            letter-spacing: -0.065em;
        }

        .mara-full-title {
            max-width: 790px;
            margin-top: 1rem;
            color: #ffffff;
            font-size: 1.27rem;
            font-weight: 600;
            line-height: 1.45;
        }

        .mara-description {
            max-width: 720px;
            margin-top: 1rem;
            color: rgba(255, 255, 255, 0.82);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .mara-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.5rem;
        }

        .mara-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .mara-chip-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #f6cbd9;
        }

        .mara-visual {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 300px;
        }

        .cycle-orbit {
            display: grid;
            place-items: center;
            width: 255px;
            height: 255px;
            padding: 22px;
            border-radius: 50%;
            background:
                conic-gradient(
                    from -40deg,
                    #f4b9cb 0deg 72deg,
                    #e994b2 72deg 172deg,
                    #d65d8b 172deg 252deg,
                    #9f456c 252deg 360deg
                );
            box-shadow:
                0 24px 50px rgba(44, 20, 31, 0.24),
                inset 0 0 0 1px rgba(255, 255, 255, 0.35);
        }

        .cycle-center {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            padding: 1.25rem;
            border-radius: 50%;
            background: rgba(64, 36, 50, 0.96);
            color: #ffffff;
            text-align: center;
        }

        .cycle-center strong {
            color: #ffffff;
            font-size: 1.25rem;
        }

        .cycle-center span {
            margin-top: 0.35rem;
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .mara-visual-note {
            max-width: 290px;
            margin-top: 1.2rem;
            color: rgba(255, 255, 255, 0.76);
            font-size: 0.82rem;
            line-height: 1.5;
            text-align: center;
        }

        .mara-metric-grid {
            position: relative;
            z-index: 4;
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: -2.7rem 1.4rem 2.6rem;
        }

        .mara-metric-card {
            padding: 1.2rem 1.25rem;
            border: 1px solid var(--mara-border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.97);
            box-shadow: 0 14px 35px rgba(113, 50, 79, 0.10);
        }

        .mara-metric-label {
            color: var(--mara-muted);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.045em;
            text-transform: uppercase;
        }

        .mara-metric-value {
            margin-top: 0.35rem;
            color: var(--mara-deep);
            font-size: 1.8rem;
            font-weight: 750;
            letter-spacing: -0.04em;
        }

        .mara-metric-caption {
            margin-top: 0.3rem;
            color: var(--mara-muted);
            font-size: 0.79rem;
        }

        .mara-section-intro {
            max-width: 760px;
            color: var(--mara-muted);
            font-size: 1.02rem;
            line-height: 1.7;
        }
        
        .mara-creator {
            margin-top: 0.55rem;
            color: var(--mara-berry);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: grid;
            grid-template-columns:
                repeat(5, minmax(0, 1fr));
            gap: 0.65rem;
            padding: 0.7rem;
            margin: 0.35rem 0 2rem;
            border: 1px solid var(--mara-border);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow:
                0 12px 32px rgba(113, 50, 79, 0.08);
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"] {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 3.35rem;
            padding: 0.75rem 0.85rem;
            border: 1px solid var(--mara-border);
            border-radius: 15px;
            background: #ffffff;
            cursor: pointer;
            text-align: center;
            transition:
                transform 160ms ease,
                border-color 160ms ease,
                background 160ms ease,
                box-shadow 160ms ease;
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]:hover {
            transform: translateY(-2px);
            border-color: var(--mara-pink);
            background: var(--mara-blush);
            box-shadow:
                0 9px 22px rgba(113, 50, 79, 0.10);
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]:has(
            input:checked
        ) {
            border-color: var(--mara-berry);
            background:
                linear-gradient(
                    135deg,
                    var(--mara-berry),
                    var(--mara-pink)
                );
            box-shadow:
                0 11px 27px rgba(113, 50, 79, 0.20);
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]:has(
            input:checked
        ) p {
            color: #ffffff !important;
            font-weight: 750 !important;
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]
        p {
            color: var(--mara-deep);
            font-size: 0.91rem;
            font-weight: 650;
            line-height: 1.25;
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]
        > div:first-child {
            display: none;
        }
        
        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label[data-baseweb="radio"]:focus-within {
            outline:
                3px solid rgba(214, 93, 139, 0.35);
            outline-offset: 2px;
        }

        .product-overview-kicker {
            margin-bottom: 1.2rem;
            border-color: var(--mara-border);
            background: var(--mara-blush);
            color: var(--mara-berry);
        }
        
        @media (max-width: 960px) {
            div[data-testid="stRadio"]
            > div[role="radiogroup"] {
                grid-template-columns:
                    repeat(2, minmax(0, 1fr));
            }
        }
        
        @media (max-width: 560px) {
            div[data-testid="stRadio"]
            > div[role="radiogroup"] {
                grid-template-columns: 1fr;
            }
        
            div[data-testid="stRadio"]
            div[role="radiogroup"]
            label[data-baseweb="radio"] {
                min-height: 3rem;
            }
        }

        div[data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.1rem;
            border: 1px solid var(--mara-border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 10px 28px rgba(113, 50, 79, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--mara-muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--mara-berry);
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border-width: 1px;
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--mara-border);
            border-radius: 16px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--mara-border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.74);
        }

        .stButton > button {
            min-height: 2.75rem;
            border-radius: 999px;
            border-color: var(--mara-pink);
            font-weight: 650;
        }

        .stButton > button:hover {
            border-color: var(--mara-berry);
            color: var(--mara-berry);
        }

        .mara-footer {
            margin-top: 4rem;
            padding: 1.4rem 0;
            border-top: 1px solid var(--mara-border);
            color: var(--mara-muted);
            font-size: 0.82rem;
            text-align: center;
        }
        
        .method-card,
        .mara-metric-card {
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
        }
        
        .method-card:hover,
        .mara-metric-card:hover {
            transform: translateY(-3px);
            border-color: var(--mara-rose);
            box-shadow:
                0 16px 38px rgba(113, 50, 79, 0.12);
        }
            
        @media (max-width: 900px) {
            .mara-hero {
                grid-template-columns: 1fr;
                padding: 2rem;
            }

            .mara-visual {
                min-height: auto;
            }

            .cycle-orbit {
                width: 205px;
                height: 205px;
            }

            .mara-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 1rem;
            }
        }

        @media (max-width: 560px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .mara-hero {
                padding: 1.5rem;
                border-radius: 22px;
            }

            .mara-hero h1 {
                font-size: 3.6rem;
            }

            .mara-full-title {
                font-size: 1.05rem;
            }

            .mara-metric-grid {
                grid-template-columns: 1fr;
                margin-left: 0;
                margin-right: 0;
            }
        }
        
        button:focus-visible,
        a:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        select:focus-visible,
        [role="radio"]:focus-visible {
            outline: 3px solid #71324F !important;
            outline-offset: 3px !important;
        }
        
        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }

        .quick-navigation-header {
            margin: 1.25rem 0 1rem;
        }
        
        .quick-navigation-kicker {
            color: var(--mara-pink);
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        
        .quick-navigation-title {
            margin-top: 0.2rem;
            color: var(--mara-deep);
            font-size: 1.25rem;
            font-weight: 780;
            letter-spacing: -0.025em;
        }
        
        .quick-navigation-copy {
            margin-top: 0.25rem;
            color: var(--mara-muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        
        .st-key-quick_nav_product_overview button,
        .st-key-quick_nav_held_out_benchmark button,
        .st-key-quick_nav_sensor_routing_lab button,
        .st-key-quick_nav_robustness_audit button,
        .st-key-quick_nav_mara_ai_guide button {
            min-height: 4rem;
            padding: 0.9rem 1rem;
            border: 1px solid var(--mara-border) !important;
            border-radius: 16px !important;
            background: #ffffff !important;
            color: var(--mara-berry) !important;
            font-size: 0.93rem;
            font-weight: 750;
            line-height: 1.25;
            white-space: normal;
            box-shadow:
                0 8px 22px rgba(113, 50, 79, 0.08);
            transition:
                transform 160ms ease,
                border-color 160ms ease,
                background 160ms ease,
                color 160ms ease,
                box-shadow 160ms ease;
        }
        
        .st-key-quick_nav_product_overview button:hover,
        .st-key-quick_nav_held_out_benchmark button:hover,
        .st-key-quick_nav_sensor_routing_lab button:hover,
        .st-key-quick_nav_robustness_audit button:hover,
        .st-key-quick_nav_mara_ai_guide button:hover {
            transform: translateY(-2px);
            border-color: var(--mara-pink) !important;
            background: var(--mara-blush) !important;
            color: var(--mara-berry) !important;
            box-shadow:
                0 13px 28px rgba(113, 50, 79, 0.14);
        }
        
        .st-key-quick_nav_product_overview button:disabled,
        .st-key-quick_nav_held_out_benchmark button:disabled,
        .st-key-quick_nav_sensor_routing_lab button:disabled,
        .st-key-quick_nav_robustness_audit button:disabled,
        .st-key-quick_nav_mara_ai_guide button:disabled {
            border-color: var(--mara-berry) !important;
            background:
                linear-gradient(
                    135deg,
                    var(--mara-berry),
                    var(--mara-pink)
                ) !important;
            color: #ffffff !important;
            opacity: 1 !important;
            cursor: default !important;
            transform: none !important;
            box-shadow:
                0 11px 26px rgba(113, 50, 79, 0.17);
        }

        .st-key-synthetic_journey_metrics
        [data-testid="stMetric"] {
            min-height: 8.8rem;
            padding: 1rem 1.05rem;
        }
        
        .st-key-synthetic_journey_metrics
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            line-height: 1.25 !important;
        }
        
        .st-key-synthetic_journey_metrics
        [data-testid="stMetricValue"] {
            width: 100% !important;
            font-size: clamp(
                1.35rem,
                1.8vw,
                1.4rem
            ) !important;
            line-height: 1.12 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
        }
        
        .st-key-synthetic_journey_metrics
        [data-testid="stMetricValue"] > div {
            font-size: inherit !important;
            line-height: inherit !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
        }

        </style>
        """,
    )


def render_mara_hero(
    summary: dict[str, Any],
) -> None:
    model = summary["model"]

    threshold = percentage(
        model["top_two_threshold"]
    )

    hero_html = f"""
<section class="mara-hero">
    <div class="mara-hero-copy">
        <div class="mara-kicker">
            Challenge 5 · Women&#39;s Hormonal Health
        </div>
        <h1>MARA</h1>
        <div class="mara-full-title">
            Missing-Modality-Aware Routing with Abstention
            for Wearable Menstrual Phase Estimation
        </div>
        <div class="mara-description">
            MARA combines resting-heart-rate and wrist-temperature
            evidence, adapts when wearable sensors are missing,
            and abstains when the available evidence is insufficient.
        </div>
        <div class="mara-chip-row">
            <span class="mara-chip">
                <span class="mara-chip-dot"></span>
                Participant-level evaluation
            </span>
            <span class="mara-chip">
                <span class="mara-chip-dot"></span>
                Missing-sensor routing
            </span>
            <span class="mara-chip">
                <span class="mara-chip-dot"></span>
                Two-phase or no-call
            </span>
        </div>
    </div>
    <div class="mara-visual">
        <div class="cycle-orbit">
            <div class="cycle-center">
                <strong>Four phases</strong>
                <span>
                    Menstrual · Follicular<br>
                    Fertility · Luteal
                </span>
            </div>
        </div>
        <div class="mara-visual-note">
            MARA returns a compact phase pair only when
            combined evidence reaches the frozen
            {threshold} threshold.
        </div>
    </div>
</section>
"""

    st.html(hero_html)


def render_mara_metric_strip(
    summary: dict[str, Any],
) -> None:
    evaluation = summary["evaluation"]
    results = summary["overall_test_results"]

    participants = evaluation[
        "held_out_participants"
    ]

    days = evaluation[
        "held_out_participant_days"
    ]

    sensor_coverage = percentage(
        results["sensor_coverage"]
    )

    set_coverage = percentage(
        results["final_pair_set_coverage"]
    )

    metric_html = f"""
    <div class="mara-metric-grid">
        <div class="mara-metric-card">
            <div class="mara-metric-label">
                Unseen participants
            </div>
            <div class="mara-metric-value">
                {participants}
            </div>
            <div class="mara-metric-caption">
                Participant-level held-out test
            </div>
        </div>
    
        <div class="mara-metric-card">
            <div class="mara-metric-label">
                Held-out days
            </div>
            <div class="mara-metric-value">
                {days}
            </div>
            <div class="mara-metric-caption">
                Frozen one-time evaluation
            </div>
        </div>
    
        <div class="mara-metric-card">
            <div class="mara-metric-label">
                Sensor availability
            </div>
            <div class="mara-metric-value">
                {sensor_coverage}
            </div>
            <div class="mara-metric-caption">
                At least one predictive sensor
            </div>
        </div>
    
        <div class="mara-metric-card">
            <div class="mara-metric-label">
                Pair-set coverage
            </div>
            <div class="mara-metric-value">
                {set_coverage}
            </div>
            <div class="mara-metric-caption">
                Truth inside decided phase sets
            </div>
        </div>
    </div>
    """
    
    st.html(metric_html)


def render_mara_footer() -> None:
    footer_html = """
    <div class="mara-footer">
        <div>
            MARA is a research prototype for wearable-model robustness.
            It is not intended for diagnosis, ovulation confirmation,
            fertility decisions, contraception, or treatment.
        </div>
    
        <div class="mara-creator">
            FAA
        </div>
    </div>
    """

    st.html(footer_html)