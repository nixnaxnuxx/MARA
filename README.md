# PhaseWise — MOSAIC-Phase

**Missing-sensor-aware menstrual-phase estimation with calibrated abstention**

PhaseWise is the interactive demonstration for **MOSAIC-Phase**, a
research prototype that estimates plausible menstrual-cycle phases from
passive wearable signals while explicitly handling missing sensors and
uncertain evidence.

> Research prototype only. It must not be used for diagnosis, fertility decisions, contraception, or treatment.

## Research question

How robust is menstrual-phase estimation when wearable modalities are
unavailable, and can calibrated uncertainty prevent unreliable outputs
for unseen participants?

## Approach

1. Train separate resting-heart-rate and wrist-temperature experts.
2. Activate only the experts supported by available sensors.
3. Fuse their probabilities equally when both sensors are available.
4. Apply probability temperature scaling learned from participant-grouped
   out-of-fold training predictions.
5. Return the two highest-ranked phases when their combined calibrated
   probability reaches the frozen threshold.
6. Return **no-call** otherwise.

## Frozen routing policy

- Both sensors: Equal probability fusion
- RHR only: Resting-heart-rate expert
- Temperature only: Temperature expert
- Neither sensor: No-call
- Probability temperature: `0.9012`
- Top-two evidence threshold: `0.62`
- Single-phase outputs enabled: `False`

## Evaluation design

- Dataset: mcPHASES
- Participants: 42
- Labelled participant-days: 5,658
- Target classes: Menstrual, Follicular, Fertility and Luteal
- Split unit: participant
- Held-out participants: 7
- Held-out participant-days: 959
- Model and policy frozen before opening the test set
- Test evaluation performed once

Hormones and direct menstrual-flow indicators were excluded from model
inputs to reduce target leakage.

## Frozen held-out results

| Metric | Result |
|---|---:|
| Predictive-sensor coverage | 97.29% |
| Unrestricted top-1 accuracy | 34.41% |
| Unrestricted top-1 balanced accuracy | 34.76% |
| Unrestricted top-1 macro F1 | 34.17% |
| Unrestricted top-2 coverage | 62.92% |
| Final two-phase decision coverage | 44.84% |
| Final pair-set coverage | 66.51% |
| Final no-call rate | 55.16% |

## Approved interpretation

> On unseen participants, the frozen policy returned a two-phase plausible set on 44.84% of participant-days. The true phase was contained in that set for 66.51% of those decisions.

> The system returned no-call on 55.16% of participant-days when predictive evidence was unavailable or insufficient.

The unrestricted top-1 prediction is benchmark context only. The
official system output is a two-phase plausible set or no-call.

## Interactive application

The Streamlit application includes:

- Research overview
- Frozen held-out benchmark
- Interactive sensor routing
- Hypothetical probability sandbox
- Study-interval robustness analysis
- Phase-level robustness analysis
- Leakage controls and limitations

The sandbox demonstrates the frozen decision policy. It does not infer
menstrual phase from a user's personal medical measurements.

## Run locally

Windows:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run src\app.py

macOS or Linux:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run src/app.py

## Public repository structure

    mosaic-phase/
    ├── app_assets/
    ├── docs/
    ├── src/
    │   └── app.py
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── requirements-research.txt

Raw data, processed participant-level tables, split assignments, model
objects and individual predictions are excluded from the public release.

## Main limitations

- Only 42 participants were available.
- The held-out test contains seven participants.
- Performance differs across phases and collection intervals.
- The system abstains on more than half of held-out participant-days.
- A returned two-phase set does not always contain the true phase.
- External evaluation on independent devices and populations is needed.
- The prototype is not clinically validated.

## Responsible use

PhaseWise must not be used for diagnosis, ovulation confirmation,
pregnancy planning, contraception decisions, fertility treatment,
medication decisions or treatment decisions.
