# MARA Submission Summary

**Project title:** MARA: Missing-Modality-Aware Routing and Abstention for Menstrual Phase Estimation  
**Challenge:** Hack-Nation Challenge 5  
**Submission version:** 1.0  
**Submission date:** 19 July 2026  
**Author:** FAA 
**Repository:** https://github.com/nixnaxnuxx/MARA
**Live application:** [Add deployment URL if available]

---

## 1. Executive Summary

MARA is a sensor-aware selective prediction system for wearable menstrual-phase estimation. It uses resting heart rate and wrist temperature, routes each observation according to sensor availability, and returns either a two-phase prediction set or a no-call when evidence is insufficient or no predictive sensor is available.

The project focuses on three practical research problems:

1. wearable modalities may be missing;
2. a model should not be forced to predict when evidence is weak;
3. uncertainty and routing decisions should be visible to users and reviewers.

MARA is presented as a research demonstration, not a medical device.

## 2. Problem

Wearable health datasets commonly contain incomplete sensor measurements. A conventional model may fail, silently discard observations, or produce overconfident predictions when one or more modalities are missing.

For menstrual-phase estimation, forced predictions are particularly problematic because wearable signals are indirect, phase boundaries are uncertain, sensor availability changes over time, and dataset labels are not equivalent to clinical confirmation.

MARA addresses this by combining missing-modality-aware routing with an explicit abstention policy.

## 3. Proposed Solution

MARA uses two modality-specific predictive inputs:

- resting heart rate;
- wrist temperature.

Its routing policy is:

| Available sensors | Route |
|---|---|
| Both sensors | Equal fusion |
| Resting heart rate only | Resting-heart-rate expert |
| Wrist temperature only | Temperature expert |
| Neither sensor | No-call |

After routing and calibration, MARA sums the two highest phase probabilities.

- When top-two evidence is at least 0.62, MARA returns a two-phase set.
- When evidence is below 0.62, MARA returns a no-call.
- MARA does not return singleton phase predictions.

The frozen probability temperature is 0.9012.

## 4. Target Labels

MARA uses four source-dataset phase labels:

- Menstrual
- Follicular
- Fertility
- Luteal

The label **“Fertility”** is a dataset phase label. It does not confirm fertility, ovulation, or a fertile window.

## 5. Innovation

The submission combines:

- explicit routing for changing sensor availability;
- modality-specific experts;
- equal fusion when both sensors are available;
- calibrated top-two evidence;
- abstention rather than forced prediction;
- a two-phase output policy rather than a singleton;
- public risk-coverage visualization;
- robustness analysis under missing modalities;
- a synthetic longitudinal demonstration;
- an explainable judge-facing application;
- a constrained AI guide that explains only public project information.

The main contribution is the machine-learning policy for routing, uncertainty, and abstention rather than a modification of wearable hardware.

## 6. Dataset and Split

The research pipeline uses the restricted mcPHASES dataset hosted on PhysioNet.

- Total participants: 42
- Training participants: 28
- Validation participants: 7
- Held-out test participants: 7

The final test set was frozen and was not used for threshold tuning. The restricted dataset is not included in this submission.

## 7. Frozen Held-Out Results

The held-out test set contains 959 days from 7 participants.

| Metric | Result |
|---|---:|
| Sensor coverage | 0.9729 |
| Top-1 accuracy | 0.3441 |
| Balanced accuracy | 0.3476 |
| Macro F1 | 0.3417 |
| Unrestricted top-2 coverage | 0.6292 |
| Final decision coverage | 0.4484 |
| Pair-set coverage | 0.6651 |
| No-call rate | 0.5516 |

Additional counts:

- Missing-sensor observations: 26
- Insufficient-evidence no-calls: 503

These results are research findings from a small restricted cohort and are not evidence of clinical effectiveness.

## 8. Application Features

### Product Overview

Introduces the problem, MARA’s routing and abstention policy, the frozen benchmark, and research-use limitations.

### Judge Mode

Provides a guided scenario-based walkthrough for reviewers.

### Held-out Benchmark

Displays frozen held-out performance and supporting aggregate visualizations.

### Sensor-Routing Lab

Allows users to explore hypothetical sensor-availability and probability scenarios and inspect the active route, calibrated phase probabilities, top-two evidence, frozen threshold, and phase-pair or no-call output.

### Synthetic 28-Day Journey

Shows a deterministic synthetic sequence with deliberately missing modalities. It demonstrates how routes and outputs change over time without exposing participant data.

### Validation Risk-Coverage Explorer

Uses public aggregate validation assets to visualize the trade-off between decision coverage and pair-set coverage.

### Robustness Audit

Displays route-level, phase-level, and missing-modality robustness summaries.

### Downloadable Scenario Report

Allows users to download the current hypothetical scenario report and chart values. The report contains no participant or personal health data.

### MARA AI Guide

Answers questions about MARA and its public results. It is constrained against diagnosis, fertility assessment, pregnancy guidance, contraception advice, and clinical recommendations.

## 9. Validation Risk-Coverage Snapshot

At the frozen threshold of 0.62:

- Validation rows: 949
- Sensor-available rows: 925
- Missing-sensor rows: 24
- Decided observations: 485
- Decision coverage: 0.511064
- Pair-set coverage: 0.738144
- No-call rate: 0.488936
- Insufficient-evidence no-calls: 440

Only aggregate validation information is included in the public application.

## 10. Privacy and Safety

The submission excludes raw mcPHASES data, participant identifiers, participant-level wearable records, row-level validation predictions, row-level test predictions, personal health information, and API keys.

The project is not intended for diagnosis, treatment, contraception, pregnancy planning, fertility assessment, confirmation of ovulation, or clinical decision-making.

## 11. Repository Structure

```text
mosaic-phase/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-research.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── app_assets/
│   └── public aggregate assets
├── docs/
│   ├── MODEL_CARD.md
│   ├── SUBMISSION.md
│   └── screenshots/
├── scripts/
│   └── research and reproducibility scripts
└── src/
    ├── app.py
    ├── ai_mara_chat.py
    ├── interactive_components.py
    ├── judge_mode.py
    ├── risk_coverage_explorer.py
    ├── robustness_dashboard.py
    ├── submission_audit.py
    ├── synthetic_journey.py
    └── ui_theme.py
```

The exact contents of `scripts/` may vary depending on which reproducibility scripts are approved for release.

## 12. Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Recommended runtime versions:

```text
streamlit==1.59.2
pandas==2.3.3
numpy==2.2.6
altair==5.5.0
openai==2.46.0
```

Check the environment:

```bash
python -m pip check
```

## 13. Running the Application

From the repository root:

```bash
streamlit run src/app.py
```

The application contains five main sections:

1. Product Overview
2. Held-out Benchmark
3. Sensor-Routing Lab
4. Robustness Audit
5. MARA AI Guide

## 14. OpenAI Configuration

The MARA AI Guide requires an OpenAI API key.

Create a local file:

```text
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "replace_with_your_api_key"
```

Do not commit the real secrets file. A safe placeholder may be included as `.streamlit/secrets.toml.example`.

The predictive benchmark and Sensor-Routing Lab do not depend on the AI guide.

## 15. Submission Contents

The final submission should contain:

- source code for the public application;
- public aggregate application assets;
- synthetic demonstration logic;
- model card;
- submission summary;
- README;
- runtime requirements;
- optional research requirements;
- safe Streamlit configuration;
- selected screenshots;
- submission audit utility;
- approved reproducibility scripts.

The final submission must not contain restricted data or secrets.

## 16. Known Limitations

- The dataset contains only 42 participants.
- The study is retrospective.
- The model uses only resting heart rate and wrist temperature.
- Wearable measurements may be affected by device, adherence, illness, stress, medication, sleep, and environment.
- The source labels are not direct clinical measurements.
- The held-out results do not support clinical deployment.
- The model has not been prospectively validated.
- Population generalizability has not been established.
- Real-world missingness may differ from study conditions.
- The AI guide is explanatory and is not part of the predictive model.

## 17. Reproducibility Notes

The public repository is designed to reproduce the application using aggregate and synthetic assets.

Full model retraining requires authorized access to the restricted source dataset, compliance with its terms, research dependencies, and approved preprocessing, training, calibration, and evaluation scripts.

The final held-out test results must remain frozen.

## 18. Final Submission Checklist

- [ ] Repository URL added
- [ ] Deployment URL added when available
- [ ] README reviewed
- [ ] MODEL_CARD.md reviewed
- [ ] SUBMISSION.md reviewed
- [ ] All Python files compile
- [ ] Submission audit reports zero failures
- [ ] All five app sections load
- [ ] Downloadable scenario report opens correctly
- [ ] Chart-values CSV opens correctly
- [ ] No API key is tracked
- [ ] No restricted dataset file is tracked
- [ ] No participant-level prediction file is tracked
- [ ] Final ZIP is created from committed Git files
- [ ] Final ZIP is tested in a clean folder

