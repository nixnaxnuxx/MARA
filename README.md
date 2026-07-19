# MARA

## Missing-Modality-Aware Routing and Abstention for Menstrual Phase Estimation

MARA is a research prototype for wearable menstrual-phase estimation under changing sensor availability. It uses resting heart rate and wrist temperature, routes each observation to the available modality expert or an equal-fusion pathway, calibrates the resulting probabilities, and returns either a two-phase prediction set or a no-call.

> **Research use only.** MARA is not a medical device and must not be used for diagnosis, treatment, contraception, pregnancy planning, fertility assessment, or confirmation of ovulation.

---

## Links

- **Live demo:** https://mara-hacknation-hackathon.streamlit.app/
- **Source repository:** https://github.com/nixnaxnuxx/
- **Model card:** docs/MODEL_CARD.md
- **Submission summary:** docs/SUBMISSION.md

---

## Challenge

MARA was developed for **Hack-Nation Challenge 5: Women's Hormonal Health**.

The project addresses a focused research question:

> How robust is wearable menstrual-phase estimation when predictive sensor modalities are missing, and can calibrated abstention prevent unreliable outputs for unseen participants?

The contribution is the machine-learning routing and selective-output policy rather than a modification of wearable hardware.

---

## Dataset

### Source

The research pipeline uses the restricted **mcPHASES** dataset hosted on PhysioNet.

mcPHASES contains longitudinal information that can include Fitbit-derived wearable signals, continuous glucose monitoring, hormone measurements, menstrual-cycle information, sleep, and symptoms.

The raw dataset is **not included** in this repository. Access requires authorization through PhysioNet and compliance with the dataset's terms.

### Cohort and participant split

The working cohort contains 42 participants:

| Partition | Participants |
|---|---:|
| Training | 28 |
| Validation | 7 |
| Held-out test | 7 |
| **Total** | **42** |

The split is performed by participant rather than by daily row. A participant appears in only one partition.

### Predictive inputs used by MARA

MARA uses two wearable-derived predictive modalities:

1. **Resting heart rate**
2. **Wrist temperature**

Hormone measurements and direct menstrual-flow indicators are excluded from the predictive inputs. They are not used to generate the public model outputs.

Sleep and activity were evaluated during development but were not selected as predictive routes in the frozen MARA policy.

### Target labels

The four source-dataset phase labels are:

- Menstrual
- Follicular
- Fertility
- Luteal

The label **Fertility** is a dataset phase label. It does not confirm fertility, ovulation, or a fertile window.

---

## Data Processing and Feature Engineering

The research pipeline converts the longitudinal records into participant-day observations.

Key processing rules include:

- daily aggregation of wearable measurements;
- participant-level train, validation, and test separation;
- current-day wearable features;
- participant-relative baselines;
- past-only rolling summaries and trends;
- explicit sensor-availability indicators;
- exclusion of future information from temporal features;
- exclusion of hormone and direct-flow variables from predictive inputs.

Past-only features are constructed without using future days or the current observation inside its historical reference window. This reduces temporal leakage.

The main processing scripts include:

```text
scripts/audit_daily_aggregation.py
scripts/audit_labels_and_overlap.py
scripts/build_master_table.py
scripts/build_temporal_features.py
scripts/create_participant_split.py
```

---

## Model Used for Training

MARA is a **sensor-aware ensemble of two separately trained probabilistic modality experts**:

- an **RHR expert** trained on resting-heart-rate and past-only temporal features;
- a **temperature expert** trained on wrist-temperature and participant-relative temporal features.

The exact base estimator, preprocessing pipeline, selected feature columns, and training parameters are implemented in the research scripts, particularly:

```text
scripts/train_baselines.py
scripts/compare_robust_feature_sets.py
scripts/train_modality_experts.py
scripts/select_robust_policy.py
```

This README describes the verified final architecture and policy without inferring an estimator name from filenames. Review `train_modality_experts.py` before release if the exact estimator class and hyperparameters need to be copied into this section.

### Routing policy

| Sensor availability | Active model route |
|---|---|
| RHR and temperature available | Equal fusion of both expert probability vectors |
| RHR only | RHR expert |
| Temperature only | Temperature expert |
| Neither available | Immediate no-call |

When both experts are available, their probability vectors are combined with equal weights:

```text
fused probability = 0.5 × RHR probability + 0.5 × temperature probability
```

### Why separate experts?

A single model that expects every feature may fail or require imputation when a sensor is missing. MARA instead trains modality-specific experts and explicitly selects the appropriate route from the sensor state.

This makes missingness visible and allows the system to degrade gracefully.

---

## Calibration and Selective Prediction

Raw routed probabilities are calibrated with temperature scaling.

### Frozen calibration

```text
Probability temperature: 0.9012
```

The temperature was learned using grouped out-of-fold training predictions.

### Frozen selective-output threshold

```text
Top-two evidence threshold: 0.62
```

For each participant-day:

1. MARA ranks the four calibrated phase probabilities.
2. It adds the probabilities of the two highest-ranked phases.
3. It returns those two phases only when their combined evidence is at least 0.62.
4. It returns no-call when the threshold is not reached.
5. It also returns no-call when neither predictive sensor is available.

MARA does not issue an official singleton phase output.

---

## Training and Model-Selection Workflow

The training workflow is:

1. Inspect and audit the restricted dataset.
2. Aggregate records to participant-day observations.
3. construct the four phase labels;
4. create a participant-level 28/7/7 split;
5. build current-day and past-only temporal features;
6. compare baseline and robust feature configurations;
7. train separate RHR and temperature experts;
8. evaluate sensor-aware probability fusion;
9. learn probability-temperature scaling from grouped out-of-fold training predictions;
10. compare selective-output policies on validation data;
11. freeze the complete routing, calibration, and threshold policy;
12. evaluate the held-out test partition once.

The held-out test partition was not used to tune the model, calibration temperature, or selective threshold.

---

## Evaluation

### Held-out test set

- **Unseen participants:** 7
- **Participant-days:** 959
- **Sensor coverage:** 0.9729
- **Missing-sensor observations:** 26
- **Insufficient-evidence no-calls:** 503

### Frozen results

| Metric | Result |
|---|---:|
| Top-1 accuracy | 0.3441 |
| Balanced accuracy | 0.3476 |
| Macro F1 | 0.3417 |
| Unrestricted top-2 coverage | 0.6292 |
| Final decision coverage | 0.4484 |
| Pair-set coverage | 0.6651 |
| No-call rate | 0.5516 |

### Metric interpretation

- **Decision coverage** is the proportion of all held-out days on which MARA returns a two-phase set.
- **Pair-set coverage** is the proportion of decided cases for which the reference phase is inside the returned two-phase set.
- **No-call rate** is the proportion of observations where MARA abstains because evidence is insufficient or no predictive route is available.
- **Unrestricted top-2 coverage** is calculated before applying the final selective threshold.

These results are research findings from a small retrospective cohort and are not evidence of clinical effectiveness.

---

## Validation Risk-Coverage Snapshot

At the frozen threshold of 0.62:

| Quantity | Value |
|---|---:|
| Validation rows | 949 |
| Sensor-available rows | 925 |
| Missing-sensor rows | 24 |
| Decided observations | 485 |
| Decision coverage | 0.511064 |
| Pair-set coverage | 0.738144 |
| No-call rate | 0.488936 |
| Insufficient-evidence no-calls | 440 |

The public repository includes aggregate validation risk-coverage assets only. It does not include row-level validation predictions.

---

## Application Features

The Streamlit application contains five main sections.

### Product Overview

Introduces MARA, the four phase labels, the routing policy, the frozen result, and the research limitations.

### Held-out Benchmark

Displays the frozen held-out metrics, route-level results, phase-level summaries, and interval-level summaries.

### Sensor-Routing Lab

Allows users to explore hypothetical expert probability vectors and four sensor-availability states:

- both sensors;
- RHR only;
- temperature only;
- neither sensor.

The lab displays the active route, calibrated probabilities, top-two evidence, frozen threshold, and final phase-pair or no-call output.

### Synthetic 28-Day Journey

Provides a deterministic synthetic demonstration with deliberately missing modalities. It does not use participant data and is not intended to represent a clinically typical cycle.

### Robustness Audit

Displays route, interval, phase, and missing-modality robustness summaries.

### MARA AI Guide

Provides explanations of MARA and its public aggregate results when an OpenAI API key is configured.

The AI guide is separate from the predictive model. It does not train the modality experts, generate benchmark predictions, or alter the routing policy.

---

## Public Application Assets

The deployed app loads approved public assets from `app_assets/`:

```text
app_assets/
├── approved_claims.md
├── final_benchmark.json
├── interval_benchmark.csv
├── phase_benchmark.csv
├── route_benchmark.csv
├── validation_risk_coverage.csv
└── validation_risk_coverage_metadata.json
```

These files contain aggregate summaries and approved explanatory text. They do not contain raw participant records or participant identifiers.

---

## Repository Structure

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
│   └── public aggregate application assets
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

---

## Installation

### 1. Create or activate an environment

The project was developed with Python 3.10.

Using Conda:

```bash
conda create -n mosaic python=3.10
conda activate mosaic
```

### 2. Install runtime dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The main runtime dependencies are:

```text
streamlit==1.59.2
pandas==2.3.3
numpy==2.2.6
altair==5.5.0
openai==2.46.0
```

Research and training dependencies are listed separately in:

```text
requirements-research.txt
```

---

## Run the Application

From the repository root:

```bash
streamlit run src/app.py
```

The local app will normally open at:

```text
http://localhost:8501
```

---

## OpenAI Configuration

The predictive model and benchmark do not require OpenAI.

The optional MARA AI Guide requires an API key.

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "replace_with_your_api_key"
```

Do not commit the real secrets file.

The repository may include a safe placeholder:

```text
.streamlit/secrets.toml.example
```

---

## Reproducibility

The public repository supports reproduction of the application, public aggregate analyses, and synthetic demonstrations.

Full end-to-end retraining requires:

- authorized access to mcPHASES;
- compliance with PhysioNet and dataset terms;
- the research dependencies;
- the preprocessing and training scripts;
- the frozen participant split;
- the documented calibration and evaluation procedure.

The raw restricted dataset, participant-level processed tables, and row-level prediction files are not distributed.

---

## Privacy and Data Governance

The repository must not include:

```text
.streamlit/secrets.toml
.env
raw mcPHASES files
participant identifiers
participant-level wearable measurements
row-level validation predictions
row-level held-out test predictions
personal health information
```

The public app uses only aggregate and synthetic assets.

---

## Limitations

- The dataset contains only 42 participants.
- The final held-out benchmark contains 7 participants.
- The study is retrospective.
- Only resting heart rate and wrist temperature drive the frozen predictive routes.
- Dataset phase labels are not direct clinical measurements.
- Performance varies across phases, study intervals, and sensor routes.
- More than half of held-out participant-days receive a no-call.
- A returned phase pair can still exclude the reference phase.
- Device and population generalization require independent evaluation.
- Real-world missingness may differ from the study and synthetic conditions.
- No prospective clinical validation has been completed.
- The current performance does not support clinical or consumer-health deployment.

---

## Safety

MARA must not be used for:

- diagnosis;
- treatment;
- contraception;
- pregnancy planning;
- fertility assessment;
- confirmation of ovulation;
- prediction of a fertile window;
- clinical decision-making.

Future deployment would require additional external validation, fairness evaluation, privacy and consent controls, security testing, human-factors assessment, monitoring, and regulatory review.

---

## Documentation

- docs/MODEL_CARD.md — technical, ethical, safety, and evaluation documentation
- docs/SUBMISSION.md — concise judge-facing submission summary
- docs/screenshots/ — selected application screenshots

---

## Author

**Author:** FAA
**Challenge:** Hack-Nation Challenge 5  
**Repository:** https://github.com/nixnaxnuxx/
**Live demo:** https://mara-hacknation-hackathon.streamlit.app/

---

## License


The repository license does not override the access, use, or redistribution terms of the restricted mcPHASES dataset.
