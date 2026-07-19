# MARA Model Card

**Model name:** MARA  
**Full title:** Missing-Modality-Aware Routing and Abstention for Menstrual Phase Estimation  
**Version:** 1.0  
**Release date:** 19 July 2026  
**Status:** Research demonstration  
**Author:** FAA

---

## 1. Model Overview

MARA is a missing-modality-aware selective prediction system for wearable-based menstrual-phase estimation. It uses resting heart rate and wrist temperature as predictive inputs, routes each observation according to sensor availability, and returns either a two-phase prediction set or a no-call when evidence is insufficient or no predictive sensor is available.

MARA does not return a singleton phase prediction. Its abstention policy is intended to make uncertainty visible rather than force a prediction for every observation.

The public application is a research demonstration. It is not a medical device and is not intended for clinical or personal health decision-making.

## 2. Intended Use

MARA is intended for:

- research on wearable-based menstrual-phase estimation;
- demonstration of missing-modality-aware routing;
- study of selective prediction and abstention;
- analysis of risk-coverage trade-offs;
- educational exploration of sensor availability, evidence strength, and uncertainty;
- hackathon evaluation using public aggregate and synthetic demonstration assets.

## 3. Out-of-Scope Use

MARA must not be used for:

- medical diagnosis or treatment decisions;
- contraception or pregnancy planning;
- fertility assessment;
- confirmation of ovulation or a fertile window;
- emergency or clinical decision-making;
- individual health monitoring without appropriate governance and validation.

The dataset label **“Fertility”** is a source-dataset phase label. It does not confirm fertility, ovulation, or a fertile window.

## 4. Dataset

The research pipeline was developed using the restricted **mcPHASES** dataset hosted on PhysioNet.

### Participant split

- Total participants: 42
- Training participants: 28
- Validation participants: 7
- Held-out test participants: 7

### Public release policy

The restricted dataset is not included in the public repository or submission package. The public application contains approved aggregate benchmark summaries, aggregate validation risk-coverage assets, synthetic demonstration data, and non-identifying application metadata only.

Participant-level records, raw wearable measurements, row-level predictions, and participant identifiers are excluded.

## 5. Predictive Inputs

MARA uses two wearable-derived inputs:

1. Resting heart rate
2. Wrist temperature

The model does not use hormone measurements, direct menstrual-flow features, confirmed ovulation markers, pregnancy tests, or fertility tests.

## 6. Target Labels

The model predicts among four source-dataset phase labels:

- Menstrual
- Follicular
- Fertility
- Luteal

These labels should not be interpreted as direct clinical measurements.

## 7. Model Architecture and Routing

MARA contains modality-specific predictive experts and a routing policy based on sensor availability.

| Available inputs | Active route |
|---|---|
| Resting heart rate and wrist temperature | Equal-probability fusion |
| Resting heart rate only | Resting-heart-rate expert |
| Wrist temperature only | Temperature expert |
| Neither sensor | No-call |

When both modalities are available, MARA uses equal fusion. When only one modality is available, it routes to the corresponding expert. When neither modality is available, it does not produce a phase-set prediction.

## 8. Calibration and Selective Policy

The frozen policy is:

- Probability temperature: **0.9012**
- Top-two evidence threshold: **0.62**
- Allowed outputs: **two-phase set** or **no-call**
- Singleton phase output: **not used**

For each observation, the two highest calibrated phase probabilities are summed. MARA returns the associated two-phase set only when the top-two evidence meets or exceeds the frozen threshold. Otherwise, it returns a no-call.

The threshold and calibration temperature were selected before final held-out test evaluation. The held-out test set was not used for tuning.

## 9. Evaluation Protocol

Evaluation used participant-level splits to reduce leakage across train, validation, and test partitions. The final test set was frozen before reporting held-out results.

### Held-out test set

- Participants: 7
- Days: 959
- Sensor coverage: 0.9729
- Missing-sensor observations: 26
- Insufficient-evidence no-calls: 503

## 10. Frozen Held-Out Results

| Metric | Result |
|---|---:|
| Top-1 accuracy | 0.3441 |
| Balanced accuracy | 0.3476 |
| Macro F1 | 0.3417 |
| Unrestricted top-2 coverage | 0.6292 |
| Final decision coverage | 0.4484 |
| Pair-set coverage | 0.6651 |
| No-call rate | 0.5516 |

Final decision coverage is the proportion of held-out observations for which MARA returned a two-phase set after applying its missing-modality and evidence policies. Pair-set coverage measures how often the returned two-phase set contained the reference phase among observations where a set was returned.

These results are research findings from a small restricted cohort and are not evidence of clinical effectiveness.

## 11. Validation Risk-Coverage Summary

At the frozen threshold of 0.62:

- Validation rows: 949
- Sensor-available rows: 925
- Missing-sensor rows: 24
- Decided observations: 485
- Decision coverage: 0.511064
- Pair-set coverage: 0.738144
- No-call rate: 0.488936
- Insufficient-evidence no-calls: 440

These values are aggregate validation summaries and do not expose participant-level predictions.

## 12. Missing-Modality Behaviour

MARA explicitly represents four sensor states: both sensors available, resting heart rate only, wrist temperature only, and neither sensor available.

The public Sensor-Routing Lab and synthetic 28-day journey allow users to inspect how route selection, calibrated probabilities, evidence, and selective output change across these states. The synthetic journey is deterministic and contains no participant data.

## 13. Public Application Features

The public Streamlit application includes:

- Product Overview
- Guided Judge Mode
- Frozen held-out benchmark
- Sensor-Routing Lab
- Synthetic 28-day MARA journey
- Validation risk-coverage explorer
- Robustness and missing-modality audit
- Downloadable current-scenario report
- Downloadable chart-value table
- Constrained MARA AI Guide

The downloadable report contains hypothetical demonstration values only.

## 14. MARA AI Guide

The MARA AI Guide is a separate explanatory interface powered by an OpenAI model when an API key is configured.

The guide is constrained to explain MARA and summarize public aggregate results. It should refuse diagnosis, fertility assessment, pregnancy advice, contraception guidance, and clinical recommendations.

The guide should not receive personal health information, participant identifiers, restricted dataset content, row-level predictions, or raw wearable data.

The AI component is not part of the predictive routing model and is not used to generate benchmark results.

## 15. Limitations

1. The dataset contains only 42 participants.
2. The study is retrospective.
3. The model uses resting heart rate and wrist temperature only.
4. Source labels are not direct clinical measurements.
5. The model has not been clinically validated.
6. Performance may vary across demographic, physiological, device, behavioural, and environmental groups not adequately represented in the dataset.
7. Wearable measurements may be affected by device differences, adherence, illness, stress, medication, sleep, and environment.
8. Real-world missingness may differ from study and synthetic conditions.
9. Reported held-out performance does not support clinical or consumer-health deployment.
10. No prospective or longitudinal user-facing validation has been completed.

## 16. Safety and Ethical Considerations

MARA is designed to surface uncertainty through abstention rather than force a prediction. Safety measures in the public application include:

- persistent research-use disclaimer;
- no singleton prediction output;
- explicit no-call state;
- visible evidence threshold;
- visible sensor-routing logic;
- no participant-level data;
- no personal-health data importer;
- constrained AI-guide behaviour;
- clear distinction between dataset labels and clinical concepts.

Any future deployment would require additional clinical validation, fairness analysis, privacy, consent, security, monitoring, human-factors work, and regulatory review.

## 17. Privacy and Data Governance

The public repository must not include:

- raw mcPHASES data;
- participant identifiers;
- participant-level wearable measurements;
- row-level validation predictions;
- row-level held-out test predictions;
- personal health information;
- API secrets.

Public application assets should contain only approved aggregate outputs and synthetic data.

## 18. Reproducibility

The repository separates runtime application code, public aggregate assets, research scripts, and restricted local data.

Launch the public app with:

```bash
streamlit run src/app.py
```

Primary runtime dependencies are listed in `requirements.txt`. Research dependencies may be listed separately in `requirements-research.txt`.

Because the source dataset is restricted, full end-to-end reproduction requires authorized dataset access and compliance with the dataset terms.

## 19. Version and Change Control

**Version:** 1.0-hackathon  
**Release date:** 19 July 2026  
**Evaluation status:** Frozen held-out test results  
**Calibration temperature:** 0.9012  
**Top-two threshold:** 0.62

Future versions should document changes to preprocessing, participant splits, model features, expert models, fusion, calibration, threshold, output policy, metrics, and application behaviour.

