---
title: Parkinson's Voice Analysis
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# Parkinson's Voice Analysis Demo

Live demo of an acoustic-feature-based Parkinson's disease classifier.

Upload or record a sustained `/a/` vowel, and the system extracts 16 voice
quality features using Praat (jitter, shimmer, HNR, F0 statistics), then runs
a Random Forest classifier trained on the UCI Parkinson's voice dataset with
subject-grouped 5-fold cross-validation.

## Performance

| Metric                  | Random Forest    |
|-------------------------|------------------|
| Accuracy (subject-grouped CV) | 75.3% ± 8.0%   |
| ROC-AUC                 | 78.9% ± 11.9%    |
| PR-AUC                  | 90.6% ± 8.9%     |

## Honest disclaimers

- **Not a medical device.** Cannot diagnose Parkinson's disease.
- Trained on 31 subjects (UCI dataset) — limited demographic generalization.
- Educational portfolio project, not clinically validated.

## Source code

Full methodology, training pipeline, and documentation:
[github.com/vpatel2398/Parkinson-Disease-Detection-with-Audio](https://github.com/vpatel2398/Parkinson-Disease-Detection-with-Audio)