# Parkinson's Disease Detection from Voice

Machine learning pipeline for detecting Parkinson's disease (PD) from sustained vowel phonation, using acoustic features that capture the vocal symptoms characteristic of PD: reduced pitch range, increased jitter and shimmer, and decreased harmonics-to-noise ratio.

This project reproduces and extends the classic [Little et al. (2007)](https://www.nature.com/articles/nature06292) and [Tsanas et al. (2010)](https://ieeexplore.ieee.org/document/5466111) work on voice-based PD detection, with an end-to-end pipeline from raw audio to web-based prediction.

> **Status:** Educational portfolio project. Not a medical device. Predictions are not clinical diagnoses.

---

## Demo

A live demo is deployed at: *[link to Hugging Face Space — to be added]*

Upload a 5+ second recording of a sustained `/a/` vowel. The system extracts 18 acoustic features using Praat (via `parselmouth`) and returns a calibrated probability score.

![Demo Screenshot](docs/demo.png)

---

## Why this project

About **90% of people with Parkinson's develop voice impairments**, often years before motor symptoms become obvious. Acoustic analysis of sustained phonation is non-invasive, low-cost, and can be done remotely — making it a promising tool for early screening and longitudinal monitoring.

Companies like [Winterlight Labs](https://winterlightlabs.com/) (now part of Cambridge Cognition) have built clinical-grade products around this principle. This project is my hands-on exploration of the same problem space.

---

## What the pipeline does

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw audio   │ ──► │ Feature         │ ──► │  Classifier  │ ──► │ Calibrated   │
│  (.wav)      │     │ extraction      │     │  (XGBoost)   │     │ probability  │
│              │     │ (Praat features)│     │              │     │ + SHAP plot  │
└──────────────┘     └─────────────────┘     └──────────────┘     └──────────────┘
```

1. **Audio ingestion** — accepts `.wav` (16 kHz mono recommended)
2. **Feature extraction** — uses [Praat](https://www.fon.hum.uva.nl/praat/) via `parselmouth` to compute scientifically valid acoustic measures (F0 statistics, jitter, shimmer, HNR)
3. **Classification** — compares Logistic Regression, SVM, Random Forest, XGBoost, and a small MLP on a 5-fold stratified cross-validation
4. **Explainability** — SHAP values explain why a specific prediction was made

---

## Dataset

**Primary dataset:** UCI ML Repository [Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) — 195 sustained-vowel recordings from 31 subjects (23 with PD, 8 healthy), originally curated by Max Little at the University of Oxford.

**Important caveats — read these:**

- The dataset is **small (195 samples)** and **subject-imbalanced**. Naïve cross-validation that splits at the sample level leaks subject identity and inflates accuracy.
- This project uses **subject-grouped cross-validation** (`GroupKFold`) so the same speaker never appears in both train and test folds. This is the correct evaluation protocol for this dataset and is **omitted in most blog-post tutorials** — which is why their accuracy numbers (often 95%+) are misleading.
- The original UCI features include nonlinear-dynamics measures (RPDE, DFA, PPE, D2) computed from proprietary research code (Little et al.). These cannot be reliably reproduced from raw audio with standard libraries. **This project uses an 18-feature subset that can be honestly recomputed from any uploaded audio.**

---

## Results

All metrics reported with **subject-grouped 5-fold cross-validation** (mean ± std across folds):

| Model               | Accuracy        | F1 (PD class)   | ROC-AUC          | PR-AUC          |
|---------------------|-----------------|-----------------|------------------|-----------------|
| Logistic Regression | 0.746 ± 0.081   | 0.828 ± 0.070   | 0.713 ± 0.149    | 0.882 ± 0.110   |
| SVM (RBF kernel)    | **0.771** ± 0.076 | **0.861** ± 0.045 | 0.613 ± 0.182  | 0.823 ± 0.100   |
| Random Forest       | 0.753 ± 0.080   | 0.844 ± 0.055   | **0.789** ± 0.119 | **0.906** ± 0.089 |
| XGBoost             | 0.748 ± 0.090   | 0.837 ± 0.061   | 0.747 ± 0.129    | 0.884 ± 0.082   |

**Selected model: Random Forest** — best ROC-AUC and PR-AUC. The SVM achieves the highest raw accuracy but its low ROC-AUC (0.613) indicates poor probability calibration, which makes it unsuitable for a screening tool where ranking matters more than thresholded decisions.

### How to read these numbers

The dataset is imbalanced (75% PD recordings), which inflates accuracy and F1. ROC-AUC and PR-AUC are the more meaningful metrics here:

- **PR-AUC of 0.906** for Random Forest means the model ranks PD recordings above healthy recordings with high reliability across the operating curve.
- **High standard deviations (0.08–0.18)** reflect the small subject pool (31 people). When one heavy-data subject lands in the test fold, performance swings noticeably. This is honest noise, not a bug.
- **Why ~77% accuracy and not 95%+** like blog tutorials? Most online tutorials use random splits that leak speaker identity between train and test. With proper subject-grouped CV, ~70–85% is consistent with peer-reviewed results on this dataset (e.g. Tsanas et al. 2010).

Confusion matrices, ROC curves, and PR curves are in `results/`.

---

## Tech stack

- **Audio analysis:** `praat-parselmouth` (Praat bindings), `librosa`
- **Machine learning:** `scikit-learn`, `xgboost`
- **Explainability:** `shap` (used in `notebooks/02_shap_analysis.ipynb`)
- **Web app:** `flask`
- **Deployment:** Hugging Face Spaces

---

## Getting started

### Prerequisites

- Python 3.10+
- `ffmpeg` (for audio decoding)

### Installation

```bash
git clone https://github.com/vpatel2398/Parkinson-Disease-Detection-with-Audio.git
cd Parkinson-Disease-Detection-with-Audio
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train all models

```bash
python -m src.train
```

This runs the full pipeline: data loading → subject-grouped 5-fold CV across all models → saves metrics, plots, and the best model to `results/`.

### Predict on an audio file

```bash
python -m src.predict path/to/recording.wav
```

### Run the web app locally

```bash
python -m src.app
# Open http://localhost:5000
```

---

## Project structure

```
.
├── src/
│   ├── features.py        # Acoustic feature extraction via Praat
│   ├── train.py           # Training + cross-validation across models
│   ├── predict.py         # Inference on a single audio file
│   └── app.py             # Flask web app
├── data/
│   └── parkinsons.csv     # UCI dataset (committed for reproducibility)
├── results/               # Generated: metrics, plots, saved models
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory data analysis
├── tests/                 # Unit tests for feature extraction
├── requirements.txt
└── README.md
```

---

## What I learned / What I'd do differently

This project went through a complete rewrite after I noticed several problems with my first version. Documenting them here because the lessons are more interesting than the final accuracy number:

1. **Don't fake features at inference time.** My initial code computed 16 features from raw audio and filled in 6 more (RPDE, DFA, etc.) with `np.random.rand()` as placeholders, because I couldn't easily reproduce the original research code. The model was trained on real values and saw noise at inference — predictions on uploaded audio were essentially meaningless. The fix: drop features I can't honestly compute, and clearly document what the remaining set captures.

2. **Audio-domain libraries matter.** I originally computed jitter as `np.mean(np.abs(np.diff(signal)))` on raw audio samples. That's not jitter — real jitter measures cycle-to-cycle period variation, which requires glottal-pulse detection. Switching to Praat (via `parselmouth`) gave me scientifically validated measures in two lines of code.

3. **Subject-level data leakage is the silent accuracy killer.** With 31 subjects and 195 samples, a random 80/20 split puts ~6 recordings per subject in the test set — and the model essentially memorizes voices, not disease patterns. Switching to `GroupKFold` on subject ID dropped accuracy substantially but produced numbers that actually generalize.

4. **Bigger models aren't better.** My first version used a 6-layer dense network (256→128→64→32→16→2) on 22 tabular features. Logistic regression with proper regularization performs comparably and is far more interpretable. For small tabular datasets, classical ML is usually the right answer.

5. **Scale after splitting, not before.** Fitting `StandardScaler` on the full dataset before splitting leaks test-set statistics into training. Small effect, but it adds up with everything else.

---

## Limitations and honest disclosures

- **Not a medical device.** This model has not been clinically validated. It cannot diagnose Parkinson's disease.
- **Small dataset.** 31 subjects is not enough for population-level claims. The model captures patterns specific to this cohort and may not generalize across demographics, recording equipment, or languages.
- **Recording conditions matter.** The UCI dataset was recorded in a clinical setting. Predictions on phone-recorded audio will be noisier.
- **Sustained vowel only.** This pipeline analyzes `/a/` vowel phonation. It does not handle continuous or spontaneous speech.

---

## References

- Little, M. A., et al. (2007). *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* Nature Precedings.
- Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2010). *Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests.* IEEE Transactions on Biomedical Engineering.
- Boersma, P., & Weenink, D. (2024). *Praat: doing phonetics by computer.* Version 6.4. http://www.praat.org/

---

## License

MIT License — see [LICENSE](LICENSE) file.

## Contact

Vivek Patel — [vpatel2398@gmail.com](mailto:vpatel2398@gmail.com) — [GitHub](https://github.com/vpatel2398)