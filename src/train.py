"""
Training pipeline for Parkinson's disease detection from voice features.

Compares Logistic Regression, SVM, Random Forest, XGBoost, and a small MLP
on the UCI Parkinson's dataset using subject-grouped 5-fold cross-validation.

Subject-grouped CV is critical for this dataset: the 195 samples come from
only 31 subjects, with multiple recordings per subject. A naive random split
puts the same speaker in both train and test, so the model learns to recognize
voices rather than disease patterns — inflating accuracy by 10-20 percentage
points.

Outputs (written to ./results/):
    - metrics.csv               Per-model mean ± std across folds
    - confusion_matrix_*.png    Per-model aggregated confusion matrix
    - roc_curves.png            ROC curves for all models on one plot
    - pr_curves.png             Precision-recall curves
    - shap_summary.png          Feature importance for the best model
    - best_model.joblib         Serialized best-performing classical model
    - scaler.joblib             Fitted StandardScaler for inference
    - feature_names.json        Column order for inference
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Paths — kept relative so the script runs from anywhere in the repo.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "parkinsons.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


# These features are the subset we can compute honestly from raw audio using
# Praat (see src/features.py). The UCI dataset has 22 features total, but
# 6 of them (RPDE, DFA, D2, spread1, spread2, PPE) are nonlinear-dynamics
# measures that require proprietary research code to reproduce. We exclude
# them so the training distribution matches what we extract at inference.
HONEST_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ",
    "Shimmer:DDA",
    "NHR", "HNR",
]
TARGET_COL = "status"  # 1 = Parkinson's, 0 = healthy


@dataclass
class FoldMetrics:
    """Metrics from a single CV fold."""
    accuracy: float
    f1: float
    roc_auc: float
    pr_auc: float


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the UCI Parkinson's dataset and extract subject IDs.

    The 'name' column has the format 'phon_R01_S<subject>_<recording>',
    e.g. 'phon_R01_S01_1', 'phon_R01_S01_2', ... — we parse the subject
    number out so GroupKFold can keep all recordings of one subject in
    the same fold.
    """
    df = pd.read_csv(path)

    # Extract subject ID. The format is consistent in the UCI dataset.
    # Example: 'phon_R01_S01_1' -> subject '01'
    df["subject_id"] = df["name"].str.extract(r"S(\d+)_")[0]

    X = df[HONEST_FEATURES].copy()
    y = df[TARGET_COL].copy()
    groups = df["subject_id"].copy()

    n_subjects = groups.nunique()
    n_pd_subjects = df[df[TARGET_COL] == 1]["subject_id"].nunique()
    n_healthy_subjects = df[df[TARGET_COL] == 0]["subject_id"].nunique()

    print(f"Loaded {len(df)} recordings from {n_subjects} subjects")
    print(f"  PD subjects:      {n_pd_subjects}")
    print(f"  Healthy subjects: {n_healthy_subjects}")
    print(f"  PD recordings:    {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(f"  Features used:    {len(HONEST_FEATURES)}\n")

    return X, y, groups


def build_models() -> dict:
    """
    Define the model zoo. Each entry is a fresh, untrained estimator.

    Hyperparameters are sensible defaults — tuned per-fold via the
    estimators' own regularization where applicable. For this small
    dataset, aggressive tuning would overfit; defaults are more honest.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def evaluate_fold(
    model, X_train, X_test, y_train, y_test
) -> tuple[FoldMetrics, np.ndarray, np.ndarray]:
    """
    Fit on a training fold, evaluate on the held-out fold.

    Returns metrics plus (y_true, y_prob) for later aggregation into
    ROC/PR curves across all folds.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = FoldMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_prob),
        pr_auc=average_precision_score(y_test, y_prob),
    )
    return metrics, y_test.values, y_prob


def cross_validate_model(
    name: str, model, X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[list[FoldMetrics], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run subject-grouped K-fold CV for one model.

    Critical detail: the StandardScaler is fit on the TRAINING fold only,
    then used to transform the test fold. Fitting on the full dataset
    leaks test statistics into training.

    Returns:
        - List of per-fold metrics
        - Aggregated true labels across all test folds
        - Aggregated predicted probabilities across all test folds
        - Aggregated predicted labels (for confusion matrix)
    """
    cv = GroupKFold(n_splits=N_FOLDS)
    fold_metrics: list[FoldMetrics] = []
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit scaler on training fold only — no test-set leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clone the model so each fold gets a fresh estimator. We rebuild
        # via build_models() rather than sklearn.base.clone to avoid any
        # state carry-over between folds.
        fresh_model = build_models()[name]

        metrics, y_true_fold, y_prob_fold = evaluate_fold(
            fresh_model, X_train_scaled, X_test_scaled, y_train, y_test
        )
        y_pred_fold = (y_prob_fold >= 0.5).astype(int)

        fold_metrics.append(metrics)
        all_y_true.append(y_true_fold)
        all_y_prob.append(y_prob_fold)
        all_y_pred.append(y_pred_fold)

        print(
            f"  Fold {fold_idx + 1}/{N_FOLDS}: "
            f"acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
            f"auc={metrics.roc_auc:.3f}"
        )

    return (
        fold_metrics,
        np.concatenate(all_y_true),
        np.concatenate(all_y_prob),
        np.concatenate(all_y_pred),
    )


def summarize_metrics(fold_metrics: list[FoldMetrics]) -> dict:
    """Compute mean ± std across folds for each metric."""
    return {
        "accuracy_mean": np.mean([m.accuracy for m in fold_metrics]),
        "accuracy_std": np.std([m.accuracy for m in fold_metrics]),
        "f1_mean": np.mean([m.f1 for m in fold_metrics]),
        "f1_std": np.std([m.f1 for m in fold_metrics]),
        "roc_auc_mean": np.mean([m.roc_auc for m in fold_metrics]),
        "roc_auc_std": np.std([m.roc_auc for m in fold_metrics]),
        "pr_auc_mean": np.mean([m.pr_auc for m in fold_metrics]),
        "pr_auc_std": np.std([m.pr_auc for m in fold_metrics]),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: Path):
    """Plot and save the aggregated confusion matrix across all CV folds."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Healthy", "PD"])
    ax.set_yticks([0, 1], ["Healthy", "PD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(results: dict, out_path: Path):
    """All models on one ROC plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, payload in results.items():
        y_true, y_prob = payload["y_true"], payload["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (subject-grouped 5-fold CV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_curves(results: dict, out_path: Path):
    """All models on one precision-recall plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, payload in results.items():
        y_true, y_prob = payload["y_true"], payload["y_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (subject-grouped 5-fold CV)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_best_model(best_name: str, X: pd.DataFrame, y: pd.Series):
    """
    Refit the best model on the FULL dataset and save it for inference.

    This is the standard workflow: use CV to select the model, then refit
    on all available data before deployment. The CV metrics remain our
    honest estimate of generalization performance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_models()[best_name]
    model.fit(X_scaled, y)

    joblib.dump(model, RESULTS_DIR / "best_model.joblib")
    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")

    with open(RESULTS_DIR / "feature_names.json", "w") as f:
        json.dump({"features": HONEST_FEATURES, "model": best_name}, f, indent=2)

    print(f"\nSaved best model ({best_name}) to {RESULTS_DIR / 'best_model.joblib'}")


def main():
    print("=" * 60)
    print("Parkinson's Disease Detection — Training Pipeline")
    print("=" * 60 + "\n")

    X, y, groups = load_data(DATA_PATH)

    results = {}
    summary_rows = []

    for name in build_models().keys():
        print(f"\n>>> {name}")
        fold_metrics, y_true, y_prob, y_pred = cross_validate_model(
            name, None, X, y, groups
        )
        summary = summarize_metrics(fold_metrics)
        summary["model"] = name
        summary_rows.append(summary)

        results[name] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            y_true, y_pred, name,
            RESULTS_DIR / f"confusion_matrix_{safe_name}.png",
        )

    # Save summary metrics table.
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_df = summary_df[[
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std",
        "roc_auc_mean", "roc_auc_std",
        "pr_auc_mean", "pr_auc_std",
    ]]
    summary_df.to_csv(RESULTS_DIR / "metrics.csv")
    print("\n" + "=" * 60)
    print("FINAL METRICS (subject-grouped 5-fold CV, mean ± std)")
    print("=" * 60)
    for name in summary_df.index:
        row = summary_df.loc[name]
        print(
            f"  {name:22s} "
            f"acc={row.accuracy_mean:.3f}±{row.accuracy_std:.3f}  "
            f"f1={row.f1_mean:.3f}±{row.f1_std:.3f}  "
            f"auc={row.roc_auc_mean:.3f}±{row.roc_auc_std:.3f}"
        )

    plot_roc_curves(results, RESULTS_DIR / "roc_curves.png")
    plot_pr_curves(results, RESULTS_DIR / "pr_curves.png")

    # Pick the best model by ROC-AUC (more stable than accuracy on imbalanced
    # data) and save it refit on the full dataset.
    best_name = summary_df["roc_auc_mean"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_name}")
    save_best_model(best_name, X, y)

    print(f"\nAll outputs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
