"""
Inference: predict Parkinson's disease likelihood from a single audio file.

Loads the model and scaler saved by `src/train.py`, extracts acoustic
features from the input audio, applies the same scaling used at training
time, and returns a calibrated probability.

Usage:
    python -m src.predict path/to/recording.wav
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.features import extract_features


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_artifacts():
    """
    Load the trained model, fitted scaler, and feature schema.

    Raises a clear error if training has not been run yet — this is one
    of the most common confusing failure modes for someone cloning the repo.
    """
    model_path = RESULTS_DIR / "best_model.joblib"
    scaler_path = RESULTS_DIR / "scaler.joblib"
    schema_path = RESULTS_DIR / "feature_names.json"

    if not all(p.exists() for p in [model_path, scaler_path, schema_path]):
        raise FileNotFoundError(
            "Trained model artifacts not found. "
            "Run `python -m src.train` first to train the model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(schema_path) as f:
        schema = json.load(f)

    return model, scaler, schema


def predict(audio_path: str | Path) -> dict:
    """
    Run end-to-end inference on a single audio file.

    Args:
        audio_path: Path to a .wav file with sustained vowel phonation.

    Returns:
        Dict with:
            - probability_pd: float in [0, 1], probability of PD class
            - predicted_class: 'PD' or 'Healthy'
            - confidence: |probability - 0.5| * 2, in [0, 1]
            - features: extracted feature values (for transparency)
            - model_name: which classifier produced the prediction

    Important: the returned probability is a model output, not a clinical
    diagnosis. See README for limitations.
    """
    model, scaler, schema = load_artifacts()

    # Extract features. This is the same code path used during training.
    features = extract_features(audio_path)
    feature_vector = features.to_feature_vector().reshape(1, -1)

    # Sanity check: feature order must match what the model was trained on.
    expected = schema["features"]
    actual = features.feature_names()
    if expected != actual:
        raise RuntimeError(
            "Feature order mismatch between training and inference. "
            f"Expected: {expected}\nGot: {actual}"
        )

    # Apply the same scaling fit at training time.
    scaled = scaler.transform(feature_vector)

    # Predict probability of the positive class (PD = 1).
    proba_pd = float(model.predict_proba(scaled)[0, 1])
    predicted_class = "PD" if proba_pd >= 0.5 else "Healthy"
    confidence = abs(proba_pd - 0.5) * 2.0

    return {
        "probability_pd": proba_pd,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_name": schema["model"],
        "features": {
            name: float(value)
            for name, value in zip(features.feature_names(), feature_vector.flatten())
        },
        "audio_duration_s": features.duration_s,
        "sample_rate_hz": features.sample_rate,
    }


def format_prediction(result: dict) -> str:
    """Human-readable formatting of a prediction result."""
    lines = [
        "=" * 50,
        "Parkinson's Voice Analysis — Prediction",
        "=" * 50,
        f"Audio: {result['audio_duration_s']:.2f}s @ {result['sample_rate_hz']} Hz",
        f"Model: {result['model_name']}",
        "",
        f"  Predicted class:     {result['predicted_class']}",
        f"  Probability of PD:   {result['probability_pd']:.1%}",
        f"  Confidence:          {result['confidence']:.1%}",
        "",
        "Key acoustic features:",
        f"  Mean F0:             {result['features']['MDVP:Fo(Hz)']:.1f} Hz",
        f"  Jitter (local):      {result['features']['MDVP:Jitter(%)']:.3f}%",
        f"  Shimmer (local):     {result['features']['MDVP:Shimmer']:.3f}",
        f"  HNR:                 {result['features']['HNR']:.2f} dB",
        "",
        "Disclaimer: This is an educational demo, not a medical diagnosis.",
        "=" * 50,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = predict(audio_path)
    print(format_prediction(result))
