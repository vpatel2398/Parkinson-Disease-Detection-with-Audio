"""
Streamlit demo for Parkinson's voice analysis.

This app runs on Hugging Face Spaces and provides a browser-based interface
for the trained model. Users can either record audio directly in the browser
or upload a .wav file, and the app returns a probability score along with
the extracted acoustic features.

This is a thin UI layer — all the ML logic (feature extraction, model
inference) is identical to the local CLI pipeline (`python -m src.predict`).
The same model artifacts are used in both places.

Disclaimer: This is an educational demo only. Not a medical device. Not a
diagnostic tool. See the full README for limitations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from features import extract_features


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🎙️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load model artifacts once at startup (cached so we don't reload per request)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and schema from disk. Cached across sessions."""
    here = Path(__file__).resolve().parent
    model = joblib.load(here / "best_model.joblib")
    scaler = joblib.load(here / "scaler.joblib")
    with open(here / "feature_names.json") as f:
        schema = json.load(f)
    return model, scaler, schema


MODEL, SCALER, SCHEMA = load_artifacts()
FEATURE_NAMES = SCHEMA["features"]
MODEL_NAME = SCHEMA["model"]


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def analyze_audio_file(audio_path: str):
    """Run the full inference pipeline on a single audio file."""
    features = extract_features(audio_path)
    feature_vector = features.to_feature_vector().reshape(1, -1)

    # Sanity check
    if features.feature_names() != FEATURE_NAMES:
        raise RuntimeError("Feature schema mismatch between training and inference.")

    scaled = SCALER.transform(feature_vector)
    proba_pd = float(MODEL.predict_proba(scaled)[0, 1])

    return features, feature_vector.flatten(), proba_pd


def render_results(audio_bytes: bytes, file_suffix: str = ".wav"):
    """
    Run analysis on raw audio bytes and render the result UI.

    Args:
        audio_bytes: Raw audio data (from recorder or file uploader).
        file_suffix: File extension for the temp file (e.g. '.wav', '.mp3').
    """
    # Praat needs a file path, so we write the bytes to a temp file.
    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Audio playback
    st.audio(audio_bytes)

    # Run analysis
    with st.spinner("Extracting acoustic features and running model..."):
        try:
            features, feature_values, proba_pd = analyze_audio_file(tmp_path)
            proba_healthy = 1.0 - proba_pd
            confidence = abs(proba_pd - 0.5) * 2.0
            predicted_class = (
                "Parkinson's pattern detected"
                if proba_pd >= 0.5
                else "Healthy pattern"
            )

            # ---- Results display ----
            st.divider()
            st.subheader("Results")

            # Top-level metrics in 3 columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted class", predicted_class)
            col2.metric("Probability (PD)", f"{proba_pd:.1%}")
            col3.metric("Model confidence", f"{confidence:.1%}")

            # Probability bars
            st.markdown("**Class probabilities**")
            st.progress(proba_pd, text=f"Parkinson's: {proba_pd:.1%}")
            st.progress(proba_healthy, text=f"Healthy: {proba_healthy:.1%}")

            # Key acoustic markers
            st.markdown("### Key acoustic markers")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean F0", f"{features.mdvp_fo_hz:.1f} Hz")
            m2.metric("Jitter (local)", f"{features.mdvp_jitter_pct:.3f}%")
            m3.metric("Shimmer (local)", f"{features.mdvp_shimmer:.4f}")
            m4.metric("HNR", f"{features.hnr:.2f} dB")

            st.caption(
                f"Audio: {features.duration_s:.2f}s @ {features.sample_rate} Hz "
                f"· Model: {MODEL_NAME}"
            )

            # Full feature table in an expander
            with st.expander("Show all 16 acoustic features"):
                feature_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Value": [f"{v:.6f}" for v in feature_values],
                })
                st.dataframe(feature_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"**Error analyzing audio:** {type(e).__name__}: {e}")
            st.markdown("""
            **Common causes:**
            - Audio is too short (need 1+ seconds of sustained vowel)
            - Audio doesn't contain voiced speech (silence, music, noise)
            - Recording is too quiet — try speaking louder and closer to the mic
            - Recording is in a noisy environment
            """)
        finally:
            # Clean up the temp file
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎙️ Parkinson's Voice Analysis Demo")

st.markdown("""
Record or upload a **sustained `/a/` vowel** (5+ seconds works best) and the
system will extract 16 acoustic features using
[Praat](https://www.fon.hum.uva.nl/praat/) and run a Random Forest classifier
trained on the UCI Parkinson's voice dataset.

**How to record a good sample:**
1. Find a quiet room
2. Take a deep breath
3. Say "aaaaaaaah" at a comfortable pitch for 5+ seconds
4. Keep your mouth open at the same shape the whole time
""")

st.divider()

# ---------------------------------------------------------------------------
# Two input methods in tabs: record in browser OR upload file
# ---------------------------------------------------------------------------
tab_record, tab_upload = st.tabs(["🎤 Record in browser", "📁 Upload a file"])

with tab_record:
    st.markdown(
        "Click the button below to start recording, then click it again to stop. "
        "Your browser will ask for microphone permission the first time."
    )

    audio = mic_recorder(
        start_prompt="🔴 Start recording",
        stop_prompt="⏹️ Stop recording",
        just_once=False,        # Allow re-recording
        use_container_width=True,
        format="wav",           # Praat is happiest with WAV
        key="recorder",
    )

    if audio is not None:
        # mic_recorder returns a dict: {'bytes': ..., 'sample_rate': ..., 'sample_width': ..., 'id': ...}
        audio_bytes = audio["bytes"]

        # Approximate duration check — give the user feedback before analysis.
        # The bytes object includes the WAV header (44 bytes) plus 16-bit PCM.
        sample_rate = audio.get("sample_rate", 16000)
        sample_width = audio.get("sample_width", 2)
        approx_duration = max(0, (len(audio_bytes) - 44)) / (sample_rate * sample_width)

        if approx_duration < 1.0:
            st.warning(
                f"Recording is only ~{approx_duration:.1f}s. Try recording for "
                f"at least 5 seconds of sustained vowel."
            )
        else:
            st.success(f"Recorded ~{approx_duration:.1f}s of audio. Analyzing...")

        render_results(audio_bytes, file_suffix=".wav")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a .wav file with sustained /a/ vowel phonation",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        help="Recommended: 5+ seconds of steady /a/ vowel at 16 kHz or higher",
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".wav"
        render_results(uploaded_file.read(), file_suffix=suffix)

# ---------------------------------------------------------------------------
# Disclaimer footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
### ⚠️ Important disclaimers

- **Not a medical device.** This model has not been clinically validated.
- **Cannot diagnose** Parkinson's disease or any condition.
- Trained on **31 subjects only** — results may not generalize to your demographic, language, or recording setup.
- For real concerns about voice or motor symptoms, see a neurologist.

This is a portfolio project demonstrating end-to-end ML engineering: scientifically
valid feature extraction (Praat), subject-grouped cross-validation, and honest
reporting of limitations. [Full methodology on GitHub](https://github.com/vpatel2398/Parkinson-Disease-Detection-with-Audio).
""")