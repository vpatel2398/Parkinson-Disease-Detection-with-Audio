"""
Acoustic feature extraction for Parkinson's disease detection from voice.

Uses Praat (via the parselmouth library) to compute scientifically validated
voice quality measures from sustained vowel phonation.

The features computed here match a subset of the UCI Parkinson's dataset
(Little et al. 2007), specifically the features that can be honestly
reproduced from raw audio using open-source tools.

Features NOT computed here (and why):
    - RPDE, DFA, D2: nonlinear dynamics features from proprietary research
      code. Approximations exist (e.g. via the `nolds` library) but they do
      not match the original implementation exactly, so we exclude them
      rather than introduce silent distribution shift between training
      and inference.
    - spread1, spread2, PPE: derived from a nonlinear transform of F0 used
      in the original paper. Same reasoning as above.

Reference:
    Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A., &
    Moroz, I. M. (2007). Exploiting nonlinear recurrence and fractal scaling
    properties for voice disorder detection. BioMedical Engineering OnLine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call


# Praat analysis settings. These match Praat's default voice report settings
# and the parameters used in the Little et al. (2007) reference implementation.
F0_MIN_HZ = 75.0          # Lower bound for pitch tracking (adult voice floor)
F0_MAX_HZ = 600.0         # Upper bound for pitch tracking
TIME_STEP_S = 0.0         # 0 = let Praat choose (typically 0.75 / F0_MIN)
JITTER_PERIOD_MIN = 0.0001  # 0.1 ms — Praat default
JITTER_PERIOD_MAX = 0.02    # 20 ms — Praat default
JITTER_MAX_FACTOR = 1.3     # Praat default: reject periods differing by >30%
SHIMMER_MAX_AMP_FACTOR = 1.6  # Praat default for shimmer


@dataclass
class VoiceFeatures:
    """
    Container for the 18 acoustic features used by this project.

    Naming follows the UCI Parkinson's dataset column names so that features
    extracted from a new audio file can be passed directly to a model trained
    on the UCI data.
    """
    # Fundamental frequency statistics (Hz)
    mdvp_fo_hz: float       # Mean F0
    mdvp_fhi_hz: float      # Maximum F0
    mdvp_flo_hz: float      # Minimum F0

    # Jitter — cycle-to-cycle variation in F0 period
    mdvp_jitter_pct: float  # Local jitter as percentage
    mdvp_jitter_abs: float  # Local jitter in seconds
    mdvp_rap: float         # Relative average perturbation (3-point)
    mdvp_ppq: float         # Five-point period perturbation quotient
    jitter_ddp: float       # Difference of differences of periods (= 3 * RAP)

    # Shimmer — cycle-to-cycle variation in amplitude
    mdvp_shimmer: float     # Local shimmer (relative)
    mdvp_shimmer_db: float  # Local shimmer in dB
    shimmer_apq3: float     # 3-point amplitude perturbation quotient
    shimmer_apq5: float     # 5-point amplitude perturbation quotient
    mdvp_apq: float         # 11-point amplitude perturbation quotient
    shimmer_dda: float      # Difference of differences of amplitudes

    # Harmonics-to-noise measures
    nhr: float              # Noise-to-harmonics ratio
    hnr: float              # Harmonics-to-noise ratio (dB)

    # Source file metadata (for traceability — not used as a model feature)
    duration_s: float
    sample_rate: int

    def to_feature_vector(self) -> np.ndarray:
        """
        Return features in the fixed order expected by the trained model.
        Excludes metadata fields (duration_s, sample_rate).
        """
        return np.array([
            self.mdvp_fo_hz, self.mdvp_fhi_hz, self.mdvp_flo_hz,
            self.mdvp_jitter_pct, self.mdvp_jitter_abs,
            self.mdvp_rap, self.mdvp_ppq, self.jitter_ddp,
            self.mdvp_shimmer, self.mdvp_shimmer_db,
            self.shimmer_apq3, self.shimmer_apq5, self.mdvp_apq,
            self.shimmer_dda,
            self.nhr, self.hnr,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """Column names in the same order as `to_feature_vector()`."""
        return [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)",
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ",
            "Shimmer:DDA",
            "NHR", "HNR",
        ]


def extract_features(audio_path: str | Path) -> VoiceFeatures:
    """
    Extract acoustic features from a single audio file.

    Args:
        audio_path: Path to a .wav (or any format Praat reads) audio file
                    containing sustained vowel phonation. Recommended:
                    at least 3 seconds of steady /a/ vowel, 16 kHz or higher
                    sample rate, mono.

    Returns:
        VoiceFeatures instance with all 16 model features populated.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If the audio is too short or pitch tracking fails
                    (e.g. silence, noise, non-vocal content).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the sound. Praat handles the resampling and format conversion.
    sound = parselmouth.Sound(str(audio_path))
    duration = sound.get_total_duration()

    if duration < 1.0:
        raise ValueError(
            f"Audio is too short ({duration:.2f}s). "
            f"Need at least 1 second of sustained vowel phonation."
        )

    # ----------------------------------------------------------------------
    # F0 (fundamental frequency) statistics
    # ----------------------------------------------------------------------
    # We use Praat's autocorrelation-based pitch tracker. The PointProcess
    # below is built from the same pitch object, so all subsequent jitter
    # measurements are consistent with these F0 values.
    pitch = call(sound, "To Pitch", TIME_STEP_S, F0_MIN_HZ, F0_MAX_HZ)

    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    if np.isnan(mean_f0):
        raise ValueError(
            "Pitch tracking failed — audio may not contain voiced speech."
        )

    # ----------------------------------------------------------------------
    # Build a PointProcess (glottal pulse train) for jitter/shimmer
    # ----------------------------------------------------------------------
    # The PointProcess marks the location of each glottal pulse in time.
    # Jitter measures variation in time between consecutive pulses;
    # shimmer measures variation in amplitude at those pulses.
    point_process = call(
        sound, "To PointProcess (periodic, cc)", F0_MIN_HZ, F0_MAX_HZ
    )

    # ----------------------------------------------------------------------
    # Jitter measurements
    # ----------------------------------------------------------------------
    # All jitter measures use the same Praat call signature:
    #   (start_time, end_time, period_floor, period_ceiling, max_period_factor)
    # Setting start=end=0 means "use entire signal".
    jitter_local = call(
        point_process, "Get jitter (local)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX, JITTER_MAX_FACTOR
    )
    jitter_local_abs = call(
        point_process, "Get jitter (local, absolute)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX, JITTER_MAX_FACTOR
    )
    jitter_rap = call(
        point_process, "Get jitter (rap)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX, JITTER_MAX_FACTOR
    )
    jitter_ppq5 = call(
        point_process, "Get jitter (ppq5)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX, JITTER_MAX_FACTOR
    )
    jitter_ddp = call(
        point_process, "Get jitter (ddp)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX, JITTER_MAX_FACTOR
    )

    # ----------------------------------------------------------------------
    # Shimmer measurements
    # ----------------------------------------------------------------------
    # Shimmer needs both the Sound and the PointProcess. Signature is:
    #   (sound, point_process, start, end, period_floor, period_ceiling,
    #    max_period_factor, max_amplitude_factor)
    shimmer_local = call(
        [sound, point_process], "Get shimmer (local)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )
    shimmer_local_db = call(
        [sound, point_process], "Get shimmer (local_dB)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )
    shimmer_apq3 = call(
        [sound, point_process], "Get shimmer (apq3)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )
    shimmer_apq5 = call(
        [sound, point_process], "Get shimmer (apq5)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )
    shimmer_apq11 = call(
        [sound, point_process], "Get shimmer (apq11)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )
    shimmer_dda = call(
        [sound, point_process], "Get shimmer (dda)",
        0, 0, JITTER_PERIOD_MIN, JITTER_PERIOD_MAX,
        JITTER_MAX_FACTOR, SHIMMER_MAX_AMP_FACTOR
    )

    # ----------------------------------------------------------------------
    # Harmonics-to-noise ratio (HNR)
    # ----------------------------------------------------------------------
    # HNR in dB. A healthy voice typically shows HNR around 20 dB for /a/;
    # pathological voices show lower values due to increased aperiodic noise.
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, F0_MIN_HZ, 0.1, 1.0)
    hnr_db = call(harmonicity, "Get mean", 0, 0)

    # Noise-to-harmonics ratio: the inverse of HNR, expressed as a linear ratio.
    # Convert HNR from dB to a linear ratio first: 10^(HNR/10) = harmonics/noise.
    # NHR = noise/harmonics = 1 / (10^(HNR/10)).
    nhr = 1.0 / (10.0 ** (hnr_db / 10.0)) if not np.isnan(hnr_db) else np.nan

    return VoiceFeatures(
        mdvp_fo_hz=float(mean_f0),
        mdvp_fhi_hz=float(max_f0),
        mdvp_flo_hz=float(min_f0),
        # Praat returns jitter as a fraction (e.g. 0.005); the UCI dataset
        # stores it as a percentage (e.g. 0.5). Multiply by 100 to match.
        mdvp_jitter_pct=float(jitter_local) * 100.0,
        mdvp_jitter_abs=float(jitter_local_abs),
        mdvp_rap=float(jitter_rap),
        mdvp_ppq=float(jitter_ppq5),
        jitter_ddp=float(jitter_ddp),
        mdvp_shimmer=float(shimmer_local),
        mdvp_shimmer_db=float(shimmer_local_db),
        shimmer_apq3=float(shimmer_apq3),
        shimmer_apq5=float(shimmer_apq5),
        mdvp_apq=float(shimmer_apq11),
        shimmer_dda=float(shimmer_dda),
        nhr=float(nhr),
        hnr=float(hnr_db),
        duration_s=float(duration),
        sample_rate=int(sound.sampling_frequency),
    )


if __name__ == "__main__":
    # Quick smoke test when the module is run directly.
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.features <path_to_audio.wav>")
        sys.exit(1)

    features = extract_features(sys.argv[1])
    print(f"Audio duration: {features.duration_s:.2f}s @ {features.sample_rate} Hz\n")
    for name, value in asdict(features).items():
        print(f"  {name:25s} {value}")
