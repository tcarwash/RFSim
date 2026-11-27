"""Measured-inspired Baofeng K6 FD6818B GP0 microphone source impairment.

This is an AUDIO-SOURCE model, not an RF channel, not a Sotto codec, and not a
sample-exact reconstruction of individual GP0 sigma-delta bits. It is calibrated
only by the Nov 27 2026 attended physical K6 tone and generator-OFF captures.
The profile preserves the observed *pre-vocoder* noise independently of the
receiver's RF impairment. It intentionally outputs nonzero source noise when
the analog microphone input is silent.

Input x: finite mono float at 8000 samples/s. An input sine of amplitude 1.0
corresponds to a 0.1 Vpp signal-generator sine across the microphone input.
That convention is a laboratory reference only, NOT real speech calibration.
Output: approximate GP0-path PCM in float full-scale (-1..+1); optional clipping.

Noise spectral weighting is based on a different Nov 26 PCM diagnostic's short
residual band fractions, NOT a fully identified GP0 RFIC noise spectrum.
Synthetic extrapolation outside the measured 500, 1000 and 2800Hz response
is explicitly assumed. The empirical captures are only 32-44ms each.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

FS = 8000
MEASURED_HZ = (500.0, 1000.0, 2800.0)
# All measured with 0.1 Vpp source, original unfiltered Rust decimator.
MEASURED_PCM_PEAK_FS = (0.3983881, 0.4183966, 0.3227898)
MEASURED_SNR_DB = (16.52549, 17.05840, 15.10228)
# Source: GP0 RAW AIR generator OFF, CRC-verified longest contiguous segment,
# replayed through the existing Rust decimator.
OFF_PCM_RMS_FS = 0.0423111
# Independent short pre-vocoder PCM diagnostic, Nov 26 1kHz tone residual.
# Fractions sum to exactly 1.0; source noise may vary with RFIC settings.
RESIDUAL_BAND_EDGES_HZ = (0.,300.,800.,1500.,2500.,3500.,4000.)
RESIDUAL_BAND_FRACTIONS = (.085,.115,.150,.266,.263,.121)


@dataclass(frozen=True)
class K6MicProfile:
    fs: int = FS
    reference_vpp: float = 0.1
    noise_rms_fs: float = OFF_PCM_RMS_FS
    response_hz: tuple[float,...] = MEASURED_HZ
    response_peak_fs: tuple[float,...] = MEASURED_PCM_PEAK_FS


def _validate(x, fs, noise_rms_fs, input_vpp):
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 1 or len(data) < 128 or not np.isfinite(data).all():
        raise ValueError("finite mono audio with >=128 samples required")
    if fs != FS:
        raise ValueError("physical GP0 source profile is calibrated only at 8000Hz")
    if not np.isfinite(noise_rms_fs) or noise_rms_fs < 0 or noise_rms_fs > 1:
        raise ValueError("noise RMS must be finite and between 0 and 1")
    if not np.isfinite(input_vpp) or input_vpp <= 0 or input_vpp > 1:
        raise ValueError("input_vpp must be finite, >0 and <=1")
    return data


def _source_transfer(n: int, fs: float, *, kind: str) -> np.ndarray:
    """Frequency-domain amplitude response from only 3 measured tone points."""
    freq = np.fft.rfftfreq(n, 1./fs)
    if kind == "constant":
        # Sensitivity test: assume flat response equal to measured 1kHz.
        return np.full(len(freq), MEASURED_PCM_PEAK_FS[1], dtype=float)
    if kind != "measured":
        raise ValueError("response must be 'measured' or 'constant'")
    # 0..500Hz left clamped and 2800..Nyquist right clamped: both unmeasured.
    # This extrapolation is NOT evidence of a physical flat end-band.
    return np.interp(freq, MEASURED_HZ, MEASURED_PCM_PEAK_FS)


def _noise_weights(freq: np.ndarray, *, kind: str) -> np.ndarray:
    if kind == "white":
        return np.ones(len(freq), dtype=float)
    if kind != "banded":
        raise ValueError("noise model must be 'banded' or 'white'")
    edges = np.array(RESIDUAL_BAND_EDGES_HZ)
    fractions = np.array(RESIDUAL_BAND_FRACTIONS)
    idx = np.clip(np.searchsorted(edges[1:], freq, side="right"), 0, len(fractions)-1)
    return np.sqrt(fractions[idx] / np.diff(edges)[idx])


def generate_gp0_noise(n: int, *, seed: int = 0,
                       model: str = "banded", rms_fs: float = OFF_PCM_RMS_FS) -> np.ndarray:
    """Produce repeatable zero-mean PCM noise shaped using limited physical evidence.

    This is a *statistical approximation*, not replayed true K6 popcount noise.
    No external data or file accesses, so suitable for DORC Monte Carlo runs.
    """
    if not isinstance(n, int) or n < 128 or n > 20_000_000:
        raise ValueError("noise length must be 128..20m samples")
    if not np.isfinite(rms_fs) or not 0 <= rms_fs <= 1:
        raise ValueError("invalid RMS")
    if model not in ("white", "banded"):
        raise ValueError("unknown noise model")
    if rms_fs == 0:
        return np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    weights = _noise_weights(np.fft.rfftfreq(n, 1. / FS), kind=model)
    shaped = np.fft.irfft(np.fft.rfft(noise) * weights, n=n)
    shaped -= shaped.mean()
    sigma = np.std(shaped)
    if sigma <= 0 or not math.isfinite(float(sigma)):
        raise ArithmeticError("invalid simulated noise")
    return shaped * (rms_fs / sigma)


def k6_gp0_pcm(x, fs: int = FS, *, seed: int = 0,
               input_vpp: float = 0.1, noise_rms_fs: float = OFF_PCM_RMS_FS,
               response: str = "measured", noise_model: str = "banded",
               clip: bool = True) -> np.ndarray:
    """Approximate K6 microphone PCM before Sotto coding.

    x is normalized to the specified laboratory input_vpp reference; a sine
    with x amplitude=1 models that Vpp. This function *does not* model a
    different hardware input gain when changing input_vpp; it merely rescales
    the reference sine amplitude, preserving the measured noise floor.
    """
    data = _validate(x, fs, noise_rms_fs, input_vpp)
    if response not in ("measured", "constant") or noise_model not in ("banded", "white"):
        raise ValueError("invalid GP0 model option")
    gain = _source_transfer(len(data), fs, kind=response)
    source = np.fft.irfft(np.fft.rfft(data) * gain, n=len(data))
    # input_vpp is the actual voltage modeled by x amplitude=1.
    source *= input_vpp / 0.1
    noise = generate_gp0_noise(len(data), seed=seed, model=noise_model, rms_fs=noise_rms_fs)
    out = source + noise
    if clip:
        np.clip(out, -1, 1, out=out)
    return out


def k6_gp0_profile() -> dict:
    """Self-describing model metadata for RFSim consumers and MCP clients."""
    return {
        "name": "k6-fd6818b-gp0-audio-source",
        "stage": "microphone-to-PCM BEFORE vocoder and RF channel",
        "sample_rate_hz": FS,
        "reference_input_vpp": 0.1,
        "physical_tone_points_hz": list(MEASURED_HZ),
        "physical_tone_peaks_full_scale": list(MEASURED_PCM_PEAK_FS),
        "physical_tone_snr_db": list(MEASURED_SNR_DB),
        "physical_generator_off_pcm_rms_full_scale": OFF_PCM_RMS_FS,
        "noise_band_edges_hz": list(RESIDUAL_BAND_EDGES_HZ),
        "estimated_noise_band_fractions": list(RESIDUAL_BAND_FRACTIONS),
        "uncertainty": [
            "Only three measured tone frequencies and 32-44ms GP0 captures",
            "Noise bands from a separate short physical pre-vocoder PCM capture",
            "Stationary additive source noise is an approximation, not a diagnosed mechanism",
            "Outside 500-2800Hz response is held constant, an unsupported extrapolation",
            "Input amplitude calibration assumes 0.1Vpp test generator, not speech SPL",
            "Does not model exact GP0 1-bit PDM, correlated timing/aliasing, gain AGC or input overload",
            "Does not model Sotto encoder or decoder or packet loss",
            "FM receiver quieting belongs to a separate RF channel and FM discriminator model",
        ],
    }
