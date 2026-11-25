"""Opt-in batched IQ simulation using NumPy or PyTorch (CUDA/ROCm).

The PyTorch path transfers *the same* seeded NumPy fading/noise realization
used by apply_iq, and accelerates convolution/multipath/channel arithmetic.
It intentionally does not claim that CPU-side coefficient generation is GPU
accelerated. That preserves a useful correctness oracle while enabling large
batched workloads. No torch import is needed for the NumPy path.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import itertools
import math

import numpy as np
from scipy.fft import next_fast_len

from . import PROFILES, _gain, apply_iq, load_profile


_ALLOWED_CASE_KEYS = frozenset(
    {"profile", "seed", "snr_db", "noise_bw_hz", "signal_power", "cfo_hz", "clock_ppm"}
)


def _prepare(samples, fs, cases, batch_size):
    x = np.asarray(samples, dtype=np.complex128)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("expected nonempty, finite, 1-D IQ samples")
    if not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be positive and finite")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    if not isinstance(cases, Iterable) or isinstance(cases, (str, bytes, Mapping)):
        raise ValueError("cases must be an iterable of mappings")
    return x


def _validate_case(case, fs):
    if not isinstance(case, Mapping):
        raise ValueError("each case must be a mapping")
    unexpected = set(case).difference(_ALLOWED_CASE_KEYS)
    if unexpected:
        raise ValueError(f"unsupported case options: {sorted(unexpected)}")
    name = case.get("profile", "awgn")
    if not isinstance(name, str):
        raise ValueError("profile must be a built-in profile name or registry path")
    # Resolve before allocating GPU resources; same profile semantics as apply_iq.
    taps = PROFILES[name] if name in PROFILES else load_profile(name)
    if not taps:
        raise ValueError("empty profile")
    seed = case.get("seed", 0)
    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    ppm = case.get("clock_ppm", 0)
    if ppm != 0:
        raise NotImplementedError(
            "batched torch path does not implement clock resampling; "
            "use apply_iq or the NumPy backend"
        )
    for field in ("snr_db", "noise_bw_hz", "signal_power", "cfo_hz"):
        v = case.get(field)
        if v is not None and (
            type(v) not in (int, float) or not math.isfinite(v)
        ):
            raise ValueError(f"{field} must be finite")
    bw = case.get("noise_bw_hz")
    if bw is not None and not 0 < bw <= fs:
        raise ValueError("noise bandwidth must be within (0, sample rate]")
    power = case.get("signal_power")
    if power is not None and power < 0:
        raise ValueError("negative signal power")
    return tuple(taps)


def _torch_batch(x, fs, cases, *, device, dtype, torch):
    """Run one bounded group using shared input FFTs and GPU tensor math."""
    n = len(x)
    complex_type = {"complex64": torch.complex64, "complex128": torch.complex128}[dtype]
    base = torch.as_tensor(x, dtype=complex_type, device=device)
    result = torch.zeros((len(cases), n), dtype=complex_type, device=device)
    delayed_cache = {}
    freq_cache = {}
    nfft = next_fast_len(n + 64)

    def delayed(delay):
        if delay in delayed_cache:
            return delayed_cache[delay]
        integer = int(np.floor(delay))
        frac = delay - integer
        if integer >= n:
            value = torch.zeros(n, dtype=complex_type, device=device)
        else:
            if frac:
                k = np.arange(-32, 33)
                kernel = np.sinc(k - frac) * np.kaiser(len(k), 8)
                kernel /= kernel.sum()
                if "input_fft" not in freq_cache:
                    freq_cache["input_fft"] = torch.fft.fft(base, n=nfft)
                h = torch.as_tensor(kernel, device=device, dtype=complex_type)
                raw = torch.fft.ifft(
                    freq_cache["input_fft"] * torch.fft.fft(h, n=nfft),
                    n=nfft,
                )[32:32 + n]
            else:
                raw = base
            if integer:
                value = torch.cat(
                    [torch.zeros(integer, device=device, dtype=complex_type), raw[:n - integer]]
                )
            else:
                value = raw
        delayed_cache[delay] = value
        return value

    # First collect exact seeded channel realizations from the NumPy oracle.
    # Group each unique fractional-delay tap for a batched device multiply.
    contributions = {}
    noise = np.zeros((len(cases), n), dtype=np.complex128)
    cfo = []
    input_power = float(np.mean(np.abs(x)**2))
    for index, case in enumerate(cases):
        taps = _validate_case(case, fs)
        seed = case.get("seed", 0)
        fade_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(fade_seed)
        weights = np.array([10**(t.power_db / 10) for t in taps], dtype=float)
        weights /= weights.sum()
        for tap, weight in zip(taps, weights):
            delay = tap.delay_s * fs
            gain = _gain(n, fs, tap, rng)
            if int(np.floor(delay)) >= n:
                continue
            key = delay
            if key not in contributions:
                contributions[key] = np.zeros((len(cases), n), dtype=np.complex128)
            contributions[key][index, :] += np.sqrt(weight) * gain
        snr = case.get("snr_db")
        if snr is not None:
            power = input_power if case.get("signal_power") is None else case["signal_power"]
            bw = fs if case.get("noise_bw_hz") is None else case["noise_bw_hz"]
            nrng = np.random.default_rng(noise_seed)
            noise[index, :] = np.sqrt(power * fs / bw * 10**(-snr / 10) / 2) * (
                nrng.normal(size=n) + 1j * nrng.normal(size=n)
            )
        cfo.append(case.get("cfo_hz") or 0.0)

    with torch.no_grad():
        for delay, coefficients in contributions.items():
            amp = torch.as_tensor(coefficients, dtype=complex_type, device=device)
            result.add_(amp * delayed(delay)[None, :])
        if any(cfo):
            frequencies = torch.as_tensor(cfo, device=device, dtype=torch.float64)[:, None]
            indices = torch.arange(n, device=device, dtype=torch.float64)[None, :]
            phase = torch.remainder(frequencies * indices / fs, 1.0)
            rotation = torch.exp(2j * torch.pi * phase).to(complex_type)
            result.mul_(rotation)
        if any(c.get("snr_db") is not None for c in cases):
            result.add_(torch.as_tensor(noise, dtype=complex_type, device=device))
        return result.cpu().numpy()


def iter_iq_batches(
    samples,
    fs,
    cases,
    *,
    backend="numpy",
    device=None,
    batch_size=16,
    dtype="complex64",
):
    """Yield (metadata, output_array) chunks of simulated complex IQ.

    The CPU NumPy path produces the exact legacy apply_iq output. The optional
    torch path executes channel arithmetic on device while retaining seeded
    NumPy coefficient generation, enabling case-by-case numerical comparison.

    No implicit GPU->CPU fallback: backend='torch' requires a device explicitly,
    and an unavailable CUDA/ROCm device raises an error. Unlike legacy apply_iq,
    the torch path rejects clock_ppm != 0 instead of silently approximating it.

    The output dtype for torch is complex64 or complex128. For NumPy it always
    matches legacy complex128. Chunking controls temporary/device memory, not
    the size of the underlying input waveform.
    """
    x = _prepare(samples, fs, cases, batch_size)
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be numpy or torch")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("dtype must be complex64 or complex128")
    if backend == "torch":
        # A typical batch stages multiple complex coefficient arrays, plus
        # output, input and FFT workspaces; reject unbounded direct requests.
        # The DORC benchmark automatically reduces batch_size as needed.
        if len(x) * batch_size * 16 * 4 > 512 * 1024 * 1024:
            raise ValueError("torch batch staging exceeds 512 MiB; reduce batch_size")
        if device is None:
            raise ValueError("specify device='cuda' or 'cpu' for torch")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is optional; install a CUDA/ROCm-compatible torch build "
                "in the experiment environment"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable; torch backend will not silently use CPU")
    else:
        if device not in (None, "cpu"):
            raise ValueError("NumPy backend supports only the CPU")
        torch = None

    iterator = iter(cases)
    for _ in itertools.count(0, batch_size):
        group = list(itertools.islice(iterator, batch_size))
        if not group:
            break
        if backend == "numpy":
            # CPU backend intentionally supports clock_ppm like legacy RFSim.
            for c in group:
                if not isinstance(c, Mapping):
                    raise ValueError("each case must be a mapping")
                unexpected = set(c).difference(_ALLOWED_CASE_KEYS)
                if unexpected:
                    raise ValueError(f"unsupported case options: {sorted(unexpected)}")
            values = np.stack([apply_iq(x, fs, **c) for c in group])
        else:
            values = _torch_batch(x, fs, group, device=device, dtype=dtype, torch=torch)
        yield group, values


def apply_iq_batch(samples, fs, cases, **kwargs):
    """Return all IQ cases as one 2-D array; use iter_iq_batches for large sweeps."""
    chunks = [values for _, values in iter_iq_batches(samples, fs, cases, **kwargs)]
    if not chunks:
        raise ValueError("cases cannot be empty")
    return np.concatenate(chunks, axis=0)
