"""Device-native stochastic RF channel realizations for optional Torch IQ batches.

The existing numpy-seeded RFSim channel remains the scientific exact-reference
backend. This module uses an *independent* per-case Torch RNG stream on the
requested device. No full-length fading/noise arrays are generated on the host.

The result is reproducible for a particular Torch/device runtime and
independent of batch grouping/order, but **not** samplewise identical across
NumPy, CUDA, ROCm, or different Torch versions. Validate channel statistics.
"""
from __future__ import annotations

import math

import numpy as np


def _generator(torch, device, seed: int, tap: int, stream: int):
    """Stable, independent per-(case,tap,stream) RNG; no global RNG mutation."""
    # numpy SeedSequence derives only tiny integer seeds on CPU; it never
    # generates the stochastic IQ arrays in this backend.
    seed64 = int(np.random.SeedSequence([seed, tap, stream, 0x52465349])
                 .generate_state(1, dtype=np.uint64)[0])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed64)
    return generator


def _complex_gaussian(torch, device, dtype, count: int, generator):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    parts = torch.randn(
        (2, count), generator=generator, device=device, dtype=real_dtype,
    )
    return torch.complex(parts[0], parts[1]) * (1 / math.sqrt(2))


def gain(n, fs, tap, seed, tap_index, *, torch, device, dtype):
    """GPU/torch-native Rayleigh/Rician fading matching RFSim's model.

    Uses the same Gaussian low-pass kernel, normalization, interpolation
    rate, and K-factor as RFSim's NumPy oracle, but an independent RNG.
    """
    if tap.spread_hz == 0:
        return 1.0

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    rate = min(fs, max(32 * tap.spread_hz, 1.0))
    sigma = rate / (math.sqrt(8) * math.pi * (tap.spread_hz / 2))
    pad = int(math.ceil(5 * sigma)) + 2
    count = int(math.ceil(n * rate / fs)) + 2
    radius = int(5 * sigma + 0.5)
    with torch.no_grad():
        generator = _generator(torch, device, seed, tap_index, 0)
        # Separate real and imaginary Gaussian streams, each with variance 1/2.
        real_imag = torch.randn(
            (1, 2, count + 2 * pad), generator=generator,
            device=device, dtype=real_dtype,
        )
        offsets = torch.arange(
            -radius, radius + 1, device=device, dtype=real_dtype
        )
        weights = torch.exp(-0.5 * (offsets / sigma).square())
        weights = weights / weights.sum()
        taps_kernel = weights.view(1, 1, -1).expand(2, 1, -1).contiguous()
        smooth = torch.nn.functional.conv1d(
            real_imag, taps_kernel, groups=2, padding=radius
        )[0, :, pad:pad + count]
        # After Gaussian filtering the original complex white noise has
        # E|z|^2=1. sqrt(sum(weights**2)) is the impulse-response L2 norm.
        smooth = smooth * (1 / math.sqrt(2)) / weights.square().sum().sqrt()
        # np.interp equivalent with clamp against the +2 interpolation guard.
        position = torch.arange(n, device=device, dtype=real_dtype) * (rate / fs)
        lower = position.floor().to(torch.long).clamp(0, count - 2)
        frac = position - lower.to(real_dtype)
        fading = torch.complex(
            torch.lerp(smooth[0, lower], smooth[0, lower + 1], frac),
            torch.lerp(smooth[1, lower], smooth[1, lower + 1], frac),
        )
        if tap.k_db is not None:
            k = 10 ** (tap.k_db / 10)
            fading = (fading + math.sqrt(k)) / math.sqrt(1 + k)
        return fading


def noise(n, variance, seed, *, torch, device, dtype):
    """Generate proper complex AWGN directly on device, no host-sized arrays."""
    if variance < 0 or not math.isfinite(variance):
        raise ValueError("noise variance must be nonnegative and finite")
    generator = _generator(torch, device, seed, 0, 1)
    z = _complex_gaussian(torch, device, dtype, n, generator)
    return z * math.sqrt(variance)
