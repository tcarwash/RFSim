# RFSim GPU-native fading and AWGN

**Opt in only.** The original RFSim `apply_iq` and DORC common CPU
regression remain unchanged. The PyTorch batch backend now offers two
independent stochastic modes:

| Choice | Random fading / noise | Validation | Use case |
|---|---|---|---|
| `backend="torch", rng="numpy"` (previous default) | NumPy CPU arrays copied to GPU | Samplewise comparison to `apply_iq` | Numerical parity and debugging |
| `backend="torch", rng="torch"` (new) | Device-native Torch generators, Gaussian filtering, interpolation, Rician gain and AWGN | Statistical fidelity and deterministic per-device replay | Long Monte Carlo / repeated GPU experiments |

Both CUDA and ROCm use PyTorch's `device="cuda"` spelling. The default
`backend="numpy"` still uses legacy RFSim without a Torch dependency.

```python
from rfsim.batch import apply_iq_batch
iq = ...  # 1-D, finite complex numpy array
cases = [
    {"profile":"awgn", "snr_db":10, "seed":17},
    {"profile":"mpp", "snr_db":5, "seed":23},
    {"profile":"vhf-mobile", "snr_db":0, "seed":37},
]
output = apply_iq_batch(
    iq, 12000, cases, backend="torch", device="cuda",
    rng="torch", batch_size=4, dtype="complex64",
)
```

## What moves onto the GPU

The optional mode uses `torch.Generator(device=...)` with separate,
reproducible per-case/per-tap/noise streams. Every tap's white Gaussian
process is generated on the device, convolved with RFSim's Gaussian
Doppler-spread kernel, normalized and interpolated into the complex gain.
The existing tap-delay and CFO mixing remain device operations; AWGN is
also generated on device. No full-size random fading/noise arrays are
created or transferred on the CPU.

The CPU still creates tiny integer seed descriptors and the common input IQ
waveform is transferred once per batch. Output IQ is copied back because
the public batched API returns NumPy arrays. There is no implicit CPU
fallback, and the Torch path still deliberately rejects `clock_ppm`
resampling.

## Reproducibility and scientific limits

- Same seed/case on the **same** Torch runtime and device repeats exactly,
  independently of batch grouping, request order and global Torch RNG state.
- Torch uses different random streams than NumPy; do **not** require
  samplewise equality with `apply_iq`. CUDA, ROCm, and future framework
  releases can also generate different sequences.
- Validate AWGN variance/SNR, nonfinite output, Rayleigh/Rician fading
  populations, and pooled channel-power statistics. The DORC benchmark
  explicitly labels `validation_mode="statistical"`, not samplewise.
- Built-in HF, NVIS, VHF and UHF profiles are tested. The generic GPU
  path still rejects unsupported clock-resampling and enforces tap/memory
  bounds. Distribution screening is not a formal radio-path measurement,
  nor does it establish modem BER or PER.
- The existing `rng="numpy"` path remains a critical numerical reference.

## Cold-start versus steady-state benchmark

```sh
python -m experiments.dorc_cpu_gpu_compare \
  --config /inputs/suite/fixtures/rfsim/common_suite.json \
  --rng torch --samples 131072 --batch-size 4 \
  --reference-cases 45 --gpu-warmups 1 --gpu-trials 3
```

In DORC, `EXPERIMENT_OUTPUT=/output` is injected by the runner. Artifacts
remain flat and provenance-compatible: paired report and metrics, separate
CPU/GPU metrics and CSVs, plus `gpu_runs.json` recording cold execution,
warmup count, all measured trial times and statuses.

The first GPU iteration includes runtime/context and kernel setup.
The separate **warm median** includes full channel generation and GPU/host
transfers but excludes initial context and compilation. The CPU baseline
is measured on the same machine. Both speedup metrics are reported; neither
should replace the other.

Initial one-session host-Python studies using the common 45-case,
131,072-complex-sample matrix showed approximately 0.56× (CUDA) and
0.14× (ROCm) CPU speed in a cold start, but ~11.6× and ~11.5× CPU speed
respectively after warming, with three measured warm trials each.
These results require independent containerized DORC replication and
should not be interpreted as confidence intervals or GPU universal wins.

The original CPU-only baseline stays the default for tiny or one-shot
experiments. A contributor must independently review and allowlist its
exact container image ID to run the new GPU workload.
