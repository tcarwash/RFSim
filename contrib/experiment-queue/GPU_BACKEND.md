# Optional RFSim batched IQ GPU backend

The original `apply_iq` and DORC common 45-case regression baseline remain
NumPy/CPU-only. GPU acceleration is an **explicit alternative**, not a
default or an automatic fallback.

The opt-in API in `rfsim.batch` exposes `apply_iq_batch` and the bounded
streaming `iter_iq_batches`. They accept a common input IQ waveform and
multiple independent case dictionaries (`profile`, `snr_db`, `seed`,
`cfo_hz`, `noise_bw_hz`, `signal_power`).

The optional PyTorch backend supports `device="cuda"` for **both** NVIDIA
CUDA and AMD ROCm (PyTorch's ROCm device API uses the `cuda` spelling). The
NumPy backend works without PyTorch installed. A missing or unusable GPU
raises an error instead of switching to CPU.

```python
import numpy as np
from rfsim.batch import apply_iq_batch

iq = np.ones(65536, np.complex128)
cases = [
    {"profile": "awgn", "snr_db": 5, "seed": 17},
    {"profile": "mpp", "snr_db": 10, "seed": 23},
    {"profile": "vhf-mobile", "snr_db": 20, "seed": 37},
]
cpu = apply_iq_batch(iq, 12000, cases, backend="numpy")
gpu = apply_iq_batch(
    iq, 12000, cases, backend="torch",
    device="cuda", dtype="complex64", batch_size=3,
)
np.testing.assert_allclose(gpu, cpu, rtol=8e-4, atol=8e-4)
```

The backend uses the original NumPy/Scipy seeded generator for **fading and
noise**, then performs batched convolution, tap multiplication, CFO mixing and
noise addition with Torch tensors. This preserves near-parity with RFSim's
legacy reference channel and avoids changing the random channel's physics
while introducing acceleration.

**Performance caveat:** stochastic coefficient generation, GPU transfers and
result copies are still counted and are CPU-bound. In initial tests on
Tobios (GTX 1080 Ti), 45×131,072-sample cases achieved approximately
5.47 M complex samples/s on NumPy versus 2.62 M/s using CUDA Torch. A
small GPU test is usually slower. This first backend provides an honest
platform for profiling and subsequent GPU-native stochastic generation.

Current restrictions: only offline `apply_iq` burst-style tap profiles;
no clock-ppm resampling in Torch; no `ChannelSession` timelines,
`MeasuredChannel` replay, audio modulation, or integrated modem decode in
the accelerated API. Invalid unsupported options fail explicitly. The NumPy
API is unchanged.

## DORC throughput benchmark

Run the pinned DORC matrix with the optional backend:

```sh
python -m experiments.dorc_backend_benchmark \
  --config /path/to/dorc_experiments/fixtures/rfsim/common_suite.json \
  --backend torch --device cuda --dtype complex64 \
  --samples 131072 --repetitions 1 --batch-size 8 \
  --reference-cases 12 --output /tmp/rfsim-benchmark
```

Set `--backend numpy` to collect a CPU result **on the same machine**.
The benchmark records end-to-end throughput, numerical differences against
a configurable number of CPU-reference cases, torch/NumPy versions,
device name and a case CSV. It outputs `report.md`, `metrics.json` and
`cases.csv`. For larger Monte Carlo sweeps, increase `--repetitions` and
`--samples`; chunking and `--staging-megabytes` limit staging memory.
Torch also checks the **number of distinct tap delays** before allocating
GPU buffers, limiting memory use from large custom multipath profiles.
Nonfinite sample outputs and overflowed channel power are reported as a
`NEGATIVE_FINDING` with valid JSON/CSV evidence instead of crashing during
artifact serialization.
A benchmark without numerical reference comparisons is marked
`UNVERIFIED`, not `PASS`.

For a *paired on-host CPU-vs-GPU run*, use
`python -m experiments.dorc_cpu_gpu_compare`, which produces flat result
artifacts for DORC and measures speedup without comparing different machines.
See [GPU contributor images](GPU_CONTAINERS.md) for reproducible CUDA/ROCm
Dockerfiles and the queue policy process.

The GPU runtime is intentionally **not included** in RFSim's default
dependencies. Install a Torch wheel compatible with the contributor's
driver and architecture inside an *approved, pinned container image*.
GTX 1080 Ti is compute 6.1 (Pascal) and may not work with newer Torch
builds; RX 6700 XT (gfx1031) may require the contributor's tested
`HSA_OVERRIDE_GFX_VERSION` and a supported ROCm Torch build. Check the
numerical smoke test on the target device before accepting a GPU job.

Use DORC's dedicated GPU experiment issue template, which requires a pinned
RFSim source commit and a separately pinned DORC matrix commit. The
contributor's local policy remains authoritative for images, GPU devices,
CPU/RAM/VRAM and runtime budgets. Do not expand its allowlist through an issue.
