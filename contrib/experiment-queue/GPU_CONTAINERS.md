# DORC GPU contributor container runtimes

RFSim's CPU-only baseline is unchanged. The optional **paired CPU/GPU**
throughput and numerical-accuracy benchmark uses a separate GPU-aware
container image *per contributor*. The image contains Python, NumPy,
SciPy, and compatible PyTorch only. The actual RFSim source is mounted
read-only at a pinned Git commit, and the test matrix is a separately
pinned DORC input.

## Reviewed build inputs

| Backend | Build definition | Intended device |
|---|---|---|
| CUDA | `Dockerfile.cuda` | NVIDIA GTX 1080 Ti (Pascal, sm_61) and other supported CUDA GPUs |
| ROCm | `Dockerfile.rocm` | AMD RX 6700 XT (gfx1031) with previously validated HSA override |

Both Dockerfiles pin their parent image digest and use PyTorch 2.5.1.
The ROCm image installs the official ROCm 6.2 Torch wheel; the CUDA
image is based on the official PyTorch 2.5.1 CUDA 12.1 runtime.

Build locally (network access is **only** needed for building):

```sh
docker build -f contrib/experiment-queue/Dockerfile.cuda \
  -t rfsim-dorc-gpu:cuda2.5.1 .
docker build -f contrib/experiment-queue/Dockerfile.rocm \
  -t rfsim-dorc-gpu:rocm6.2-torch2.5.1 .
docker image inspect rfsim-dorc-gpu:cuda2.5.1 --format '{{.Id}}'
```

Build each image separately on the appropriate contributor (no assumptions
about binary compatibility, image IDs, disk space, or GPU availability).
Perform a full numerical smoke test using **the same sandbox flags** used
by that contributor's `experiment_queue` instance. Confirm the intended
GPU was actually selected and that the 12 or more NumPy reference
comparisons pass.

The contributor may then allowlist **only that tested immutable local
image ID** in its *private* policy file. DORC's source, image and input
revisions must all be pinned in the issue manifest; a GitHub issue cannot
grant GPU access, approve a new image, or change a contributor's resource
ceilings. Do not give a container host Docker socket access, broad
filesystem mounts, unrestricted network, or `--privileged`.

## Paired benchmark

The paired entry point measures the same deterministic waveform/case
matrix with NumPy and PyTorch **on the same workstation**, then reports
speedup (GPU throughput / CPU throughput) and NumPy parity.

```sh
python -m experiments.dorc_cpu_gpu_compare \
  --config /inputs/suite/fixtures/rfsim/common_suite.json \
  --samples 131072 --batch-size 4 --reference-cases 12
```

The queue injects `EXPERIMENT_OUTPUT=/output`. The experiment writes flat
artifacts accepted by `experiment_queue`: `report.md`, `metrics.json`,
`cpu_metrics.json`, `gpu_metrics.json`, `cpu_cases.csv`, and
`gpu_cases.csv`. All should be included in the result PR's provenance
SHA-256 list.

The elapsed backend time includes CPU-side fading/noise generation,
GPU transfers and tensor arithmetic. A result below **1×** means GPU
execution is slower. No GPU speedup is assumed. This suite tests RFSim
numerical outputs and throughput; it does not measure modem BER/PER
or empirical RF propagation.

## Device-specific cautions

- **Tobios (GTX 1080 Ti)**: tested with PyTorch 2.5.1+cu121;
  many newer frameworks drop CUDA sm_61 support.
- **Desktop (RX 6700 XT)**: tested with PyTorch 2.5.1+rocm6.2 and
  locally authorized `HSA_OVERRIDE_GFX_VERSION=10.3.0` through the
  runner's ROCm GPU policy. This is an unsupported hardware compatibility
  override and must be verified on the actual GPU.
- Docker's `--memory` limit does **not** constrain GPU VRAM. These
  workloads remain opt-in and bounded by separate local GPU device
  selection and `batch_size` / tap-stage memory checks.
- CPU workloads should continue to use the much smaller
  `Dockerfile` runtime. Do not pollute the CPU default with GPU
  libraries or assume those images are interchangeable.
