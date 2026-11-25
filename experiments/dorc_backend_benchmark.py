"""Benchmark RFSim CPU and optional batched torch IQ backends for DORC.

The benchmark records wall-clock *end-to-end* channel generation, including
CPU-generated fading/noise and transfer overhead. It compares a configurable
prefix of cases against the original NumPy oracle. A numerical discrepancy is
recorded as a negative finding, while infrastructure failures exit nonzero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from rfsim import apply_iq
from rfsim.batch import iter_iq_batches
from experiments.dorc_channel_suite import read_config


def benchmark(
    config_path: Path,
    output_dir: Path,
    *,
    backend: str = "numpy",
    device: str | None = None,
    dtype: str = "complex64",
    samples: int | None = None,
    repetitions: int = 1,
    batch_size: int = 8,
    reference_cases: int = 12,
    staging_megabytes: int = 256,
):
    config = read_config(config_path)
    if type(repetitions) is not int or not 1 <= repetitions <= 1000:
        raise ValueError("repetitions must be 1..1000")
    n = config["samples"] if samples is None else samples
    if type(n) is not int or not 1024 <= n <= 1_048_576:
        raise ValueError("samples must be 1024..1048576")
    if type(reference_cases) is not int or reference_cases < 0:
        raise ValueError("reference_cases must be nonnegative")
    if type(staging_megabytes) is not int or not 64 <= staging_megabytes <= 4096:
        raise ValueError("staging_megabytes must be 64..4096")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be 1..256")

    rng = np.random.default_rng(config.get("waveform_seed", 2026))
    x = ((2 * rng.integers(0, 2, n) - 1)
         + 1j * (2 * rng.integers(0, 2, n) - 1)) / math.sqrt(2)
    cases = [
        {
            "profile": profile, "snr_db": snr,
            "seed": (seed + repetition * 104729) % 2**32,
        }
        for repetition in range(repetitions)
        for profile in config["profiles"]
        for snr in config["snr_db"]
        for seed in config["seeds"]
    ]
    if len(cases) > 10000:
        raise ValueError("maximum 10000 cases per benchmark")
    # Worst-case ~4 temporary complex arrays per case; keeps CPU staging
    # bounded while leaving room for FFT workspace and GPU allocator overhead.
    by_memory = max(1, staging_megabytes * 1024 * 1024 // (n * 16 * 4))
    effective_batch_size = min(batch_size, by_memory)

    iterator = iter_iq_batches(
        x, config["sample_rate"], cases,
        backend=backend, device=device, dtype=dtype,
        batch_size=effective_batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    elapsed = 0.0
    max_error = 0.0
    max_relative_rmse = 0.0
    comparisons = 0
    offset = 0
    while True:
        start = time.perf_counter()
        try:
            group, values = next(iterator)
        except StopIteration:
            break
        elapsed += time.perf_counter() - start
        for case, y in zip(group, values):
            power = float(np.mean(np.abs(y)**2))
            finite = bool(np.isfinite(y).all()) and math.isfinite(power)
            max_abs = None
            relative_rmse = None
            matches_reference = None
            if offset < reference_cases:
                ref = apply_iq(x, config["sample_rate"], **case)
                difference = y.astype(np.complex128) - ref
                max_abs = float(np.max(np.abs(difference)))
                relative_rmse = float(
                    np.sqrt(np.mean(np.abs(difference)**2)) /
                    max(float(np.sqrt(np.mean(np.abs(ref)**2))), 1e-12)
                )
                tolerance = 1e-7 if dtype == "complex128" else 8e-4
                matches_reference = bool(np.allclose(y, ref, rtol=tolerance, atol=tolerance))
                max_error = max(max_error, max_abs)
                max_relative_rmse = max(max_relative_rmse, relative_rmse)
                comparisons += 1
            rows.append({
                "profile": case["profile"],
                "target_snr_db": case["snr_db"],
                "seed": case["seed"],
                "output_power": power if finite else None,
                "finite": finite,
                "max_absolute_error": max_abs,
                "relative_rmse": relative_rmse,
                "matches_numpy": matches_reference,
            })
            offset += 1

    failures = [r for r in rows if not r["finite"] or r["matches_numpy"] is False]
    status = (
        "NEGATIVE_FINDING" if failures
        else "UNVERIFIED" if backend == "torch" and comparisons == 0
        else "PASS"
    )
    metrics = {
        "schema": 1,
        "suite": "DORC RFSim optional batched IQ backend benchmark",
        "status": status,
        "backend": backend,
        "device": device or "cpu",
        "dtype": dtype if backend == "torch" else "complex128",
        "torch_version": None,
        "gpu_name": None,
        "numpy_version": np.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "waveform_sha256": hashlib.sha256(np.asarray(x, dtype=np.complex128).tobytes()).hexdigest(),
        "samples_per_case": n,
        "total_cases": len(cases),
        "reference_cases": comparisons,
        "failed_cases": len(failures),
        "effective_batch_size": effective_batch_size,
        "elapsed_backend_seconds": elapsed,
        "cases_per_second": len(cases) / elapsed if elapsed > 0 else None,
        "million_complex_samples_per_second": len(cases) * n / elapsed / 1e6 if elapsed > 0 else None,
        "max_absolute_error": max_error if comparisons else None,
        "max_relative_rmse": max_relative_rmse if comparisons else None,
        "cases": rows,
        "interpretation": (
            "End-to-end throughput includes CPU-generated stochastic channel "
            "coefficients, host-to-device transfers, device arithmetic, and "
            "device-to-host output transfers. CUDA and ROCm both use torch "
            "device='cuda'. This is not decoder BER/PER or a physical RF test."
        ),
    }
    if backend == "torch":
        import torch
        metrics["torch_version"] = torch.__version__
        if device == "cuda":
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# DORC RFSim IQ backend throughput benchmark", "",
        f"**Status:** {status}",
        f"**Backend/device:** {backend}/{metrics['device']}",
        f"**Cases:** {len(cases)}; **samples per case:** {n}",
        f"**Wall-clock channel time:** {elapsed:.3f} s",
        f"**Throughput:** {metrics['million_complex_samples_per_second']:.3f} million complex samples/s",
        f"**Batch size:** {effective_batch_size}; **reference checks:** {comparisons}",
        f"**Numerical differences:** max abs {max_error:.4g}; relative RMSE {max_relative_rmse:.4g}",
        f"**NumPy:** {np.__version__}; **PyTorch:** {metrics['torch_version'] or 'not used'}",
        f"**GPU:** {metrics['gpu_name'] or 'none'}", "",
        "All metrics and case-level numerical comparisons are in metrics.json and cases.csv.",
        "",
        "**Interpretation:** " + metrics["interpretation"],
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=os.environ.get("EXPERIMENT_OUTPUT"))
    p.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    p.add_argument("--samples", type=int)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--reference-cases", type=int, default=12)
    p.add_argument("--staging-megabytes", type=int, default=256)
    args = p.parse_args()
    if args.output is None:
        p.error("--output or EXPERIMENT_OUTPUT is required")
    result = benchmark(
        args.config, args.output,
        backend=args.backend, device=args.device,
        dtype=args.dtype, samples=args.samples,
        repetitions=args.repetitions, batch_size=args.batch_size,
        reference_cases=args.reference_cases,
        staging_megabytes=args.staging_megabytes,
    )
    print(result["status"], result["total_cases"], "cases",
          f"{result['million_complex_samples_per_second']:.3f} Msamples/s")


if __name__ == "__main__":
    main()
