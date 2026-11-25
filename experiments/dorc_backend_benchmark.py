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
    rng: str = "numpy",
    awgn_snr_tolerance_db: float = 1.25,
):
    config = read_config(config_path)
    if rng not in ("numpy", "torch"):
        raise ValueError("rng must be numpy or torch")
    if rng == "torch" and backend != "torch":
        raise ValueError("device-native rng requires the torch backend")
    if not math.isfinite(awgn_snr_tolerance_db) or not 0 < awgn_snr_tolerance_db <= 10:
        raise ValueError("AWGN SNR tolerance must be in (0, 10] dB")
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

    waveform_rng = np.random.default_rng(config.get("waveform_seed", 2026))
    x = ((2 * waveform_rng.integers(0, 2, n) - 1)
         + 1j * (2 * waveform_rng.integers(0, 2, n) - 1)) / math.sqrt(2)
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
        batch_size=effective_batch_size, rng=rng,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    elapsed = 0.0
    max_error = 0.0
    max_relative_rmse = 0.0
    comparisons = 0
    comparable_errors = 0
    offset = 0
    while True:
        start = time.perf_counter()
        try:
            group, values = next(iterator)
        except StopIteration:
            break
        elapsed += time.perf_counter() - start
        if values.shape != (len(group), n):
            raise ValueError(
                "backend returned wrong IQ dimensions: "
                f"{values.shape} expected {(len(group), n)}"
            )
        for case, y in zip(group, values):
            finite = bool(np.isfinite(y).all())
            # Invalid sample values (or overflowed powers) are scientific
            # negative findings. They must not corrupt JSON/CSV publication.
            power = float(np.mean(np.abs(y)**2)) if finite else None
            finite = finite and math.isfinite(power)
            if not finite:
                power = None
            max_abs = None
            relative_rmse = None
            matches_reference = None
            validation_kind = None
            awgn_snr_error_db = None
            if offset < reference_cases and rng == "torch":
                comparisons += 1
                # Independent device-native random streams cannot be
                # samplewise compared with NumPy. Validate the actual AWGN
                # variance for AWGN, and bound physically plausible output
                # power for fading channels. The paired CPU/GPU experiment
                # additionally validates *profile-group* power statistics
                # against the independently generated CPU channel.
                if finite:
                    input_power = float(np.mean(np.abs(x) ** 2))
                    if case["profile"] == "awgn":
                        validation_kind = "awgn_noise_power"
                        target_snr = case.get("snr_db")
                        expected_variance = (
                            input_power * 10 ** (-target_snr / 10)
                            if target_snr is not None else 0.0
                        )
                        measured_variance = float(np.mean(np.abs(y - x) ** 2))
                        if expected_variance > 0 and measured_variance > 0:
                            awgn_snr_error_db = 10 * math.log10(
                                measured_variance / expected_variance
                            )
                            matches_reference = bool(
                                math.isfinite(awgn_snr_error_db)
                                and abs(awgn_snr_error_db) <= awgn_snr_tolerance_db
                            )
                        elif expected_variance == 0:
                            matches_reference = measured_variance <= 1e-10
                        else:
                            matches_reference = False
                    else:
                        validation_kind = "fading_power_sanity"
                        snr = case.get("snr_db")
                        nominal = 1 + (10 ** (-snr / 10) if snr is not None else 0)
                        ratio = power / max(input_power, 1e-12)
                        matches_reference = bool(
                            0.03 <= ratio <= 10 * nominal
                        )
                else:
                    validation_kind = "invalid_output"
                    matches_reference = False
            elif offset < reference_cases:
                comparisons += 1
                if finite:
                    ref = apply_iq(x, config["sample_rate"], **case)
                    difference = y.astype(np.complex128) - ref
                    max_abs = float(np.max(np.abs(difference)))
                    relative_rmse = float(
                        np.sqrt(np.mean(np.abs(difference)**2)) /
                        max(float(np.sqrt(np.mean(np.abs(ref)**2))), 1e-12)
                    )
                    if not (math.isfinite(max_abs) and math.isfinite(relative_rmse)):
                        max_abs = None
                        relative_rmse = None
                # Even without a comparable error metric, a nonfinite
                # output has an unequivocally failed reference check.
                if max_abs is None or relative_rmse is None:
                    matches_reference = False
                else:
                    tolerance = 1e-7 if dtype == "complex128" else 8e-4
                    matches_reference = bool(
                        np.allclose(y, ref, rtol=tolerance, atol=tolerance)
                    )
                    max_error = max(max_error, max_abs)
                    max_relative_rmse = max(max_relative_rmse, relative_rmse)
                    comparable_errors += 1
            rows.append({
                "profile": case["profile"],
                "target_snr_db": case["snr_db"],
                "seed": case["seed"],
                "output_power": power if finite else None,
                "finite": finite,
                "max_absolute_error": max_abs,
                "relative_rmse": relative_rmse,
                "matches_numpy": matches_reference if rng == "numpy" else None,
                "statistically_valid": matches_reference if rng == "torch" else None,
                "validation_kind": validation_kind,
                "awgn_snr_error_db": awgn_snr_error_db,
            })
            offset += 1

    failures = [
        r for r in rows
        if not r["finite"]
        or r["matches_numpy"] is False
        or r["statistically_valid"] is False
    ]
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
        "rng": rng,
        "validation_mode": "statistical" if rng == "torch" else "samplewise",
        "awgn_snr_tolerance_db": awgn_snr_tolerance_db if rng == "torch" else None,
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
        "max_absolute_error": max_error if comparable_errors else None,
        "max_relative_rmse": max_relative_rmse if comparable_errors else None,
        "cases": rows,
        "interpretation": (
            ("GPU-native fading/AWGN uses independent per-case on-device Torch RNG streams. "
             "Random sequences differ from NumPy; validation checks empirical AWGN noise "
             "power, finite faded-channel power, and the paired benchmark's aggregate "
             "power statistics. " if rng == "torch" else
             "Seeded CPU-generated stochastic coefficients permit samplewise NumPy checks. ")
            + "End-to-end timing includes all channel generation, transfers and GPU arithmetic; "
            "CUDA and ROCm both use torch device='cuda'. This is not BER/PER or a physical RF test."
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
        f"**Backend/device:** {backend}/{metrics['device']} (rng={rng}, validation={metrics['validation_mode']})",
        f"**Cases:** {len(cases)}; **samples per case:** {n}",
        f"**Wall-clock channel time:** {elapsed:.3f} s",
        f"**Throughput:** {metrics['million_complex_samples_per_second']:.3f} million complex samples/s",
        f"**Batch size:** {effective_batch_size}; **reference checks:** {comparisons}",
        f"**Numerical differences:** max abs "
        f"{f'{max_error:.4g}' if comparable_errors else 'n/a'}; relative RMSE "
        f"{f'{max_relative_rmse:.4g}' if comparable_errors else 'n/a'}",
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
    p.add_argument("--rng", choices=("numpy", "torch"), default="numpy")
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
        staging_megabytes=args.staging_megabytes, rng=args.rng,
    )
    print(result["status"], result["total_cases"], "cases",
          f"{result['million_complex_samples_per_second']:.3f} Msamples/s")


if __name__ == "__main__":
    main()
