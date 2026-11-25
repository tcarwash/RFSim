"""Compare RFSim CPU and optional GPU on the same deterministic DORC workload.

Writes flat output files for the GitHub-backed experiment runner:
report.md, metrics.json, cpu_metrics.json, gpu_metrics.json,
cpu_cases.csv, gpu_cases.csv. A statistically/numerically negative
result is publishable; execution/configuration errors raise.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from experiments.dorc_backend_benchmark import benchmark


def compare(config: Path, output: Path, *, samples: int = 131072,
            repetitions: int = 1, batch_size: int = 4,
            reference_cases: int = 12, dtype: str = "complex64",
            device: str = "cuda"):
    if not isinstance(output, Path):
        output = Path(output)
    if device != "cuda":
        raise ValueError("GPU comparison requires an explicit CUDA/ROCm torch device")
    if reference_cases < 1:
        raise ValueError("GPU comparison requires at least one NumPy reference check")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rfsim-cpu-gpu-") as temp:
        temp_path = Path(temp)
        cpu = benchmark(
            config, temp_path / "cpu", backend="numpy",
            samples=samples, repetitions=repetitions, batch_size=batch_size,
            reference_cases=0,
        )
        gpu = benchmark(
            config, temp_path / "gpu", backend="torch", device=device,
            dtype=dtype, samples=samples, repetitions=repetitions,
            batch_size=batch_size, reference_cases=reference_cases,
        )
        for label in ("cpu", "gpu"):
            for basename in ("metrics.json", "cases.csv"):
                shutil.copyfile(
                    temp_path / label / basename, output / f"{label}_{basename}"
                )
    if (cpu["config_sha256"], cpu["waveform_sha256"], cpu["total_cases"]) != (
        gpu["config_sha256"], gpu["waveform_sha256"], gpu["total_cases"]
    ):
        raise RuntimeError("CPU and GPU used different benchmark inputs")
    if [ (c["profile"],c["target_snr_db"],c["seed"]) for c in cpu["cases"] ] != [
        (c["profile"],c["target_snr_db"],c["seed"]) for c in gpu["cases"]
    ]:
        raise RuntimeError("CPU and GPU case ordering differs")
    status = (
        "NEGATIVE_FINDING"
        if "NEGATIVE_FINDING" in (cpu["status"], gpu["status"])
        else "PASS" if (cpu["status"], gpu["status"]) == ("PASS","PASS")
        else "UNVERIFIED"
    )
    speedup = cpu["elapsed_backend_seconds"] / gpu["elapsed_backend_seconds"]
    result = {
        "schema": 1,
        "suite": "DORC RFSim paired CPU/GPU benchmark",
        "status": status,
        "cpu_status": cpu["status"],
        "gpu_status": gpu["status"],
        "cpu_seconds": cpu["elapsed_backend_seconds"],
        "gpu_seconds": gpu["elapsed_backend_seconds"],
        "gpu_speedup_vs_cpu": speedup,
        "cpu_million_complex_samples_per_second": cpu["million_complex_samples_per_second"],
        "gpu_million_complex_samples_per_second": gpu["million_complex_samples_per_second"],
        "gpu_name": gpu["gpu_name"],
        "torch_version": gpu["torch_version"],
        "numpy_version": gpu["numpy_version"],
        "total_cases": gpu["total_cases"],
        "samples_per_case": gpu["samples_per_case"],
        "reference_cases": gpu["reference_cases"],
        "failed_cases": gpu["failed_cases"],
        "max_absolute_error": gpu["max_absolute_error"],
        "max_relative_rmse": gpu["max_relative_rmse"],
        "config_sha256": gpu["config_sha256"],
        "waveform_sha256": gpu["waveform_sha256"],
        "limitations": (
            "End-to-end CPU-generated fading/noise plus GPU tensor math and transfers; "
            "GPU speedup is not assumed, and no modem BER/PER or RF measurements are implied."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output / "report.md").write_text("\n".join([
        "# DORC RFSim CPU vs GPU channel benchmark",
        "",
        f"**Status:** {status}",
        f"**GPU:** {gpu['gpu_name']}",
        f"**Workload:** {gpu['total_cases']} cases × {gpu['samples_per_case']} complex IQ samples",
        f"**Numerical reference checks:** {gpu['reference_cases']} cases, "
        f"{gpu['failed_cases']} failed cases",
        f"**CPU elapsed:** {cpu['elapsed_backend_seconds']:.3f} s",
        f"**GPU elapsed:** {gpu['elapsed_backend_seconds']:.3f} s",
        f"**GPU speedup:** {speedup:.3f}× (less than 1× means GPU is slower)",
        f"**CPU throughput:** {cpu['million_complex_samples_per_second']:.3f} million IQ samples/s",
        f"**GPU throughput:** {gpu['million_complex_samples_per_second']:.3f} million IQ samples/s",
        f"**Maximum numerical difference:** {gpu['max_absolute_error']:.4g}" if
            gpu['max_absolute_error'] is not None else "**Maximum numerical difference:** n/a",
        "",
        "See metrics.json, cpu_metrics.json, gpu_metrics.json and both CSV files "
        "for the case-level comparisons, exact version/device data and provenance.",
        "",
        "**Limitations:** " + result["limitations"],
        "",
    ])+"\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", type=Path, default=os.environ.get("EXPERIMENT_OUTPUT"))
    p.add_argument("--samples", type=int, default=131072)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--reference-cases", type=int, default=12)
    p.add_argument("--dtype", choices=("complex64","complex128"), default="complex64")
    a = p.parse_args()
    if a.output is None:
        p.error("specify --output or EXPERIMENT_OUTPUT")
    r = compare(a.config,a.output,samples=a.samples,repetitions=a.repetitions,
                batch_size=a.batch_size,reference_cases=a.reference_cases,dtype=a.dtype)
    print(r["status"],r["total_cases"],"cases",f"{r['gpu_speedup_vs_cpu']:.3f}x")


if __name__ == "__main__":
    main()
