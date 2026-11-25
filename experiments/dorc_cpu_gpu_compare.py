"""Compare RFSim CPU and optional GPU on the same deterministic DORC workload.

Writes flat output files for the GitHub-backed experiment runner:
report.md, metrics.json, cpu_metrics.json, gpu_metrics.json,
cpu_cases.csv, gpu_cases.csv. A statistically/numerically negative
result is publishable; execution/configuration errors raise.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from experiments.dorc_backend_benchmark import benchmark


def compare(config: Path, output: Path, *, samples: int = 131072,
            repetitions: int = 1, batch_size: int = 4,
            reference_cases: int = 12, dtype: str = "complex64",
            device: str = "cuda", rng: str = "numpy",
            gpu_warmups: int = 0, gpu_trials: int = 1):
    if not isinstance(output, Path):
        output = Path(output)
    if device != "cuda":
        raise ValueError("GPU comparison requires an explicit CUDA/ROCm torch device")
    if reference_cases < 1:
        raise ValueError("GPU comparison requires at least one reference check")
    if rng not in ("numpy", "torch"):
        raise ValueError("rng must be numpy or torch")
    if type(gpu_warmups) is not int or not 0 <= gpu_warmups <= 3:
        raise ValueError("gpu_warmups must be an integer in [0,3]")
    if type(gpu_trials) is not int or not 1 <= gpu_trials <= 5:
        raise ValueError("gpu_trials must be an integer in [1,5]")
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
            batch_size=batch_size, reference_cases=reference_cases, rng=rng,
        )
        # The first GPU run records cold-start overhead (imports, context and
        # first-use kernel initialization). Optional subsequent runs measure
        # *warmed* throughput, with the same pinned seeds and benchmark cases.
        # Neither metric is silently substituted for the other.
        gpu_runs = [gpu] if gpu_warmups == 0 else []
        for index in range(gpu_warmups + (gpu_trials if gpu_warmups else gpu_trials - 1)):
            warming = index < gpu_warmups
            trial = benchmark(
                config, temp_path / f"gpu_trial_{index}",
                backend="torch", device=device, dtype=dtype,
                samples=samples, repetitions=repetitions,
                batch_size=batch_size,
                reference_cases=0 if warming else reference_cases, rng=rng,
            )
            for key in ("config_sha256", "waveform_sha256", "total_cases"):
                if gpu[key] != trial[key]:
                    raise RuntimeError(f"GPU repeat altered pinned {key}")
            if not warming:
                gpu_runs.append(trial)

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
    profile_checks = []
    if rng == "torch":
        grouped = defaultdict(lambda: {"cpu": [], "gpu": []})
        for cpu_case, gpu_case in zip(cpu["cases"], gpu["cases"]):
            name = cpu_case["profile"]
            grouped[name]["cpu"].append(cpu_case["output_power"])
            grouped[name]["gpu"].append(gpu_case["output_power"])
        for name, values in sorted(grouped.items()):
            # Preserve valid NEGATIVE_FINDING artifacts if an output is NaN or
            # overflows: the component benchmark records power=None.
            valid = all(
                value is not None and math.isfinite(value)
                for kind in ("cpu", "gpu") for value in values[kind]
            )
            cpu_p = sum(values["cpu"]) / len(values["cpu"]) if valid else None
            gpu_p = sum(values["gpu"]) / len(values["gpu"]) if valid else None
            ratio = gpu_p / max(cpu_p, 1e-12) if valid else None
            # Independent channel realizations can vary strongly per seed.
            # Pool profiles over all SNRs/seeds, not waveform-level samples.
            passed = bool(valid and 0.4 <= ratio <= 2.5)
            profile_checks.append({
                "profile": name, "cpu_mean_power": cpu_p,
                "gpu_mean_power": gpu_p, "gpu_cpu_power_ratio": ratio,
                "statistically_compatible": passed,
            })
    has_invalid_profile = any(
        not check["statistically_compatible"] for check in profile_checks
    )
    has_invalid_trial = any(trial["status"] != "PASS" for trial in gpu_runs)
    # For an explicit warmup protocol, only timed *post-warmup* trials
    # contribute to the steady-state result. The cold run remains separate.
    steady_times = [trial["elapsed_backend_seconds"] for trial in gpu_runs]
    steady_median = (
        statistics.median(steady_times)
        if gpu_warmups and steady_times else None
    )
    status = (
        "NEGATIVE_FINDING"
        if "NEGATIVE_FINDING" in (cpu["status"], gpu["status"]) or has_invalid_profile or has_invalid_trial
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
        "gpu_rng": rng,
        "validation_mode": "statistical" if rng == "torch" else "samplewise",
        "profile_power_checks": profile_checks,
        "cpu_seconds": cpu["elapsed_backend_seconds"],
        "gpu_seconds": gpu["elapsed_backend_seconds"],
        "gpu_speedup_vs_cpu": speedup,
        "gpu_warmups": gpu_warmups,
        "gpu_trials": gpu_trials,
        "gpu_measured_trial_seconds": steady_times,
        "gpu_steady_median_seconds": steady_median,
        "gpu_steady_speedup_vs_cpu": (
            cpu["elapsed_backend_seconds"] / steady_median
            if steady_median is not None else None
        ),
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
            ("Torch-native stochastic fading and noise with independent device RNG; "
             if rng == "torch" else "CPU-seeded fading and noise with NumPy streams; ")
            + "device arithmetic and transfers are included. GPU cold-start and warmed "
            "throughput are distinct, and steady-state timings exclude the initial "
            "GPU context/kernel setup. No modem BER/PER or RF measurements are implied."
        ),
    }
    if gpu_warmups or gpu_trials > 1:
        (output / "gpu_runs.json").write_text(json.dumps({
            "schema": 1, "gpu_rng": rng,
            "warmup_runs": gpu_warmups, "measured_runs": gpu_trials,
            "cold_seconds": gpu["elapsed_backend_seconds"],
            "trial_seconds": steady_times,
            "trial_statuses": [trial["status"] for trial in gpu_runs],
        }, indent=2, allow_nan=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output / "report.md").write_text("\n".join([
        "# DORC RFSim CPU vs GPU channel benchmark",
        "",
        f"**Status:** {status}",
        f"**GPU:** {gpu['gpu_name']}; **RNG:** {rng}",
        f"**Validation:** {result['validation_mode']}",
        f"**Workload:** {gpu['total_cases']} cases × {gpu['samples_per_case']} complex IQ samples",
        f"**Numerical reference checks:** {gpu['reference_cases']} cases, "
        f"{gpu['failed_cases']} failed cases",
        f"**CPU elapsed:** {cpu['elapsed_backend_seconds']:.3f} s",
        f"**GPU elapsed:** {gpu['elapsed_backend_seconds']:.3f} s",
        f"**Cold GPU speedup:** {speedup:.3f}× (less than 1× means GPU is slower)",
        *([
            f"**Warm GPU median:** {steady_median:.3f} s over {len(steady_times)} trials "
            f"(after {gpu_warmups} warmup runs)",
            f"**Steady GPU speedup:** {result['gpu_steady_speedup_vs_cpu']:.3f}×"
        ] if steady_median is not None else []),
        f"**CPU throughput:** {cpu['million_complex_samples_per_second']:.3f} million IQ samples/s",
        f"**GPU throughput:** {gpu['million_complex_samples_per_second']:.3f} million IQ samples/s",
        f"**Maximum numerical difference:** {gpu['max_absolute_error']:.4g}" if
            gpu['max_absolute_error'] is not None else
            "**Maximum numerical difference:** n/a (independent RNG streams)",
        *([
            f"- {check['profile']} aggregate GPU/CPU power ratio: "
            + (f"{check['gpu_cpu_power_ratio']:.3f}×"
               if check["gpu_cpu_power_ratio"] is not None else "n/a")
            + f" ({'PASS' if check['statistically_compatible'] else 'FAIL'})"
            for check in profile_checks
        ]),
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
    p.add_argument("--rng", choices=("numpy","torch"), default="numpy")
    p.add_argument("--gpu-warmups", type=int, default=0)
    p.add_argument("--gpu-trials", type=int, default=1)
    a = p.parse_args()
    if a.output is None:
        p.error("specify --output or EXPERIMENT_OUTPUT")
    r = compare(a.config,a.output,samples=a.samples,repetitions=a.repetitions,
                batch_size=a.batch_size,reference_cases=a.reference_cases,dtype=a.dtype,
                rng=a.rng, gpu_warmups=a.gpu_warmups, gpu_trials=a.gpu_trials)
    print(r["status"],r["total_cases"],"cases",f"{r['gpu_speedup_vs_cpu']:.3f}x")


if __name__ == "__main__":
    main()
