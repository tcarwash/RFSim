#!/usr/bin/env python3
"""Reusable DORC channel-calibration suite using pinned RFSim source.

Numerical failures are evidence (exit zero); invalid configuration or
infrastructure errors fail the process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from rfsim import PROFILES, apply_iq


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("schema") != 1:
        raise ValueError("DORC suite requires schema 1")
    profiles = config.get("profiles")
    snrs = config.get("snr_db")
    seeds = config.get("seeds")
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 16:
        raise ValueError("invalid profiles list")
    if not all(isinstance(p, str) and p in PROFILES for p in profiles):
        raise ValueError("suite supports only built-in RFSim profiles")
    if len(set(profiles)) != len(profiles):
        raise ValueError("duplicate profiles")
    if not isinstance(snrs, list) or not 1 <= len(snrs) <= 16:
        raise ValueError("invalid SNR list")
    if not all(type(v) in (int, float) and math.isfinite(v) and -40 <= v <= 80 for v in snrs):
        raise ValueError("invalid SNR")
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 16:
        raise ValueError("invalid seed list")
    if not all(type(v) is int and 0 <= v <= 2**32 - 1 for v in seeds):
        raise ValueError("invalid seed")
    if len(profiles) * len(snrs) * len(seeds) > 96:
        raise ValueError("too many cases")
    samples = config.get("samples")
    sample_rate = config.get("sample_rate")
    if type(samples) is not int or not 1024 <= samples <= 262144:
        raise ValueError("invalid sample count")
    if type(sample_rate) not in (int, float) or not math.isfinite(sample_rate) or not 1000 <= sample_rate <= 1000000:
        raise ValueError("invalid sample rate")
    tolerance = config.get("awgn_snr_tolerance_db", 1.25)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 <= tolerance <= 6:
        raise ValueError("invalid AWGN tolerance")
    waveform_seed = config.get("waveform_seed", 2026)
    if type(waveform_seed) is not int or not 0 <= waveform_seed <= 2**32 - 1:
        raise ValueError("invalid waveform seed")
    return config


def execute(config_path: Path, output_dir: Path) -> dict:
    config = read_config(config_path)
    raw = config_path.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.get("waveform_seed", 2026))
    n = config["samples"]
    x = ((2 * rng.integers(0, 2, n) - 1)
         + 1j * (2 * rng.integers(0, 2, n) - 1)) / math.sqrt(2)
    signal_sha = hashlib.sha256(np.asarray(x, dtype=np.complex128).tobytes()).hexdigest()
    tx_power = float(np.mean(np.abs(x) ** 2))
    cases = []
    for profile in config["profiles"]:
        for snr in config["snr_db"]:
            for seed in config["seeds"]:
                kw = dict(snr_db=float(snr), seed=seed)
                y = apply_iq(x, config["sample_rate"], profile, **kw)
                replay = apply_iq(x, config["sample_rate"], profile, **kw)
                deterministic = bool(np.array_equal(y, replay))
                finite = bool(np.isfinite(y).all())
                rx_power = float(np.mean(np.abs(y) ** 2)) if finite else None
                error_power = float(np.mean(np.abs(y - x) ** 2)) if finite else None
                raw_snr = (
                    10 * math.log10(tx_power / error_power)
                    if error_power is not None and error_power > 0 else None
                )
                awgn_ok = (
                    abs(raw_snr - snr) <= config.get("awgn_snr_tolerance_db", 1.25)
                    if profile == "awgn" and raw_snr is not None else None
                )
                valid = finite and deterministic and rx_power is not None and rx_power > 0
                if profile == "awgn":
                    valid = valid and awgn_ok is True
                cases.append(dict(
                    profile=profile, target_snr_db=snr, seed=seed,
                    sample_count=n, deterministic=deterministic, finite=finite,
                    rx_power=rx_power, raw_error_power=error_power,
                    raw_waveform_snr_db=raw_snr, awgn_in_tolerance=awgn_ok,
                    meets_numerical_checks=bool(valid),
                ))
    passed = sum(int(c["meets_numerical_checks"]) for c in cases)
    result = {
        "schema": 1,
        "suite": config.get("name", "DORC common RFSim channel suite"),
        "status": "PASS" if passed == len(cases) else "NEGATIVE_FINDING",
        "cases_total": len(cases), "cases_passed": passed,
        "cases_negative": len(cases) - passed,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "waveform_sha256": signal_sha,
        "numpy_version": np.__version__,
        "sample_rate": config["sample_rate"],
        "samples": n,
        "cases": cases,
        "limitations": (
            "Synthetic numerical checks only; raw waveform SNR is not modem BER/PER, "
            "measured RF path accuracy, or over-the-air receiver sensitivity."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    lines = [
        "# DORC common RFSim channel suite", "",
        f"**Status:** {result['status']}",
        f"**Numerical checks:** {passed}/{len(cases)}",
        f"**Config SHA256:** {result['config_sha256']}",
        f"**Waveform SHA256:** {signal_sha}",
        f"**NumPy:** {np.__version__}",
        "",
        "The same deterministic complex waveform was passed through multiple "
        "HF/VHF/UHF channel profiles at selected SNRs and seeds. Each case "
        "was replayed to verify determinism.", "",
        "| Profile | Target SNR (dB) | Seed | Raw waveform SNR (dB) | Deterministic | Check |",
        "|---|---:|---:|---:|---|---|",
    ]
    for case in cases:
        snr = case["raw_waveform_snr_db"]
        snr_text = "n/a" if snr is None else f"{snr:.2f}"
        lines.append(
            f"| {case['profile']} | {case['target_snr_db']} | {case['seed']} "
            f"| {snr_text} | {'yes' if case['deterministic'] else 'NO'} "
            f"| {'PASS' if case['meets_numerical_checks'] else 'NEGATIVE'} |"
        )
    lines.extend([
        "",
        "**Interpretation:** AWGN raw waveform SNR is checked against the "
        "configured target. For multipath/fading profiles, input-output "
        "mismatch includes channel distortion, so no target-SNR accuracy "
        "claim is made.", "",
        "**Limitations:** " + result["limitations"], "",
        "Full machine-readable evidence: metrics.json and cases.csv.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=os.environ.get("EXPERIMENT_OUTPUT"))
    args = parser.parse_args()
    if args.output is None:
        parser.error("specify --output or EXPERIMENT_OUTPUT")
    result = execute(args.config, args.output)
    print(result["status"], result["cases_passed"], "/", result["cases_total"])


if __name__ == "__main__":
    main()
