"""Tests for the opt-in DORC numerical backend performance experiment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from experiments import dorc_backend_benchmark
from experiments.dorc_backend_benchmark import benchmark


class BackendBenchmark(unittest.TestCase):
    def config(self, base: Path):
        data = {
            "schema": 1, "name": "DORC benchmark sample",
            "sample_rate": 12000, "samples": 4096, "waveform_seed": 17,
            "profiles": ["awgn", "mpp"],
            "snr_db": [5, 15], "seeds": [3, 9],
        }
        path = base / "config.json"
        path.write_text(json.dumps(data))
        return path

    def test_cpu_benchmark_and_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = self.config(root)
            data = benchmark(
                config, root / "result", backend="numpy",
                repetitions=2, reference_cases=16, batch_size=4,
            )
            self.assertEqual(data["status"], "PASS")
            self.assertEqual(data["total_cases"], 16)
            self.assertEqual(data["reference_cases"], 16)
            self.assertEqual(data["max_absolute_error"], 0)
            self.assertGreater(data["cases_per_second"], 0)
            self.assertEqual(data["failed_cases"], 0)
            self.assertEqual(len((root / "result/cases.csv").read_text().splitlines()), 17)
            self.assertIn("throughput", (root / "result/report.md").read_text().lower())

    def test_nonfinite_gpu_output_is_a_reported_negative_finding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = self.config(root)

            def faulty_batch(*args, **kwargs):
                # Deliberately inject an invalid channel output, not an
                # infrastructure exception, into one simulated case.
                yield ([{"profile": "awgn", "seed": 3, "snr_db": 5}],
                       np.full((1, 4096), np.nan + 1j * np.inf))

            with patch.object(dorc_backend_benchmark, "iter_iq_batches", faulty_batch):
                result = benchmark(config, root / "result", reference_cases=1)
            self.assertEqual(result["status"], "NEGATIVE_FINDING")
            self.assertEqual(result["failed_cases"], 1)
            self.assertEqual(result["cases"][0]["finite"], False)
            self.assertEqual(result["cases"][0]["matches_numpy"], False)
            self.assertIsNone(result["cases"][0]["output_power"])
            self.assertIsNone(result["max_absolute_error"])
            self.assertEqual(
                json.loads((root / "result/metrics.json").read_text())["status"],
                "NEGATIVE_FINDING",
            )
            self.assertIn("NEGATIVE_FINDING", (root / "result/report.md").read_text())

    def test_overflowed_gpu_power_is_a_reported_negative_finding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = self.config(root)

            def enormous_batch(*args, **kwargs):
                yield ([{"profile": "awgn", "seed": 3, "snr_db": 5}],
                       np.full((1, 4096), 1e200 + 1e200j))

            with warnings.catch_warnings(), patch.object(
                dorc_backend_benchmark, "iter_iq_batches", enormous_batch
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                result = benchmark(config, root / "result", reference_cases=1)
            self.assertEqual(result["status"], "NEGATIVE_FINDING")
            self.assertIsNone(result["cases"][0]["output_power"])
            self.assertIsNone(result["cases"][0]["max_absolute_error"])
            self.assertIsNone(result["max_relative_rmse"])
            json.loads((root / "result/metrics.json").read_text())

    def test_device_native_mode_uses_statistical_checks_not_exact_numpy(self):
        import importlib.util
        if not importlib.util.find_spec("torch"):
            self.skipTest("optional PyTorch unavailable")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = self.config(root)
            data = benchmark(
                config, root / "native", backend="torch", device="cpu",
                rng="torch", reference_cases=8, batch_size=4,
            )
            self.assertEqual(data["status"], "PASS")
            self.assertEqual(data["rng"], "torch")
            self.assertEqual(data["validation_mode"], "statistical")
            self.assertEqual(data["reference_cases"], 8)
            self.assertTrue(all(
                row["statistically_valid"] for row in data["cases"]
            ))
            self.assertTrue(all(
                row["matches_numpy"] is None for row in data["cases"]
            ))
            self.assertTrue(
                any(row["validation_kind"] == "awgn_noise_power"
                    for row in data["cases"])
            )
            self.assertIsNone(data["max_absolute_error"])
            self.assertEqual(
                json.loads((root/"native/metrics.json").read_text())["status"], "PASS"
            )

    def test_bounds(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = self.config(root)
            for kwargs in (
                {"repetitions": 0},
                {"samples": 1023},
                {"batch_size": 0},
                {"staging_megabytes": 1},
                {"reference_cases": -1},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    benchmark(config, root / "out", **kwargs)


if __name__ == "__main__":
    unittest.main()
