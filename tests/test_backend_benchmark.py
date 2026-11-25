"""Tests for the opt-in DORC numerical backend performance experiment."""
import json
from pathlib import Path
import tempfile
import unittest

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
