"""Guard paired RFSim DORC benchmark publication and scientific conclusions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import dorc_cpu_gpu_compare


class PairedBench(unittest.TestCase):
    def test_publishes_flat_files_and_marks_gpu_slowdown(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "suite.json"
            config.write_text('{"schema":1}')
            out = root / "out"
            calls = []

            def fake_bench(path, dest, **kwargs):
                backend = kwargs["backend"]
                calls.append(kwargs)
                dest.mkdir(parents=True, exist_ok=True)
                (dest/"metrics.json").write_text("{}")
                (dest/"cases.csv").write_text("case\n1\n")
                return {
                    "status":"PASS", "config_sha256":"c"*64,
                    "waveform_sha256":"d"*64,"total_cases":5,
                    "cases":[{"profile":"awgn","target_snr_db":10,"seed":1}],
                    "elapsed_backend_seconds":1.0 if backend=="numpy" else 2.0,
                    "million_complex_samples_per_second":5 if backend=="numpy" else 2.5,
                    "gpu_name":None if backend=="numpy" else "Example GPU",
                    "torch_version":None if backend=="numpy" else "2.5.1",
                    "numpy_version":"2.1.3", "samples_per_case":131072,
                    "reference_cases":0 if backend=="numpy" else 3,
                    "failed_cases":0, "max_absolute_error":0.00001,
                    "max_relative_rmse":0.0001,
                }
            with patch.object(dorc_cpu_gpu_compare,"benchmark",fake_bench):
                result = dorc_cpu_gpu_compare.compare(config,out,reference_cases=3)
            self.assertEqual(len(calls),2)
            self.assertEqual(result["status"],"PASS")
            self.assertAlmostEqual(result["gpu_speedup_vs_cpu"],0.5)
            self.assertEqual(result["gpu_name"],"Example GPU")
            self.assertEqual(
                {p.name for p in out.iterdir()},
                {"cpu_cases.csv","cpu_metrics.json","gpu_cases.csv","gpu_metrics.json",
                 "metrics.json","report.md"},
            )
            self.assertIn("0.500×", (out/"report.md").read_text())
            json.loads((out/"metrics.json").read_text())

    def test_requires_reference_checks_and_device(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError, "reference"):
                dorc_cpu_gpu_compare.compare(root/"cfg",root/"out",reference_cases=0)
            with self.assertRaisesRegex(ValueError, "device"):
                dorc_cpu_gpu_compare.compare(root/"cfg",root/"out",device="cpu")


if __name__=="__main__":
    unittest.main()
