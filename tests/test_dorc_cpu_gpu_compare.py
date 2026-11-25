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

    def test_native_rng_requires_profile_distribution_evidence(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "suite.json"
            config.write_text('{"schema":1}')
            calls = []

            def fake_bench(path, dest, **kwargs):
                calls.append(kwargs)
                dest.mkdir(parents=True, exist_ok=True)
                (dest/"metrics.json").write_text("{}")
                (dest/"cases.csv").write_text("case\n1\n")
                gpu = kwargs["backend"] == "torch"
                power = 0.1 if gpu else 1.0
                return {
                    "status":"PASS", "config_sha256":"c"*64,
                    "waveform_sha256":"d"*64, "total_cases":1,
                    "cases":[{"profile":"mpp","target_snr_db":10,
                              "seed":1, "output_power":power}],
                    "elapsed_backend_seconds":1.0, "million_complex_samples_per_second":1.0,
                    "gpu_name":"Test GPU" if gpu else None,
                    "torch_version":"2.5.1" if gpu else None, "numpy_version":"2.1.3",
                    "samples_per_case":4096, "reference_cases":1 if gpu else 0,
                    "failed_cases":0, "max_absolute_error":None,
                    "max_relative_rmse":None,
                }

            with patch.object(dorc_cpu_gpu_compare, "benchmark", fake_bench):
                result = dorc_cpu_gpu_compare.compare(
                    config, root/"out", rng="torch", reference_cases=1
                )
            self.assertEqual(result["status"], "NEGATIVE_FINDING")
            self.assertEqual(result["validation_mode"], "statistical")
            self.assertEqual(result["profile_power_checks"][0]["statistically_compatible"],False)
            self.assertEqual(calls[1]["rng"], "torch")
            self.assertEqual(calls[0].get("rng","numpy"), "numpy")

    def test_invalid_profile_power_is_a_publishable_negative_finding(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "suite.json"
            config.write_text('{"schema":1}')

            def fake_bench(path, dest, **kwargs):
                gpu = kwargs["backend"] == "torch"
                dest.mkdir(parents=True, exist_ok=True)
                (dest/"metrics.json").write_text("{}")
                (dest/"cases.csv").write_text("case\n1\n")
                return {
                    "status":"NEGATIVE_FINDING" if gpu else "PASS",
                    "config_sha256":"c"*64,"waveform_sha256":"d"*64,
                    "total_cases":1,
                    "cases":[{"profile":"mpp","target_snr_db":10,"seed":1,
                              "output_power":None if gpu else 1.0}],
                    "elapsed_backend_seconds":1.0,
                    "million_complex_samples_per_second":1.0,
                    "gpu_name":"Test GPU" if gpu else None,
                    "torch_version":"2.5.1" if gpu else None,
                    "numpy_version":"2.1.3","samples_per_case":4096,
                    "reference_cases":kwargs["reference_cases"],
                    "failed_cases":1 if gpu else 0,
                    "max_absolute_error":None,"max_relative_rmse":None,
                }

            with patch.object(dorc_cpu_gpu_compare, "benchmark", fake_bench):
                out = root / "out"
                result = dorc_cpu_gpu_compare.compare(
                    config, out, rng="torch", reference_cases=1
                )
            self.assertEqual(result["status"], "NEGATIVE_FINDING")
            self.assertIsNone(result["profile_power_checks"][0]["gpu_cpu_power_ratio"])
            self.assertIn("mpp aggregate GPU/CPU power ratio: n/a (FAIL)",
                          (out/"report.md").read_text())
            json.loads((out/"metrics.json").read_text())

    def test_cold_and_warmed_measurements_are_recorded_separately(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "suite.json"
            config.write_text('{"schema":1}')
            calls = []

            def fake_bench(path, dest, **kwargs):
                calls.append(kwargs)
                index = len(calls)
                dest.mkdir(parents=True, exist_ok=True)
                (dest/"metrics.json").write_text("{}")
                (dest/"cases.csv").write_text("case\n1\n")
                elapsed = {1: 1.0, 2: 4.0, 3: 0.5, 4: 0.1, 5: 0.3, 6: 0.2}[index]
                gpu = kwargs["backend"] == "torch"
                return {
                    "status": "PASS" if kwargs["reference_cases"] else
                        ("UNVERIFIED" if gpu else "PASS"),
                    "config_sha256":"c"*64, "waveform_sha256":"d"*64,
                    "total_cases":1,
                    "cases":[{"profile":"awgn","target_snr_db":10,
                              "seed":1, "output_power":1.0}],
                    "elapsed_backend_seconds":elapsed,
                    "million_complex_samples_per_second": 1.0,
                    "gpu_name":"Example" if gpu else None,
                    "torch_version":"2.5.1" if gpu else None,
                    "numpy_version":"2.1.3", "samples_per_case":4096,
                    "reference_cases":kwargs["reference_cases"],
                    "failed_cases":0,"max_absolute_error":0.0,
                    "max_relative_rmse":0.0,
                }

            with patch.object(dorc_cpu_gpu_compare,"benchmark",fake_bench):
                out = root / "out"
                result = dorc_cpu_gpu_compare.compare(
                    config, out, reference_cases=1,
                    gpu_warmups=1, gpu_trials=3,
                )
            self.assertEqual(len(calls), 6)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["gpu_seconds"], 4.0)
            self.assertEqual(result["gpu_speedup_vs_cpu"], 0.25)
            self.assertEqual(result["gpu_measured_trial_seconds"], [0.1,0.3,0.2])
            self.assertAlmostEqual(result["gpu_steady_median_seconds"], 0.2)
            self.assertAlmostEqual(result["gpu_steady_speedup_vs_cpu"], 5.0)
            self.assertEqual(
                json.loads((out/"gpu_runs.json").read_text())["trial_statuses"],
                ["PASS"] * 3,
            )
            self.assertEqual(len(list(out.iterdir())), 7)

    def test_warmup_bounds(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for params in ({"gpu_warmups":-1},{"gpu_warmups":4},
                           {"gpu_trials":0},{"gpu_trials":6}):
                with self.subTest(params=params), self.assertRaises(ValueError):
                    dorc_cpu_gpu_compare.compare(
                        root/"config", root/"out", **params
                    )

    def test_requires_reference_checks_and_device(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError, "reference"):
                dorc_cpu_gpu_compare.compare(root/"cfg",root/"out",reference_cases=0)
            with self.assertRaisesRegex(ValueError, "device"):
                dorc_cpu_gpu_compare.compare(root/"cfg",root/"out",device="cpu")


if __name__=="__main__":
    unittest.main()
