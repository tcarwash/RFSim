"""Tests for the shared, pinned RFSim DORC channel benchmark."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dorc_channel_suite import execute, read_config


class DORCChannelSuite(unittest.TestCase):
    def config(self, directory):
        cfg = {
            "schema": 1,
            "name": "DORC common channel suite",
            "sample_rate": 12000,
            "samples": 4096,
            "waveform_seed": 11,
            "profiles": ["awgn", "mpp", "vhf-mobile"],
            "snr_db": [0, 10],
            "seeds": [17, 23],
            "awgn_snr_tolerance_db": 1.25,
        }
        path = Path(directory) / "suite.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_common_suite_passes_and_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            first = execute(config, Path(folder) / "first")
            second = execute(config, Path(folder) / "second")
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(first["cases_total"], 12)
            self.assertEqual(first["cases"], second["cases"])
            self.assertEqual(first["waveform_sha256"], second["waveform_sha256"])
            self.assertIn("HF/VHF/UHF", (Path(folder) / "first" / "report.md").read_text())
            self.assertEqual(
                len((Path(folder) / "first" / "cases.csv").read_text().splitlines()), 13
            )

    def test_negative_finding_has_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            doc = json.loads(config.read_text())
            doc["awgn_snr_tolerance_db"] = 0
            config.write_text(json.dumps(doc))
            result = execute(config, Path(folder) / "out")
            self.assertEqual(result["status"], "NEGATIVE_FINDING")
            self.assertGreater(result["cases_negative"], 0)
            self.assertTrue((Path(folder) / "out" / "report.md").exists())
            self.assertTrue((Path(folder) / "out" / "metrics.json").exists())

    def test_rejects_invalid_config(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            doc = json.loads(config.read_text())
            doc["profiles"] = ["../../unapproved"]
            config.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                read_config(config)


if __name__ == "__main__":
    unittest.main()
