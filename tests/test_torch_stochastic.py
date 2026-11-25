"""Statistical and reproducibility checks for device-native Torch RF randomness."""
import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from rfsim import apply_iq
from rfsim.batch import apply_iq_batch


@skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
class TorchStochasticTests(TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        cls.x = (
            (2 * rng.integers(0, 2, 8192) - 1)
            + 1j * (2 * rng.integers(0, 2, 8192) - 1)
        ) / np.sqrt(2)

    def native(self, cases, *, x=None, batch_size=4):
        return apply_iq_batch(
            self.x if x is None else x, 12000, cases,
            backend="torch", device="cpu", rng="torch",
            batch_size=batch_size,
        )

    def test_seeds_repeat_and_batch_grouping_do_not_change_random_streams(self):
        cases = [
            {"profile": p, "snr_db": 10, "seed": seed}
            for p in ("awgn", "mpp", "vhf-mobile")
            for seed in (11, 22)
        ]
        a = self.native(cases, batch_size=1)
        b = self.native(cases, batch_size=3)
        np.testing.assert_array_equal(a, b)
        reordered = [cases[i] for i in (5, 2, 1, 0, 3, 4)]
        c = self.native(reordered, batch_size=4)
        for index, case in enumerate(reordered):
            np.testing.assert_array_equal(c[index], a[cases.index(case)])
        self.assertFalse(np.array_equal(a[0], a[1]))

    def test_independent_of_global_rng_and_no_cpu_random_arrays(self):
        import torch
        cases = [{"profile": "mpp", "snr_db": 8, "seed": 7}]
        torch.manual_seed(88)
        baseline = self.native(cases)
        torch.manual_seed(99)
        with patch("numpy.random.default_rng", side_effect=AssertionError("host RNG used")):
            repeated = self.native(cases)
        np.testing.assert_array_equal(baseline, repeated)

    def test_awgn_empirical_snr_matches_target(self):
        rng = np.random.default_rng(120)
        n = 131072
        x = ((2 * rng.integers(0, 2, n) - 1)
             + 1j * (2 * rng.integers(0, 2, n) - 1)) / np.sqrt(2)
        cases = [
            {"profile": "awgn", "snr_db": snr, "seed": seed}
            for snr in (0, 10, 20) for seed in (3, 23)
        ]
        y = self.native(cases, x=x, batch_size=3)
        input_power = float(np.mean(np.abs(x) ** 2))
        for case, got in zip(cases, y):
            noise_power = float(np.mean(np.abs(got - x) ** 2))
            target = input_power * 10 ** (-case["snr_db"] / 10)
            error_db = 10 * np.log10(noise_power / target)
            self.assertLess(abs(error_db), 0.35)

    def test_fading_population_power_is_statistically_compatible(self):
        x = np.ones(32768, dtype=np.complex128)
        for profile in ("mpp", "nvis-d", "vhf-los", "vhf-mobile", "uhf-mobile"):
            cases = [{"profile": profile, "seed": seed} for seed in range(11, 19)]
            native = self.native(cases, x=x, batch_size=2)
            cpu = np.stack([apply_iq(x, 12000, **case) for case in cases])
            cpu_power = float(np.mean(np.abs(cpu) ** 2))
            native_power = float(np.mean(np.abs(native) ** 2))
            ratio = native_power / cpu_power
            self.assertGreater(ratio, 0.4, (profile, ratio))
            self.assertLess(ratio, 2.5, (profile, ratio))
            self.assertGreater(float(np.std(np.abs(native))), 0.025)

    def test_validation_of_mode_and_input(self):
        with self.assertRaisesRegex(ValueError, "rng"):
            apply_iq_batch(self.x, 12000, [{"profile": "awgn"}],
                           backend="torch", device="cpu", rng="invalid")
        with self.assertRaisesRegex(ValueError, "requires"):
            apply_iq_batch(self.x, 12000, [{"profile": "awgn"}],
                           backend="numpy", rng="torch")


if __name__ == "__main__":
    import unittest
    unittest.main()
