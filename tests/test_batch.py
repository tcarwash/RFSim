"""Tests for the optional batched CPU/PyTorch IQ acceleration interface."""
import importlib.util
import unittest

import numpy as np

from rfsim import Tap, apply_iq
from rfsim.batch import _torch_batch, apply_iq_batch, iter_iq_batches
from unittest.mock import patch
import rfsim.batch as batch_module


class BatchChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = (
            np.random.default_rng(2033).standard_normal(2048)
            + 1j * np.random.default_rng(2034).standard_normal(2048)
        ) / np.sqrt(2)
        cls.fs = 12000
        cls.cases = [
            {"profile": "awgn", "seed": 9, "snr_db": 7},
            {"profile": "mpp", "seed": 17, "snr_db": 15, "cfo_hz": 21},
            {"profile": "nvis-d", "seed": 23, "snr_db": 8},
            {"profile": "vhf-mobile", "seed": 3, "snr_db": 1, "cfo_hz": 39},
            {"profile": "uhf-mobile", "seed": 5, "snr_db": 12},
            {"profile": "vhf-los", "seed": 12, "noise_bw_hz": 3000, "snr_db": 10},
        ]

    def test_numpy_matches_each_legacy_call(self):
        batches = list(iter_iq_batches(self.x, self.fs, self.cases, batch_size=2))
        self.assertEqual([len(group) for group, _ in batches], [2, 2, 2])
        output = np.concatenate([data for _, data in batches])
        self.assertEqual(output.shape, (len(self.cases), len(self.x)))
        for i, case in enumerate(self.cases):
            np.testing.assert_array_equal(output[i], apply_iq(self.x, self.fs, **case))

    def test_reject_invalid_input_without_torch(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            apply_iq_batch(self.x, self.fs, [])
        with self.assertRaisesRegex(ValueError, "backend"):
            apply_iq_batch(self.x, self.fs, self.cases, backend="unknown")
        with self.assertRaisesRegex(ValueError, "CPU"):
            apply_iq_batch(self.x, self.fs, self.cases, device="cuda")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            apply_iq_batch(self.x, self.fs, self.cases, batch_size=0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            apply_iq_batch(self.x, self.fs, [{"profile": "awgn", "shell": "echo unsafe"}])

    def test_generator_chunks_and_cpu_clock(self):
        cases = ({"profile": "awgn", "seed": i} for i in range(5))
        chunks = list(iter_iq_batches(self.x, self.fs, cases, batch_size=2))
        self.assertEqual([len(a) for a, _ in chunks], [2, 2, 1])
        c = {"profile": "awgn", "seed": 3, "clock_ppm": 20}
        out = apply_iq_batch(self.x, self.fs, [c])
        np.testing.assert_array_equal(out[0], apply_iq(self.x, self.fs, **c))

    def test_staging_limit_rejects_unbounded_gpu_batch(self):
        x = np.zeros(1_000_000, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "staging"):
            apply_iq_batch(
                x, self.fs, [{"profile": "awgn", "seed": 1}],
                backend="torch", device="cpu", batch_size=256,
            )

    def test_large_custom_multipath_profile_fails_before_gpu_allocations(self):
        # A single case with many distinct delays allocates one full
        # coefficient matrix per delay, even if batch_size is only one.
        # This test needs no PyTorch install: the guard precedes GPU setup.
        profile = tuple(Tap(delay_s=i / 12000) for i in range(60))
        x = np.ones(1_000_000, dtype=np.complex128)
        with patch.dict(batch_module.PROFILES, {"many_taps": profile}):
            with self.assertRaisesRegex(ValueError, "tap/channel staging"):
                _torch_batch(
                    x, self.fs, [{"profile": "many_taps", "seed": 0}],
                    device="cuda", dtype="complex64", torch=object(),
                )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
    def test_torch_cpu_matches_reference(self):
        cases = list(self.cases) + [
            {"profile": "awgn", "seed": 30},
            {"profile": "vhf-mobile", "seed": 41, "snr_db": 0, "signal_power": 2.0},
        ]
        for dtype, tol in (("complex128", 5e-8), ("complex64", 6e-4)):
            torch_output = apply_iq_batch(
                self.x, self.fs, cases,
                backend="torch", device="cpu", dtype=dtype, batch_size=3,
            )
            self.assertEqual(torch_output.shape, (len(cases), len(self.x)))
            for index, case in enumerate(cases):
                ref = apply_iq(self.x, self.fs, **case)
                np.testing.assert_allclose(torch_output[index], ref, atol=tol, rtol=tol)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
    def test_torch_rejects_unsupported_features(self):
        with self.assertRaisesRegex(NotImplementedError, "clock"):
            apply_iq_batch(
                self.x, self.fs, [{"profile": "awgn", "clock_ppm": 10}],
                backend="torch", device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            apply_iq_batch(
                self.x, self.fs, [{"profile": "awgn", "docker_args": ["--privileged"]}],
                backend="torch", device="cpu",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
    def test_torch_cuda_when_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("GPU unavailable")
        values = apply_iq_batch(
            self.x, self.fs, self.cases,
            backend="torch", device="cuda", dtype="complex64", batch_size=3,
        )
        for i, case in enumerate(self.cases):
            np.testing.assert_allclose(
                values[i], apply_iq(self.x, self.fs, **case), atol=8e-4, rtol=8e-4,
            )


if __name__ == "__main__":
    unittest.main()
