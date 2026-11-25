"""Statistical and provenance tests for offline RFSim experiments."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from rfsim import apply_iq
from rfsim.session import ChannelSession
from rfsim.experiments import run_experiment,read_report

class ExperimentTests(unittest.TestCase):
    def test_paired_noise_snr_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            tx=np.ones(20000,dtype=complex)
            np.save(root/"tx.npy",tx)
            manifest={"schema":1,"name":"sweep","signal":{"path":"tx.npy","format":"iq_npy","sample_rate":20000},
                      "channel":{"profile":"awgn","noise_bw_hz":10000},
                      "sweep":{"snr_db":[0,20],"seeds":[4,5]},"output_dir":"results/paired"}
            (root/"study.json").write_text(json.dumps(manifest))
            report=run_experiment(root/"study.json")
            self.assertEqual(len(report["cases"]),4)
            self.assertEqual(read_report(root/"results/paired/report.json")["experiment"],"sweep")
            self.assertEqual(len(report["provenance"]["manifest_sha256"]),64)
            a=np.load(report["cases"][0]["output"])
            b=np.load(report["cases"][2]["output"])
            # Noise draws use the same realization at two SNRs.
            np.testing.assert_allclose((a-tx)/(b-tx),10.0,rtol=1e-12,atol=1e-10)
            self.assertAlmostEqual(float(np.mean(abs(a-tx)**2)),2.,delta=.05)
            self.assertAlmostEqual(float(np.mean(abs(b-tx)**2)),.02,delta=.002)

    def test_empty_or_too_large_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);np.save(p/"tx.npy",np.ones(500,dtype=complex))
            manifest={"schema":1,"signal":{"path":"tx.npy","sample_rate":500},
                      "sweep":{"snr_db":[1]*1001,"seeds":[0]}}
            f=p/"experiment.json";f.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): run_experiment(f)
            manifest["sweep"]={"snr_db":[None],"seeds":[0]}
            f.write_text(json.dumps(manifest))
            r=run_experiment(f)
            self.assertEqual(len(r["cases"]),1)
            self.assertIsNone(r["cases"][0]["metrics"]["raw_nmse_db"])

    def test_zero_noise_seed_reproducible(self):
        x=np.ones(5000)
        a=apply_iq(x,12000,"awgn",snr_db=3,seed=123)
        b=apply_iq(x,12000,"awgn",snr_db=3,seed=123)
        np.testing.assert_array_equal(a,b)

    def test_time_varying_channel_and_carrier_dependence(self):
        a=ChannelSession(1000,"mobile-rural",seed=3,duration_s=10,carrier_hz=146e6)
        b=ChannelSession(1000,"mobile-rural",seed=3,duration_s=10,carrier_hz=440e6)
        t=np.linspace(0,9,1000)
        _,ga=a.impulse_response(t)
        _,gb=b.impulse_response(t)
        self.assertFalse(np.allclose(ga,gb))
        self.assertTrue(np.isfinite(ga).all())
        self.assertTrue(np.isfinite(gb).all())

    def test_burst_noise_is_deterministic(self):
        s=ChannelSession(12000,"fixed",duration_s=2,seed=3)
        x=np.ones(1200,dtype=complex)
        np.testing.assert_array_equal(s.apply(x,start_s=1,snr_db=8,noise_seed=9),
                                      s.apply(x,start_s=1,snr_db=8,noise_seed=9))

if __name__=="__main__": unittest.main()
