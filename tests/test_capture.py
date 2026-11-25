import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rfsim.capture import load_iq_capture
from rfsim.measured import estimate

class CaptureTests(unittest.TestCase):
    def test_sigmf_cf32_and_npy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.default_rng(11)
            x=(rng.normal(size=4096)+1j*rng.normal(size=4096)).astype(np.complex64)
            y=np.convolve(x,np.array([.8+.2j,.1-.1j]))[:len(x)]
            x.tofile(root/"tx.sigmf-data")
            metadata={"global":{"core:datatype":"cf32_le","core:sample_rate":12000},
                      "captures":[{"core:sample_start":0,"core:frequency":146520000}]}
            (root/"tx.sigmf-meta").write_text(json.dumps(metadata))
            np.save(root/"rx.npy",y)
            tx,rate,meta=load_iq_capture(root/"tx.sigmf-meta")
            rx,rrate,_=load_iq_capture(root/"rx.npy",fs=12000)
            self.assertEqual(rate,rrate)
            self.assertEqual(meta["center_hz"],146520000)
            np.testing.assert_allclose(tx,x)
            times,delays,g,stats=estimate(tx,rx,rate,taps=2,window=1024)
            np.testing.assert_allclose(g[0],[.8+.2j,.1-.1j],atol=1e-5)

    def test_reject_unsupported_capture(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            np.save(root/"samples.npy",np.ones(8,complex))
            with self.assertRaises(ValueError):
                load_iq_capture(root/"samples.npy")
            with self.assertRaises(ValueError):
                load_iq_capture(root/"samples.npy",fs=48000,max_samples=4)

if __name__=="__main__":unittest.main()
