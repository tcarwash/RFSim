import unittest
import numpy as np
from rfsim import apply_iq, apply_audio, Tap, load_profile
class Core(unittest.TestCase):
    def test_identity(self):
        x=np.arange(100)+1j
        np.testing.assert_array_equal(apply_iq(x,12000),x)
    def test_noise(self):
        x=np.ones(100000)
        y=apply_iq(x,12000,snr_db=10,noise_bw_hz=6000)
        self.assertAlmostEqual(np.mean(abs(y-x)**2),.2,delta=.005)
    def test_delay(self):
        x=np.r_[1.,np.zeros(100)]
        y=apply_iq(x,1000,(Tap(.003),))
        self.assertEqual(np.argmax(abs(y)),3)
    def test_seed(self):
        x=np.ones(1000)
        np.testing.assert_array_equal(apply_iq(x,12000,"nvis-d",seed=3),apply_iq(x,12000,"nvis-d",seed=3))
    def test_audio(self):
        x=np.sin(np.arange(12000)*2*np.pi*1000/12000)
        np.testing.assert_allclose(apply_audio(x,12000),x,atol=1e-12)
        self.assertEqual(len(apply_audio(x,12000,mode="fm")),len(x))
    def test_custom(self):
        self.assertEqual(len(load_profile("profiles/example-measured.json")),2)
if __name__ == "__main__": unittest.main()
