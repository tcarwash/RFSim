import unittest
import tempfile
from pathlib import Path
import numpy as np
from rfsim.session import ChannelSession
from rfsim.measured import estimate, save_trace, MeasuredChannel
class Timeline(unittest.TestCase):
    def test_static(self):
        x=np.ones(1000)
        np.testing.assert_array_equal(ChannelSession(1000,"static").apply(x,start_s=5),x)
    def test_random_access(self):
        s=ChannelSession(1000,"mobile-urban",duration_s=20,seed=9)
        _,a=s.impulse_response(np.arange(1000)/1000+5)
        _,b=s.impulse_response(np.arange(2000)/1000+4)
        np.testing.assert_allclose(a,b[1000:])
    def test_outside(self):
        with self.assertRaises(ValueError):
            ChannelSession(1000,duration_s=1).apply(np.ones(1000),start_s=.5)
    def test_estimation(self):
        r=np.random.default_rng(5); x=r.normal(size=8192)+1j*r.normal(size=8192)
        h=np.array([.8+.2j,0,.3-.1j])
        y=np.convolve(x,h)[:len(x)]
        t,d,g,stats=estimate(x,y,1000,taps=3,window=1024)
        np.testing.assert_allclose(g,np.broadcast_to(h,g.shape),atol=2e-6)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"trace.npz"; save_trace(p,t,d,g,{"kind":"synthetic"})
            z=MeasuredChannel(p).apply(x[:1000],1000)
            np.testing.assert_allclose(z,y[:1000],atol=1e-5)
    def test_narrowband_rejected(self):
        with self.assertRaises(ValueError):
            estimate(np.ones(4096),np.ones(4096),1000)
