"""Exact-wrapper parity with the installed upstream Data2G channel models."""
import unittest
import numpy as np
from rfsim.data2g_bridge import fm_audio,hf_audio
try:
    from data2g import hfchannel, fmchannel, config
except ImportError:
    hfchannel=fmchannel=config=None

@unittest.skipIf(config is None,"Data2G optional dependency not installed")
class Data2GBridgeTests(unittest.TestCase):
    def test_native_hf_output(self):
        fs=config.FS
        t=np.arange(int(fs*.6))/fs
        x=np.sin(2*np.pi*1000*t)
        want=hfchannel.apply_channel(x,snr_db=10,seed=2,
                                      fading_preset=hfchannel.ITU_PRESETS["nvis-d"])
        got=hf_audio(x,fs,preset="nvis-d",snr_db=10,seed=2)
        np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)

    def test_native_fm_output(self):
        fs=config.FS
        t=np.arange(int(fs*.1))/fs
        x=np.sin(2*np.pi*1000*t)
        want=fmchannel.FmChannel(condition="fixed",radio="flat",cnr_db=25,
                                 seed=3,fs=fs,duration_s=5).apply(x)
        got=fm_audio(x,fs,condition="fixed",radio="flat",cnr_db=25,
                     seed=3,duration_s=5)
        np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)

if __name__=="__main__":unittest.main()
