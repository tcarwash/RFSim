"""Measured-inspired K6 source audio model: distinct from receiver RF noise."""
import unittest
import numpy as np
from rfsim import k6_gp0_pcm, k6_gp0_profile, generate_gp0_noise
from rfsim.k6_mic import (
    FS, MEASURED_HZ, MEASURED_PCM_PEAK_FS, OFF_PCM_RMS_FS,
    RESIDUAL_BAND_EDGES_HZ, RESIDUAL_BAND_FRACTIONS,
)


class K6MicTests(unittest.TestCase):
    def test_report_provenance_and_limitations(self):
        meta = k6_gp0_profile()
        self.assertEqual(meta['sample_rate_hz'], 8000)
        self.assertEqual(meta['physical_tone_points_hz'], [500,1000,2800])
        self.assertEqual(meta['physical_generator_off_pcm_rms_full_scale'], OFF_PCM_RMS_FS)
        self.assertGreaterEqual(len(meta['uncertainty']), 7)
        self.assertAlmostEqual(sum(RESIDUAL_BAND_FRACTIONS), 1)

    def test_silent_analog_input_still_has_source_noise(self):
        y=k6_gp0_pcm(np.zeros(8000), seed=9)
        self.assertAlmostEqual(float(np.std(y)), OFF_PCM_RMS_FS, places=7)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(len(y),8000)
        np.testing.assert_array_equal(y, k6_gp0_pcm(np.zeros(8000), seed=9))
        self.assertFalse(np.array_equal(y,k6_gp0_pcm(np.zeros(8000),seed=10)))
        np.testing.assert_array_equal(np.zeros(8000),k6_gp0_pcm(np.zeros(8000),noise_rms_fs=0))

    def test_measured_sine_transfer_three_points_without_noise(self):
        fs=FS
        t=np.arange(fs*2)/fs
        for hz,expected in zip(MEASURED_HZ,MEASURED_PCM_PEAK_FS):
            x=np.sin(2*np.pi*hz*t)
            got=k6_gp0_pcm(x,fs,noise_rms_fs=0,clip=False)
            tone=np.stack((np.sin(2*np.pi*hz*t),np.cos(2*np.pi*hz*t)),axis=1)
            coef=np.linalg.lstsq(tone,got,rcond=None)[0]
            peak=float(np.linalg.norm(coef))
            self.assertAlmostEqual(peak,expected,places=6)

    def test_source_noise_not_scaled_by_reference_input(self):
        x=np.zeros(8000)
        low=k6_gp0_pcm(x, input_vpp=.025,seed=42)
        high=k6_gp0_pcm(x, input_vpp=.25,seed=42)
        np.testing.assert_array_equal(low,high)
        t=np.arange(8000)/8000
        inp=np.sin(2*np.pi*1000*t)
        low=k6_gp0_pcm(inp,input_vpp=.05,seed=42)
        high=k6_gp0_pcm(inp,input_vpp=.1,seed=42)
        silence=k6_gp0_pcm(x,seed=42)
        np.testing.assert_allclose(high-silence,2*(low-silence),atol=1e-10)

    def test_banded_noise_is_not_white_and_preserves_rms(self):
        n=160000
        shaped=generate_gp0_noise(n,seed=17)
        white=generate_gp0_noise(n,seed=17,model='white')
        self.assertAlmostEqual(np.std(shaped),OFF_PCM_RMS_FS,places=8)
        self.assertAlmostEqual(np.std(white),OFF_PCM_RMS_FS,places=8)
        self.assertLess(np.max(np.abs(shaped)),.35)
        spectrum=np.abs(np.fft.rfft(shaped))**2
        freqs=np.fft.rfftfreq(n,1/8000)
        obs=[]
        for left,right in zip(RESIDUAL_BAND_EDGES_HZ[:-1],RESIDUAL_BAND_EDGES_HZ[1:]):
            band=(freqs>=left)&(freqs<right)
            obs.append(float(np.sum(spectrum[band])/np.sum(spectrum)))
        np.testing.assert_allclose(obs, RESIDUAL_BAND_FRACTIONS,atol=.025)

    def test_bad_inputs_and_controls_rejected_without_changes(self):
        good=np.zeros(800)
        for x in ([],[np.nan]*800, np.zeros((800,1)),np.zeros((1,800))):
            with self.assertRaises(ValueError):k6_gp0_pcm(x)
        for opts in ({'fs':44100},{'input_vpp':0},{'input_vpp':2},
                     {'noise_rms_fs':float('nan')},{'noise_model':'whatever'},
                     {'response':'whatever'}):
            with self.assertRaises(ValueError):
                k6_gp0_pcm(good,**opts)
        for opts in ({'model':'other'},{'rms_fs':-1}):
            with self.assertRaises(ValueError):generate_gp0_noise(800,**opts)
        with self.assertRaises(ValueError):generate_gp0_noise(20)

    def test_source_noise_does_not_modify_user_array(self):
        arr=np.ones(8000)*.5
        old=arr.copy()
        a=k6_gp0_pcm(arr,seed=0)
        np.testing.assert_array_equal(arr,old)
        self.assertNotEqual(float(np.std(a)),0)

if __name__=='__main__':
    unittest.main()
