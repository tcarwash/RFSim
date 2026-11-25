import unittest
from rfsim.validation import analyze_profile

class ValidationTests(unittest.TestCase):
    def test_static_is_constant(self):
        result=analyze_profile("static",duration_s=5)
        self.assertAlmostEqual(result["mean_channel_power"],1.)
        self.assertAlmostEqual(result["doppler_rms_hz"],0.)

    def test_mobile_doppler_spectral_support(self):
        result=analyze_profile("mobile-urban",carrier_hz=146e6,duration_s=60,seed=0)
        self.assertGreater(result["doppler_rms_hz"],1)
        self.assertLess(result["doppler_rms_hz"],result["nominal_max_doppler_hz"]*1.1)
        self.assertGreater(result["fraction_psd_within_nominal"],.95)
        self.assertLess(result["amplitude_db_percentiles"][0],
                        result["amplitude_db_percentiles"][2])

if __name__=="__main__":unittest.main()
