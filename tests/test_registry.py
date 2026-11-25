import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rfsim.registry import import_profile,list_imported_profiles
from rfsim import load_profile,apply_iq
from rfsim.session import ChannelSession
from rfsim.measured import save_trace

class RegistryTests(unittest.TestCase):
    def test_parameterized_and_measured_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with patch("rfsim.registry.registry_dir",return_value=p/"registered"):
                source=p/"portable.json"
                source.write_text(json.dumps({"schema":1,"provenance":{"capture":"synth-1"},
                                               "taps":[{"delay_s":0,"power_db":0,"spread_hz":.1}]}))
                result=import_profile(str(source),name="measured-demo")
                self.assertEqual(result["kind"],"taps")
                self.assertEqual(len(load_profile("measured-demo")),1)
                out=apply_iq(np.ones(2000,dtype=complex),2000,"measured-demo")
                self.assertTrue(np.isfinite(out).all())
                self.assertEqual(len(ChannelSession(2000,"measured-demo",duration_s=1).apply(np.ones(100))),100)
                self.assertEqual(len(list_imported_profiles()),1)
                self.assertEqual(import_profile(str(source),name="measured-demo")["sha256"],result["sha256"])
                source.write_text(source.read_text()+" ")
                with self.assertRaises(FileExistsError):
                    import_profile(str(source),name="measured-demo")
                with self.assertRaises(ValueError):
                    import_profile(str(source),name="../escape")
                trace=p/"trace.npz"
                save_trace(trace,np.array([0,1.]),np.array([0.]),
                           np.ones((2,1),complex),{"kind":"synthetic"})
                result=import_profile(str(trace),name="trace-demo")
                self.assertEqual(result["kind"],"measured")
                self.assertEqual(len(list_imported_profiles()),2)
if __name__=="__main__": unittest.main()
