import argparse, json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from . import PROFILES, apply_iq, apply_audio, load_profile

def simulate(input_path, output_path, fs=None, profile="awgn", mode="iq", **kw):
    p = Path(input_path)
    if profile not in PROFILES: profile = load_profile(profile)
    if mode == "iq":
        if fs is None: raise ValueError("IQ requires sample rate")
        x = np.load(p,allow_pickle=False)
        y = apply_iq(x,fs,profile,**kw)
        np.save(output_path,y)
    else:
        fs,x = wavfile.read(p)
        if x.ndim != 1: raise ValueError("mono WAV required")
        if np.issubdtype(x.dtype,np.integer):
            if x.dtype == np.uint8: x = (x.astype(float)-128)/128
            else: x = x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
        y = apply_audio(x,fs,profile,mode=mode,**kw)
        wavfile.write(output_path,fs,y.astype(np.float32))
    actual = str(output_path)
    if mode == "iq" and not actual.endswith(".npy"): actual += ".npy"
    return {"output":actual,"sample_rate":fs,"samples":len(y),"mode":mode,**kw}

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="cmd",required=True)
    s.add_parser("profiles")
    a=s.add_parser("simulate")
    a.add_argument("input_path"); a.add_argument("output_path")
    a.add_argument("--fs",type=float); a.add_argument("--profile",default="awgn")
    a.add_argument("--mode",choices=["iq","ssb","fm"],default="iq")
    for name in ("snr-db","noise-bw-hz","cfo-hz","clock-ppm"):
        a.add_argument("--"+name,type=float,default=None if name in ("snr-db","noise-bw-hz") else 0)
    a.add_argument("--seed",type=int,default=0)
    d=vars(p.parse_args()); cmd=d.pop("cmd")
    print(json.dumps(list(PROFILES) if cmd=="profiles" else simulate(**d)))
if __name__ == "__main__": main()
