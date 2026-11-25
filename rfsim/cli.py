"""File-based RF simulation CLI (no radio hardware)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from . import PROFILES, apply_iq, apply_audio, load_profile

def simulate(input_path, output_path, fs=None, profile="awgn", mode="iq", **kw):
    p=Path(input_path)
    if profile not in PROFILES: profile=load_profile(profile)
    if mode=="iq":
        if fs is None:raise ValueError("IQ requires sample rate")
        x=np.load(p,allow_pickle=False)
        y=apply_iq(x,fs,profile,**kw)
        with open(output_path,"wb") as f: np.save(f,y)
    else:
        fs,x=wavfile.read(p)
        if x.ndim!=1:raise ValueError("mono WAV required")
        if np.issubdtype(x.dtype,np.integer):
            if x.dtype==np.uint8:x=(x.astype(float)-128)/128
            else:x=x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
        y=apply_audio(x,fs,profile,mode=mode,**kw)
        wavfile.write(output_path,fs,y.astype(np.float32))
    return dict(output=str(output_path),sample_rate=fs,samples=len(y),mode=mode,**kw)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("profiles")
    a.add_argument("--sessions",action="store_true")
    a=sub.add_parser("simulate")
    a.add_argument("input_path");a.add_argument("output_path")
    a.add_argument("--fs",type=float)
    a.add_argument("--profile",default="awgn")
    a.add_argument("--mode",choices=["iq","ssb","fm"],default="iq")
    for name in ("snr-db","noise-bw-hz","cfo-hz","clock-ppm"):
        a.add_argument("--"+name,type=float,default=None if name in ("snr-db","noise-bw-hz") else 0)
    a.add_argument("--seed",type=int,default=0)

    a=sub.add_parser("session")
    a.add_argument("input_path");a.add_argument("output_path");a.add_argument("--fs",type=float,required=True)
    a.add_argument("--profile",default="fixed");a.add_argument("--duration-s",type=float,default=120)
    a.add_argument("--start-s",type=float,default=0);a.add_argument("--carrier-hz",type=float,default=146e6)
    a.add_argument("--snr-db",type=float);a.add_argument("--seed",type=int,default=0)
    a.add_argument("--noise-bw-hz",type=float)
    a=sub.add_parser("estimate")
    a.add_argument("tx_path");a.add_argument("rx_path");a.add_argument("output_path")
    a.add_argument("--fs",type=float,required=True);a.add_argument("--taps",type=int,default=8)
    a.add_argument("--window",type=int,default=2048);a.add_argument("--hop",type=int)
    a.add_argument("--provenance-json",required=True,help="JSON object with capture provenance")
    a=sub.add_parser("replay")
    a.add_argument("trace_path");a.add_argument("input_path");a.add_argument("output_path")
    a.add_argument("--fs",type=float,required=True);a.add_argument("--start-s",type=float)
    a=sub.add_parser("experiment")
    a.add_argument("manifest")
    a=sub.add_parser("data2g")
    a.add_argument("input_path");a.add_argument("output_path")
    a.add_argument("--path",choices=["hf","fm"],required=True)
    a.add_argument("--profile",default="mpp");a.add_argument("--snr-db",type=float)
    a.add_argument("--radio",default="mic");a.add_argument("--carrier-mhz",type=float,default=146)
    a.add_argument("--seed",type=int,default=0)

    d=vars(p.parse_args());cmd=d.pop("cmd")
    if cmd=="profiles":
        from .session import PATHS
        out=sorted(set(PROFILES)|set(PATHS)) if d["sessions"] else sorted(PROFILES)
    elif cmd=="simulate":
        out=simulate(**d)
    elif cmd=="session":
        from .session import ChannelSession
        x=np.load(d.pop("input_path"),allow_pickle=False)
        output=d.pop("output_path");fs=d.pop("fs")
        profile=d.pop("profile")
        s=ChannelSession(fs,profile,duration_s=d.pop("duration_s"),
                         carrier_hz=d.pop("carrier_hz"),seed=d.pop("seed"))
        y=s.apply(x,start_s=d.pop("start_s"),snr_db=d.pop("snr_db"),
                  noise_bw_hz=d.pop("noise_bw_hz"))
        with open(output,"wb") as f:np.save(f,y)
        out=dict(output=output,samples=len(y),sample_rate=fs,profile=profile)
    elif cmd=="estimate":
        from .measured import estimate,save_trace
        provenance=json.loads(d.pop("provenance_json"))
        if not isinstance(provenance,dict) or not provenance:raise ValueError("nonempty provenance object required")
        tx=d.pop("tx_path");rx=d.pop("rx_path");output=d.pop("output_path")
        fs=d.pop("fs")
        t,delays,g,stats=estimate(np.load(tx,allow_pickle=False),np.load(rx,allow_pickle=False),fs,**d)
        save_trace(output,t,delays,g,{**provenance,"tx":tx,"rx":rx,"sample_rate":fs})
        out=dict(output=output,snapshots=len(t),mean_residual_power=float(np.mean(stats["residual_power"])))
    elif cmd=="replay":
        from .measured import MeasuredChannel
        c=MeasuredChannel(d["trace_path"])
        x=np.load(d["input_path"],allow_pickle=False)
        y=c.apply(x,d["fs"],start_s=d["start_s"])
        with open(d["output_path"],"wb") as f:np.save(f,y)
        out=dict(output=d["output_path"],samples=len(y),provenance=c.provenance)
    elif cmd=="experiment":
        from .experiments import run_experiment
        r=run_experiment(d["manifest"])
        out=dict(experiment=r["experiment"],cases=len(r["cases"]))
    elif cmd=="data2g":
        from .data2g_bridge import fm_audio,hf_audio
        from .experiments import _wav
        fs,x=_wav(d["input_path"])
        if d["path"]=="fm":
            y=fm_audio(x,fs,condition=d["profile"],cnr_db=d["snr_db"],
                       seed=d["seed"],carrier_mhz=d["carrier_mhz"],
                       duration_s=max(120,len(x)/fs+1))
        else:y=hf_audio(x,fs,preset=d["profile"],snr_db=d["snr_db"],seed=d["seed"])
        wavfile.write(d["output_path"],int(fs),np.asarray(y,dtype=np.float32))
        out=dict(output=d["output_path"],samples=len(y),sample_rate=fs,path=d["path"])
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
