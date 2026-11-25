"""Deterministic, file-backed offline experiment manifests; no subprocess execution."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numpy as np
from scipy.io import wavfile
from . import apply_iq, apply_audio, PROFILES, load_profile
from .session import ChannelSession, PATHS
from .measured import MeasuredChannel

def _hash(p: Path) -> str:
    h=sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _path(root: Path, value: str) -> Path:
    if not value: raise ValueError("empty file path")
    p=Path(value).expanduser()
    return (p if p.is_absolute() else root/p).resolve()

def _wav(path):
    fs,x=wavfile.read(path)
    if x.ndim != 1: raise ValueError("mono WAV required")
    if np.issubdtype(x.dtype,np.integer):
        maxval=max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x=x.astype(float)/maxval if x.dtype!=np.uint8 else (x.astype(float)-128)/128
    else:
        x=x.astype(float)
    if not np.isfinite(x).all(): raise ValueError("nonfinite audio")
    return float(fs),x

def _metric(x,y,fs):
    n=min(len(x),len(y))
    if n==0: return {}
    x=x[:n];y=y[:n]
    p=float(np.mean(abs(x)**2))
    q=float(np.mean(abs(y)**2))
    err=float(np.mean(abs(y-x)**2))
    # Not a BER or channel SNR measurement: raw waveform alignment matters.
    return dict(tx_rms=math.sqrt(p),rx_rms=math.sqrt(q),
                raw_error_rms=math.sqrt(err),
                raw_nmse_db=(10*math.log10(err/p) if p and err else None))

def run_experiment(manifest_path: str, *, max_cases=1000) -> dict:
    mp=Path(manifest_path).expanduser().resolve()
    cfg=json.loads(mp.read_text())
    if cfg.get("schema")!=1: raise ValueError("experiment schema must be 1")
    root=mp.parent
    sig=cfg.get("signal",{})
    input_path=_path(root,sig["path"])
    fmt=sig.get("format","iq_npy")
    if fmt=="iq_npy":
        x=np.load(input_path,allow_pickle=False)
        if x.ndim!=1 or not np.iscomplexobj(x) or not len(x): raise ValueError("iq_npy needs nonempty complex vector")
        fs=float(sig["sample_rate"])
        if not np.isfinite(fs) or fs<=0: raise ValueError("invalid sample rate")
    elif fmt=="audio_wav":
        fs,x=_wav(input_path)
    else: raise ValueError("format must be iq_npy or audio_wav")
    chan=cfg.get("channel",{})
    interface=chan.get("interface","iq" if fmt=="iq_npy" else "ssb")
    profile=chan.get("profile","awgn")
    sweep=cfg.get("sweep",{})
    levels=sweep.get("snr_db",[None])
    seeds=sweep.get("seeds",[0])
    if not levels or not seeds or len(levels)*len(seeds)>max_cases: raise ValueError("invalid number of trials")
    if any(not isinstance(s,int) or s<0 for s in seeds): raise ValueError("seeds must be nonnegative integers")
    if any(s is not None and (not np.isfinite(s) or not -150<=s<=150) for s in levels): raise ValueError("invalid SNR")
    out=_path(root,cfg.get("output_dir","results/"+mp.stem))
    # Refuse a path that overwrites the waveform or manifest.
    if out==input_path or out==mp: raise ValueError("invalid output path")
    out.mkdir(parents=True,exist_ok=True)
    provenance=dict(manifest=str(mp),manifest_sha256=_hash(mp),signal=str(input_path),
                    signal_sha256=_hash(input_path),created_utc=datetime.now(timezone.utc).isoformat(),
                    profile=profile,interface=interface,fs=fs)
    rows=[]
    for i,snr in enumerate(levels):
        for j,seed in enumerate(seeds):
            kw=dict(snr_db=snr,seed=seed)
            if interface=="iq":
                if fmt!="iq_npy":raise ValueError("IQ requires iq_npy")
                p=PROFILES[profile] if profile in PROFILES else load_profile(_path(root,profile) if (root/profile).is_file() else profile)
                y=apply_iq(x,fs,p,noise_bw_hz=chan.get("noise_bw_hz"),cfo_hz=chan.get("cfo_hz",0),
                    clock_ppm=chan.get("clock_ppm",0),signal_power=chan.get("signal_power"),**kw)
            elif interface in ("ssb","fm"):
                if fmt!="audio_wav":raise ValueError("audio mode requires WAV")
                p=PROFILES[profile] if profile in PROFILES else load_profile(_path(root,profile) if (root/profile).is_file() else profile)
                y=apply_audio(x,fs,p,mode=interface,
                    noise_bw_hz=chan.get("noise_bw_hz"),cfo_hz=chan.get("cfo_hz",0),
                    signal_power=chan.get("signal_power"),**kw)
            elif interface=="session":
                if fmt!="iq_npy":raise ValueError("session requires iq_npy")
                s=ChannelSession(fs,profile,duration_s=chan.get("duration_s",120),
                    carrier_hz=chan.get("carrier_hz",146e6),seed=seed)
                y=s.apply(x,start_s=chan.get("start_s",0),snr_db=snr,
                    noise_bw_hz=chan.get("noise_bw_hz"),signal_power=chan.get("signal_power"),
                    noise_seed=chan.get("noise_seed",0))
            elif interface=="measured":
                if fmt!="iq_npy":raise ValueError("measured requires iq_npy")
                s=MeasuredChannel(_path(root,chan["trace_path"]))
                y=s.apply(x,fs,start_s=chan.get("start_s"))
                if snr is not None:
                    p=float(np.mean(abs(x)**2))
                    bw=chan.get("noise_bw_hz",fs)
                    rng=np.random.default_rng(seed)
                    if not 0<bw<=fs: raise ValueError("invalid noise bandwidth")
                    y=y+np.sqrt(p*fs/bw*10**(-snr/10)/2)*(rng.normal(size=len(y))+1j*rng.normal(size=len(y)))
            elif interface=="data2g-hf":
                if fmt!="audio_wav":raise ValueError("Data2G HF expects WAV")
                from .data2g_bridge import hf_audio
                y=hf_audio(x,fs,preset=profile,snr_db=snr,seed=seed,
                           cfo_hz=chan.get("cfo_hz",0),clock_ppm=chan.get("clock_ppm",0))
            elif interface=="data2g-fm":
                if fmt!="audio_wav":raise ValueError("Data2G FM expects WAV")
                from .data2g_bridge import fm_audio
                y=fm_audio(x,fs,condition=profile,cnr_db=snr,
                           radio=chan.get("radio","mic"),seed=seed,
                           carrier_mhz=chan.get("carrier_hz",146e6)/1e6,
                           duration_s=chan.get("duration_s",max(120.,len(x)/fs+1)),
                           start_s=chan.get("start_s",0))
            else: raise ValueError(f"unsupported interface: {interface}")
            case=f"case-{i:03d}-{j:03d}"
            output=out/(case+(".npy" if fmt=="iq_npy" else ".wav"))
            if fmt=="iq_npy":
                with output.open("wb") as f: np.save(f,y)
            else: wavfile.write(output,int(fs),np.asarray(y,dtype=np.float32))
            rows.append(dict(id=case,snr_db=snr,seed=seed,output=str(output),
                             output_sha256=_hash(output),samples=len(y),metrics=_metric(x,y,fs)))
    report=dict(schema=1,experiment=cfg.get("name",mp.stem),provenance=provenance,cases=rows,
                warning="Synthetic offline results; waveform mismatch is not BER, PER, nor measured RF sensitivity.")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report

def list_reports(directory: str) -> list[str]:
    return [str(p) for p in sorted(Path(directory).expanduser().resolve().glob("*/report.json"))]

def read_report(path: str) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text())
