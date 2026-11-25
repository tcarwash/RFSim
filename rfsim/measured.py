"""Measured complex impulse response estimation and replay."""
import json
from pathlib import Path
import numpy as np
from scipy import signal

def apply_taps(x,fs,delays,gains):
    y=np.zeros(len(x),complex)
    for d,g in zip(delays,gains.T):
        ds=d*fs; integer=int(np.floor(ds)); frac=ds-integer
        if integer>=len(x): continue
        if frac:
            k=np.arange(-32,33)
            h=np.sinc(k-frac)*np.kaiser(len(k),8); h/=h.sum()
            v=signal.fftconvolve(x,h,mode="full")[32:32+len(x)]
        else: v=x
        y[integer:]+=g[integer:]*v[:len(x)-integer]
    return y

def estimate(tx,rx,fs,*,taps=8,window=2048,hop=None,ridge=1e-6):
    """Aligned, synchronized wideband reference; causal integer-delay LS per window."""
    x=np.asarray(tx,complex); y=np.asarray(rx,complex)
    hop=window//2 if hop is None else hop
    if x.ndim!=1 or x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("aligned finite equal-length IQ required")
    if not 0<fs or not 1<=taps<window or not 1<=hop<=window or len(x)<window or ridge<0:
        raise ValueError("invalid estimation dimensions")
    padded=np.pad(x,(taps-1,0))
    rows=np.lib.stride_tricks.sliding_window_view(padded,taps)[:,::-1]
    times=[]; gains=[]; residual=[]; condition=[]
    for start in range(0,len(x)-window+1,hop):
        a=rows[start:start+window]; b=y[start:start+window]
        scale=np.linalg.norm(a)**2/taps
        if scale<=0: raise ValueError("reference has no energy")
        gram=a.conj().T@a
        cond=np.linalg.cond(gram)
        if not np.isfinite(cond) or cond>1e8: raise ValueError("reference is not sufficiently broadband to identify taps")
        h=np.linalg.solve(gram+ridge*scale*np.eye(taps),a.conj().T@b)
        gains.append(h); times.append((start+(window-1)/2)/fs)
        residual.append(float(np.mean(abs(b-a@h)**2))); condition.append(float(cond))
    return np.array(times),np.arange(taps)/fs,np.array(gains),{"residual_power":residual,"condition":condition}

def save_trace(path,times,delays,gains,provenance):
    if not isinstance(provenance,dict) or not provenance: raise ValueError("provenance required")
    with open(path,"wb") as f:
        np.savez_compressed(f,times=times,delays=delays,gains=gains,provenance=json.dumps(provenance))

class MeasuredChannel:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:
            self.times=f["times"]; self.delays=f["delays"]; self.gains=f["gains"]
            self.provenance=json.loads(str(f["provenance"]))
        if (self.times.ndim!=1 or self.delays.ndim!=1 or not len(self.times) or not len(self.delays)
            or self.gains.shape!=(len(self.times),len(self.delays)) or np.any(np.diff(self.times)<=0)
            or np.any(self.delays<0) or not all(np.all(np.isfinite(v)) for v in (self.times,self.delays,self.gains))):
            raise ValueError("invalid measured channel trace")

    def apply(self,x,fs,*,start_s=None):
        x=np.asarray(x,complex)
        if x.ndim!=1 or not len(x) or fs<=0 or not np.all(np.isfinite(x)): raise ValueError("invalid IQ")
        start=self.times[0] if start_s is None else start_s
        t=start+np.arange(len(x))/fs
        if t[0]<self.times[0] or t[-1]>self.times[-1]: raise ValueError("outside measured interval")
        g=np.stack([np.interp(t,self.times,v) for v in self.gains.T],axis=-1)
        return apply_taps(x,fs,self.delays,g)
