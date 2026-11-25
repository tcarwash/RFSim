"""Independent offline RF channel model."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy import signal, ndimage

@dataclass(frozen=True)
class Tap:
    delay_s: float = 0
    power_db: float = 0
    spread_hz: float = 0
    k_db: float | None = None

def hf(spread, delay_ms):
    return (Tap(spread_hz=spread), Tap(delay_ms/1000, spread_hz=spread))

PROFILES = {
    "awgn": (Tap(),),
    "mpg": hf(.1,.5), "mpp": hf(1,2), "mpd": hf(2,4),
    "mps": hf(.15,2), "mpa": hf(.25,2), "mpl": hf(.3,3.5),
    "nvis-d": hf(1,7), "lowlat-d": hf(10,6),
    "hilat-m": hf(10,3), "hilat-d": hf(30,7),
    "flutter": hf(10,.5),
    "vhf-los": (Tap(k_db=12, spread_hz=.3), Tap(2e-6,-12,.5)),
    "vhf-mobile": (Tap(k_db=3,spread_hz=20), Tap(3e-6,-4,20), Tap(12e-6,-10,20)),
    "uhf-mobile": (Tap(k_db=3,spread_hz=60), Tap(3e-6,-4,60), Tap(12e-6,-10,60)),
}

def load_profile(path):
    from .registry import resolve_profile_path
    doc = json.loads(resolve_profile_path(str(path)).read_text())
    if doc.get("schema") != 1 or not doc.get("provenance"):
        raise ValueError("profile requires schema=1 and provenance")
    taps = tuple(Tap(**t) for t in doc["taps"])
    if not taps or any(t.delay_s < 0 or t.spread_hz < 0 for t in taps):
        raise ValueError("invalid taps")
    return taps

def _gain(n, fs, tap, rng):
    if tap.spread_hz == 0:
        return 1.0
    rate = min(fs, max(32*tap.spread_hz, 1.0))
    sigma = rate/(np.sqrt(8)*np.pi*(tap.spread_hz/2))
    pad = int(np.ceil(5*sigma))+2
    count = int(np.ceil(n*rate/fs))+2
    z = (rng.normal(size=count+2*pad)+1j*rng.normal(size=count+2*pad))/np.sqrt(2)
    impulse = np.zeros(2*pad+1); impulse[pad] = 1
    kernel = ndimage.gaussian_filter1d(impulse,sigma,mode="constant",truncate=5)
    z = ndimage.gaussian_filter1d(z,sigma,mode="constant",truncate=5)[pad:pad+count]/np.linalg.norm(kernel)
    g = np.interp(np.arange(n)*rate/fs,np.arange(count),z)
    if tap.k_db is not None:
        k = 10**(tap.k_db/10)
        g = (g+np.sqrt(k))/np.sqrt(1+k)
    return g

def apply_iq(samples, fs, profile="awgn", *, snr_db=None, noise_bw_hz=None,
             signal_power=None, cfo_hz=0, clock_ppm=0, seed=0):
    """Offline burst; SNR referenced to input mean power in specified bandwidth."""
    x = np.asarray(samples,dtype=np.complex128)
    if x.ndim != 1 or not len(x) or fs <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("expected finite nonempty 1-D samples and positive sample rate")
    bw = fs if noise_bw_hz is None else noise_bw_hz
    if not 0 < bw <= fs or 1+clock_ppm*1e-6 <= 0:
        raise ValueError("invalid bandwidth or clock error")
    taps = (PROFILES[profile] if profile in PROFILES else load_profile(profile)) if isinstance(profile,str) else tuple(profile)
    if not taps: raise ValueError("empty profile")
    fade_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(fade_seed)
    weights = np.array([10**(t.power_db/10) for t in taps])
    weights /= weights.sum()
    y = np.zeros_like(x)
    for t,w in zip(taps,weights):
        d = t.delay_s*fs
        # Windowed sinc fractional delay; zero-padded burst boundaries.
        integer = int(np.floor(d)); frac = d-integer
        if integer >= len(x): continue
        if frac:
            k = np.arange(-32,33)
            h = np.sinc(k-frac)*np.kaiser(len(k),8)
            h /= h.sum()
            v = signal.fftconvolve(x,h,mode="full")[32:32+len(x)]
        else: v = x
        v = np.pad(v[:len(x)-integer],(integer,0))
        y += np.sqrt(w)*v*_gain(len(x),fs,t,rng)
    y *= np.exp(2j*np.pi*np.remainder(cfo_hz*np.arange(len(x))/fs,1))
    if snr_db is not None:
        p = np.mean(abs(x)**2) if signal_power is None else signal_power
        if p < 0: raise ValueError("negative signal power")
        rng = np.random.default_rng(noise_seed)
        y += np.sqrt(p*fs/bw*10**(-snr_db/10)/2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    if clock_ppm:
        y = signal.resample(y,max(1,round(len(y)/(1+clock_ppm*1e-6))))
    return y

def apply_audio(samples, fs, profile="awgn", *, mode="ssb", deviation_hz=2500, **kwargs):
    x = np.asarray(samples,dtype=float)
    if mode == "ssb":
        if kwargs.get("noise_bw_hz") is None: kwargs["noise_bw_hz"] = min(3000,fs/2)
        # Analytic signal noise has twice the positive-band density of real audio.
        kwargs["noise_bw_hz"] *= 2
        return apply_iq(signal.hilbert(x),fs,profile,**kwargs).real
    if mode != "fm": raise ValueError("mode must be ssb or fm")
    factor = max(1,int(np.ceil(96000/fs)))
    rate = fs*factor
    a = signal.resample_poly(x,factor,1)
    z = np.exp(2j*np.pi*np.cumsum(a)*deviation_hz/rate)
    if kwargs.get("noise_bw_hz") is None: kwargs["noise_bw_hz"] = 15000
    y = apply_iq(z,rate,profile,**kwargs)
    y = signal.sosfilt(signal.butter(4,7500,fs=rate,output="sos"),y)
    demod = np.angle(y[1:]*y[:-1].conj())*rate/(2*np.pi*deviation_hz)
    return signal.resample_poly(np.r_[0,demod],1,factor)
