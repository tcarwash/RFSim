"""Finite, reproducible channel timelines; random access to bursts."""
from dataclasses import dataclass
import numpy as np
from scipy import signal
from . import Tap, PROFILES, load_profile

@dataclass(frozen=True)
class Path:
    taps: tuple
    speed_kmh: float = 0
    scatter_hz: float = 0
    shadow_db: float = 0
    shadow_distance_m: float = 50
    slow_db: float = 0
    slow_s: float = 60
    flutter_db: float | None = None
    flutter_hz: float = 150
    flutter_s: float = 30

# Delay/power tuples from COST 207 as documented by Data2G.
def _taps(rows):
    return tuple(Tap(delay_s=d*1e-6,power_db=p,k_db=k) for d,p,k in rows)
TU = _taps([(0,-3,None),(.2,0,None),(.6,-2,None),(1.6,-6,None),(2.4,-8,None),(5,-10,None)])
BU = _taps([(0,-3,None),(.4,0,None),(1,-3,None),(1.6,-5,None),(5,-2,None),(6.6,-4,None)])
HT = _taps([(0,0,None),(.2,-2,None),(.4,-4,None),(.6,-7,None),(15,-6,None),(17.2,-12,None)])
RA = _taps([(0,0,6.5),(.2,-2,None),(.4,-10,None),(.6,-20,None)])
PATHS = {
    "static": Path((Tap(),)),
    "fixed": Path(_taps([(0,0,13),(.4,-15,None),(.9,-20,None)]),scatter_hz=.2),
    "fixed-fringe": Path(_taps([(0,0,0),(.4,-5,None),(.9,-10,None)]),scatter_hz=.3,slow_db=5,slow_s=120),
    "knife-edge": Path(_taps([(0,0,3),(15,-10,None)]),scatter_hz=.3,slow_db=6,slow_s=30),
    "aircraft-flutter": Path((Tap(),),flutter_db=-12),
    "pedestrian": Path(TU,5,shadow_db=6,shadow_distance_m=20),
    "mobile-rural": Path(RA,90,shadow_db=5,shadow_distance_m=200),
    "mobile-urban": Path(TU,40,shadow_db=8,shadow_distance_m=50),
    "mobile-bad-urban": Path(BU,30,shadow_db=8,shadow_distance_m=30),
    "mobile-hilly": Path(HT,80,shadow_db=6,shadow_distance_m=200),
}

def _ou(n, rate, sigma, tau, rng):
    if not sigma: return np.zeros(n)
    a=np.exp(-1/(rate*tau))
    x=rng.normal(0,sigma*np.sqrt(-np.expm1(-2/(rate*tau))),n)
    x[0]=rng.normal(0,sigma)
    return signal.lfilter([1],[1,-a],x)

def _doppler(n, rate, spread, mobile, rng):
    if not spread: return np.full(n,1+0j)
    # Gaussian 2-sigma for HF; integrated Jakes bins for mobile.
    f=np.fft.fftfreq(n,1/rate)
    if mobile:
        df=rate/n
        psd=np.arcsin(np.clip((f+df/2)/spread,-1,1))-np.arcsin(np.clip((f-df/2)/spread,-1,1))
    else:
        psd=np.exp(-.5*(f/(spread/2))**2)
    w=(rng.normal(size=n)+1j*rng.normal(size=n))*np.sqrt(psd)
    # Ensemble normalization, not per-realization: preserve deep fades.
    return np.fft.ifft(w)*n/np.sqrt(2*psd.sum())

class ChannelSession:
    def __init__(self, fs, profile="awgn", *, duration_s=120, carrier_hz=146e6, seed=0):
        if not np.isfinite(fs) or fs<=0 or not 0<duration_s<=86400 or not 0<carrier_hz:
            raise ValueError("invalid timeline")
        self.fs=float(fs); self.duration_s=duration_s; self.seed=seed
        p=PATHS[profile] if isinstance(profile,str) and profile in PATHS else Path((PROFILES[profile] if profile in PROFILES else load_profile(profile)) if isinstance(profile,str) else tuple(profile))
        if not p.taps: raise ValueError("empty path")
        self.path=p
        fd=p.scatter_hz+p.speed_kmh/3.6*carrier_hz/299792458
        flutter=p.flutter_hz*carrier_hz/146e6 if p.flutter_db is not None else 0
        self.rate=max(2.,32*max(fd,flutter,*(t.spread_hz for t in p.taps)))
        n=int(np.ceil(duration_s*self.rate))+2
        if n>5_000_000: raise ValueError("timeline grid too large; shorten duration")
        rng=np.random.default_rng(seed)
        self.grid=np.arange(n)/self.rate
        weights=np.array([10**(t.power_db/10) for t in p.taps]); weights/=weights.sum()
        self.gains=[]
        for t,w in zip(p.taps,weights):
            g=_doppler(n,self.rate,fd or t.spread_hz,bool(p.speed_kmh),rng)
            if t.k_db is not None:
                k=10**(t.k_db/10)
                los=np.exp(1j*(rng.uniform(0,2*np.pi)+2*np.pi*.7*fd*self.grid))
                g=(g+np.sqrt(k)*los)/np.sqrt(1+k)
            self.gains.append(np.sqrt(w)*g)
        tau=p.shadow_distance_m/(p.speed_kmh/3.6) if p.speed_kmh else 60
        self.amplitude=10**((_ou(n,self.rate,p.shadow_db,tau,rng)+_ou(n,self.rate,p.slow_db,p.slow_s,rng))/20)
        self.flutter=None
        if flutter:
            tri=4*abs((self.grid/p.flutter_s+rng.random())%1-.5)-1
            self.flutter=10**(p.flutter_db/20)*np.exp(1j*(2*np.pi*np.cumsum(tri*flutter)/self.rate+rng.uniform(0,2*np.pi)))

    def impulse_response(self, times):
        times=np.asarray(times,float)
        if np.any(~np.isfinite(times)) or np.any(times<0) or np.any(times>self.duration_s):
            raise ValueError("outside session timeline")
        a=np.interp(times,self.grid,self.amplitude)
        gains=np.stack([np.interp(times,self.grid,g)*a for g in self.gains],axis=-1)
        delays=np.array([t.delay_s for t in self.path.taps])
        if self.flutter is not None:
            gains=np.concatenate([gains,(np.interp(times,self.grid,self.flutter)*a)[...,None]],axis=-1)
            delays=np.r_[delays,0.]
        return delays,gains

    def apply(self, samples, *, start_s=0, snr_db=None, noise_bw_hz=None, signal_power=None, cfo_hz=0, noise_seed=0):
        from .measured import apply_taps
        x=np.asarray(samples,complex)
        if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)): raise ValueError("invalid IQ")
        times=start_s+np.arange(len(x))/self.fs
        delays,gains=self.impulse_response(times)
        y=apply_taps(x,self.fs,delays,gains)
        y*=np.exp(2j*np.pi*np.remainder(cfo_hz*times,1))
        if snr_db is not None:
            bw=self.fs if noise_bw_hz is None else noise_bw_hz
            power=np.mean(abs(x)**2) if signal_power is None else signal_power
            if not 0<bw<=self.fs or not np.isfinite(power) or power<0: raise ValueError("invalid noise reference")
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,noise_seed]))
            y+=np.sqrt(power*self.fs/bw*10**(-snr_db/10)/2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
        return y
