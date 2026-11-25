"""RX-only, SoapyFang-compatible RFSim IqSource wrapper.

Uses source.read(n), .rate and .center_hz; no SoapySDR hardware APIs.
"""
from __future__ import annotations
import math
import numpy as np
from .session import ChannelSession
from .measured import apply_taps

class RfSimSource:
    live=False
    def __init__(self, source, profile="fixed", *, duration_s=120., seed=0,
                 snr_db=None, noise_bw_hz=25000., signal_power=None):
        if not hasattr(source,"read") or not getattr(source,"rate",0):
            raise TypeError("source must expose .read(n) and .rate")
        self.source=source
        self.rate=float(source.rate)
        self.center_hz=float(source.center_hz)
        self.dbm_offset=getattr(source,"dbm_offset",-10.)
        self.auto_floor_dbm=getattr(source,"auto_floor_dbm",None)
        self.tunable=getattr(source,"tunable",False)
        self.live=getattr(source,"live",False)
        if snr_db is not None and (not np.isfinite(snr_db) or not 0<noise_bw_hz<=self.rate):
            raise ValueError("invalid noise reference")
        if signal_power is not None and (not np.isfinite(signal_power) or signal_power<0):
            raise ValueError("invalid signal power")
        self.session=ChannelSession(self.rate,profile,duration_s=duration_s,
                                    carrier_hz=self.center_hz,seed=seed)
        self.snr_db=snr_db
        self.noise_bw_hz=noise_bw_hz
        self.signal_power=signal_power
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,77]))
        self.pos=0
        self.max_delay=max(t.delay_s for t in self.session.path.taps)
        # The offline sinc delay uses a symmetric 65-tap kernel.
        # Add a common 32-sample group delay to every path so the
        # streaming output never depends on future samples.
        self.latency_samples=32 if any(abs(t.delay_s*self.rate-round(t.delay_s*self.rate))>1e-9
                                       for t in self.session.path.taps) else 0
        self.history_len=int(math.ceil(self.max_delay*self.rate))+64
        self.history=np.zeros(self.history_len,dtype=np.complex128)
        self.profile=profile

    def describe(self):
        return f"RFSim({self.profile}) over {self.source.describe()}"

    def read(self,n):
        n=int(n)
        if n<=0: raise ValueError("positive sample count required")
        x=np.asarray(self.source.read(n),complex)
        if x.ndim!=1 or len(x)!=n or not np.all(np.isfinite(x)):
            raise ValueError("underlying source returned invalid IQ")
        hist=min(self.pos,self.history_len)
        padded=np.r_[self.history[-hist:] if hist else np.empty(0,complex),x]
        times=(self.pos-hist+np.arange(len(padded)))/self.rate
        delays,gains=self.session.impulse_response(times)
        y=apply_taps(padded,self.rate,delays+self.latency_samples/self.rate,gains)[hist:]
        if self.snr_db is not None:
            power=float(np.mean(abs(x)**2)) if self.signal_power is None else self.signal_power
            sigma2=power*self.rate/self.noise_bw_hz*10**(-self.snr_db/10)
            noise=self.rng.normal(size=(n,2))
            y+=math.sqrt(sigma2/2)*(noise[:,0]+1j*noise[:,1])
        self.history=np.r_[self.history,x][-self.history_len:]
        self.pos+=n
        return np.asarray(y,dtype=np.complex64)

    def skip(self,n):
        # Intentionally render to keep the channel and RNG stream consistent.
        n=int(n)
        if n<0: raise ValueError("negative skip")
        while n:
            k=min(n,16384)
            self.read(k)
            n-=k

    def tune(self,hz):
        if not self.source.tune(hz):return False
        self.center_hz=float(self.source.center_hz)
        self.session=ChannelSession(self.rate,self.profile,duration_s=self.session.duration_s,
                                    carrier_hz=self.center_hz,seed=self.session.seed)
        self.history[:]=0
        return True

    def close(self):
        self.source.close()
