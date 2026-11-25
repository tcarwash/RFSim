"""Model diagnostic statistics for deterministic channel timelines.

Diagnostics characterize *the simulator*, not a measurement against nature.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from .session import ChannelSession,PATHS

def analyze_profile(profile="mobile-urban", *, fs=48000,duration_s=60,
                    carrier_hz=146e6,seed=0):
    channel=ChannelSession(fs,profile,duration_s=duration_s,
                           carrier_hz=carrier_hz,seed=seed)
    g=sum(channel.gains)
    if channel.flutter is not None:g=g+channel.flutter
    g=g*channel.amplitude
    mean_power=float(np.mean(abs(g)**2))
    power_db=20*np.log10(np.maximum(abs(g),1e-12))
    f,psd=signal.welch(g,fs=channel.rate,nperseg=min(len(g),4096),
                       return_onesided=False,detrend="constant")
    power=np.maximum(np.real(psd),0)
    total=power.sum()
    centroid=float((f*power).sum()/total) if total>0 else 0.
    rms=float(np.sqrt(np.dot(power,(f-centroid)**2)/total)) if total>0 else 0.
    # Fraction of PSD inside the model's nominal mobile Jakes +/- fd,
    # including one FFT bin at the edges to accommodate spectral resolution.
    nominal=None;inside=None
    if profile in PATHS and channel.path.speed_kmh:
        nominal=(channel.path.scatter_hz+
                 channel.path.speed_kmh/3.6*carrier_hz/299792458)
        guard=channel.rate/min(len(g),4096)
        inside=float(power[abs(f)<=nominal+guard].sum()/total)
    return dict(profile=profile,carrier_hz=carrier_hz,fs=fs,seed=seed,
                duration_s=duration_s,fading_grid_rate=channel.rate,
                fading_grid_samples=len(g),doppler_rms_hz=rms,
                doppler_centroid_hz=centroid,nominal_max_doppler_hz=nominal,
                fraction_psd_within_nominal=inside,
                mean_channel_power=mean_power,
                amplitude_db_percentiles=[float(v) for v in np.percentile(power_db,[5,50,95])],
                disclaimer="Simulation-model diagnostics, not measured link quality")
