"""Use upstream Data2G's authoritative full FM radio/channel chain when installed.

No Data2G source is copied; this adapter deliberately keeps its original
Artistic-2.0 package as an optional dependency.
"""
from __future__ import annotations
import numpy as np

def fm_audio(x, fs, *, condition="static", radio="mic", cnr_db=30., seed=0,
             carrier_mhz=146., duration_s=120., start_s=0., burst=0,
             drive_dev_hz=1500., rf_offset_hz=0., ppm=0., squelch_db=None,
             keyup_s=0., rx_dc=0.):
    try:
        from data2g.fmchannel import FmChannel, CONDITIONS, RADIOS
    except ImportError as e:
        raise RuntimeError("Full FM model needs installed upstream 'data2g'; "
                           "see https://github.com/arodland/Data2G") from e
    if condition not in CONDITIONS or radio not in RADIOS:
        raise ValueError("unknown Data2G FM condition or radio")
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.all(np.isfinite(x)):
        raise ValueError("finite mono audio required")
    if not np.isfinite(fs) or fs<=0 or int(fs)!=fs:
        raise ValueError("integer audio sample rate required")
    if not 0<=start_s or not 0<duration_s or start_s+len(x)/fs>duration_s:
        raise ValueError("burst exceeds session")
    channel=FmChannel(condition=condition,radio=radio,cnr_db=cnr_db,seed=seed,
                      carrier_mhz=carrier_mhz,fs=int(fs),duration_s=duration_s,
                      drive_dev_hz=drive_dev_hz,rf_offset_hz=rf_offset_hz,
                      ppm=ppm,squelch_db=squelch_db,keyup_s=keyup_s,rx_dc=rx_dc)
    return channel.apply(x,t0=start_s,burst=burst)

def hf_audio(x,fs,*,preset="mpp",snr_db=0.,cfo_hz=0.,clock_ppm=0.,seed=0):
    """Data2G's native HF channel at its own DSP clock, resampled at edges."""
    try:
        from data2g import hfchannel, config
        from scipy import signal
        from fractions import Fraction
    except ImportError as e:
        raise RuntimeError("Full HF model needs Data2G and SciPy") from e
    x=np.asarray(x,float)
    if x.ndim!=1 or not len(x) or not np.all(np.isfinite(x)) or fs<=0:
        raise ValueError("invalid mono audio")
    ratio=Fraction(int(config.FS),int(fs))
    if abs(float(config.FS)-ratio*fs)>1e-6: raise ValueError("noninteger input sample rate")
    internal=signal.resample_poly(x,ratio.numerator,ratio.denominator)
    fading=(None if preset=="awgn" else
            hfchannel.ITU_PRESETS.get(preset,hfchannel.FADING_PRESETS.get(preset)))
    if preset!="awgn" and fading is None: raise ValueError("unknown Data2G HF preset")
    result=hfchannel.apply_channel(internal,snr_db=snr_db,freq_offset_hz=cfo_hz,
                                   ppm=clock_ppm,fading_preset=fading,
                                   seed=seed)
    return signal.resample_poly(result,ratio.denominator,ratio.numerator)[:len(x)]
