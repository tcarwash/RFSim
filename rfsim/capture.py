"""Read finite IQ captures from NumPy or SoapyFang-compatible SigMF files."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_iq_capture(path, *, fs=None, max_samples=10_000_000):
    p=Path(path).expanduser().resolve()
    if p.suffix==".npy":
        x=np.load(p,mmap_mode="r",allow_pickle=False)
        if fs is None: raise ValueError("NPY IQ requires explicit sample rate")
        meta={"path":str(p),"format":"npy","sample_rate":float(fs)}
        rate=float(fs)
        if x.dtype.kind!="c" or x.ndim!=1: raise ValueError("expected complex 1-D NPY")
    else:
        base=str(p).removesuffix(".sigmf-meta").removesuffix(".sigmf-data")
        with open(base+".sigmf-meta") as f: document=json.load(f)
        core=document.get("global",{})
        if len(document.get("captures",[]))!=1 or document["captures"][0].get("core:sample_start",0)!=0:
            raise ValueError("one single-frequency SigMF capture expected")
        fmt=core["core:datatype"]
        dtype_stride={"cf32_le":("<c8",1),"ci16_le":("<i2",2),"ci8":("i1",2)}
        if fmt not in dtype_stride:raise ValueError("unsupported SigMF datatype")
        dtype,stride=dtype_stride[fmt]
        data=Path(base+".sigmf-data")
        if not data.is_file() or data.stat().st_size%(np.dtype(dtype).itemsize*stride):
            raise ValueError("missing or truncated SigMF data")
        x=np.memmap(data,mode="r",dtype=dtype)
        rate=float(core["core:sample_rate"])
        if fs is not None and abs(rate-fs)>1e-6:raise ValueError("SigMF sample rate mismatch")
        meta=dict(path=base,format="sigmf",sample_rate=rate,
                  center_hz=float(document["captures"][0]["core:frequency"]),
                  datatype=fmt,description=core.get("core:description",""))
    count=len(x)//stride if p.suffix!=".npy" else len(x)
    if not np.isfinite(rate) or rate<=0 or not 0<count<=max_samples:
        raise ValueError("invalid rate or capture size")
    if p.suffix!=".npy" and stride==2:
        factor=1/32768 if fmt=="ci16_le" else 1/128
        x=x[::2].astype(float)*factor+1j*x[1::2].astype(float)*factor
    else:
        x=np.asarray(x,complex)
    if not np.isfinite(x).all():raise ValueError("nonfinite IQ")
    return x,rate,meta
