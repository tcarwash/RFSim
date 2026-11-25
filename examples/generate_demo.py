"""Create a synthetic 2-second IQ test vector for experiments/example.json."""
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
fs=12000;n=fs*2
t=np.arange(n)/fs
# Continuous, known complex narrowband FSK-like training sequence.
frequency=np.where((t*20).astype(int)%2==0,1000,1200)
wave=np.exp(2j*np.pi*np.cumsum(frequency)/fs).astype(np.complex64)
np.save(root/"experiments"/"example.npy",wave)
print("Wrote experiments/example.npy")
