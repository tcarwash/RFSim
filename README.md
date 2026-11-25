# rfsim

Standalone offline RF channel simulator. IQ is the primary interface; mono WAV adapters provide SSB and simplified FM radio paths. No radio hardware is accessed.

## Setup
```sh
python -m venv .venv
.venv/bin/pip install -e '.[mcp]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/rfsim profiles
.venv/bin/rfsim simulate input.npy output.npy --fs 48000 --profile nvis-d --snr-db 5 --noise-bw-hz 3000 --seed 42
.venv/bin/rfsim simulate input.wav output.wav --mode fm --profile vhf-mobile --snr-db 15
```

Python: `rfsim.apply_iq(samples, fs, profile, ...)` and `apply_audio(..., mode="ssb"|"fm")`. IQ arrays are complex, one dimensional, saved as NumPy .npy. Input rate is explicit. Audio files are mono WAV; output is float WAV. SNR refers to pre-channel mean signal power and noise integrated over the requested bandwidth (default full complex sample rate for IQ, 3 kHz real audio for SSB, 15 kHz RF for FM). Supply signal_power in Python to exclude silence. Seeds pair channel/noise realizations across SNR sweeps.

## Profiles
HF parameters from Data2G commit b278852cbc03d32d3c865e4208c17d2700806f57 (data2g/hfchannel.py, scripts/cpm_floor_study.py); SSTVAE e18700bf8be8f6a13657c9f449d4c55876847568 (sstvae/hfchannel.py) is corroborating background. Both upstream projects use Artistic License 2.0. Implementation here is independent, not a verbatim source transplant or guaranteed numerical replica.
- mpg/mpp/mpd: good/poor/disturbed two-path HF
- mps/mpa/mpl: upstream measurement-derived slow/typical/long-delay models
- nvis-d/lowlat-d/hilat-m/hilat-d: severe ITU profiles as identified upstream; high latitude is an auroral-region proxy, not a calibrated aurora scattering model
- flutter: research stress case
- vhf-los/vhf-mobile/uhf-mobile: illustrative Rician/multipath profiles, not measurements; Gaussian Doppler, not yet COST/Jakes spectra

Custom JSON files are accepted as profile arguments (see profiles/example-measured.json). Schema 1 requires provenance and taps, with delay_s, power_db, spread_hz (two sigma), optional k_db. Store capture IDs, frequency, timestamps, estimation method, uncertainty and references in provenance. The example is illustrative, not actual measurement data. Raw time-varying impulse-response replay and estimation from aligned captures are future work.

## MCP
Register a local stdio server using absolute path to .venv/bin/rfsim-mcp. Tools list_profiles, get_profile, simulate_iq, simulate_audio, run_sweep, inspect_profile_file. Files are local to the server host. Use only with trusted agents: tools can read/write files under the running OS account; no network service is exposed by default. Use a separate restricted OS account for untrusted remote clients.

## Roadmap / limitations
Burst-only, not block-continuous or full session simulation; no absolute link budget, empirical replay, path estimation, interference or squelch yet. FM adapter is simplified, not Data2G's full radio model. Next: continuous seeded channel state, measured profiles and provenance validation, Data2G FM parity comparisons (shadowing, spectra, flutter, radio front-end), spectral validation, independent modem sweeps, authenticated remote MCP when needed. PixelQSO and SoapyFang integration remain separate tasks.
