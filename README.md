# RFSim — independent RF propagation test platform

Offline, deterministic simulation with complex IQ as the canonical interface and
mono audio adapters for SSB and FM. The propagation channel is independent of
Flux, SoapyFang, PixelQSO, or any specific modem.

**Workstation:** `~/dev/rfsim`. **No physical transmissions.**

## Quick start

```sh
cd ~/dev/rfsim
.venv/bin/pip install -e '.[mcp]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/rfsim profiles --sessions
.venv/bin/rfsim simulate tx.npy rx.npy --fs 48000 --profile nvis-d --snr-db 4 --noise-bw-hz 3000 --seed 42
.venv/bin/rfsim simulate tx.wav rx.wav --mode ssb --profile mpp --snr-db 8
.venv/bin/rfsim session tx.npy rx.npy --fs 48000 --profile mobile-urban --duration-s 60 --carrier-hz 146520000 --start-s 12 --snr-db 20
python examples/generate_demo.py
.venv/bin/rfsim experiment experiments/example.json
```

The basic IQ API is `apply_iq(x, fs, profile, snr_db=..., seed=...)`.
`apply_audio(x,fs,profile,mode="ssb"|"fm")` adds a waveform-level
audio adapter. The simplified FM adapter is **not** equivalent to a radio
transmitter/receiver or BK4819.

A full Data2G model, when that separately licensed package is installed:

```sh
.venv/bin/rfsim data2g tx.wav rx.wav --path hf --profile nvis-d --snr-db 10
.venv/bin/rfsim data2g tx.wav rx.wav --path fm --profile mobile-urban --radio flat --snr-db 20
```

Data2G supplies the original HF Watterson and complete FM radio TX/RX paths:
filters, emphasis, squelch, Rician fading, slow shadowing, aircraft flutter
and carrier-dependent Doppler. The wrapper resamples HF audio to/from
Data2G's native sample rate; direct internal baseband values are unchanged.
The full Data2G backend is **optional**, imported lazily, and retains its
upstream Artistic-2.0 licensing. It is a dependency, not vendored source.

## Channel profiles

Burst profiles: `awgn, mpg, mpp, mpd, mps, mpa, mpl, nvis-d, lowlat-d,
hilat-m, hilat-d, flutter, vhf-los, vhf-mobile, uhf-mobile`.
HF parameters are drawn from the source and research notes of
[Data2G](https://github.com/arodland/Data2G) at
b278852cbc03d32d3c865e4208c17d2700806f57 and
[SSTVAE](https://github.com/arodland/SSTVAE) at
e18700bf8be8f6a13657c9f449d4c55876847568.
These are not bit-for-bit copies of the upstream channel engine.

Time-persistent session profiles add `static, fixed, fixed-fringe, knife-edge,
aircraft-flutter, pedestrian, mobile-rural, mobile-urban, mobile-bad-urban,
mobile-hilly`. They provide real-time indexed fading and shadowing for
bursts at distinct session times. A burst's propagation state is deterministic
for the same seed, carrier and timeline, but burst input history is not
automatically continuous. Session time does not wrap.

The VHF/UHF examples are **model assumptions**, not calibrated measurements.
The `hilat-*` conditions represent high-latitude HF channels, not a validated
auroral flutter/angle-spread model. "Tropical" is approximated using the
low-latitude profile, rather than a measured equatorial scintillation model.

## Measured profiles

Validated JSON and measured NPZ files can be registered with the
MCP register_profile tool or rfsim.registry.import_profile(path,
name="my-path"). Registered profiles live in
~/.local/share/rfsim/profiles/. Existing names cannot be overwritten
with different files. Names of registered parametric profiles can be
used by apply_iq and ChannelSession.

Parametric JSON profiles (see `profiles/example-measured.json`) require
`schema=1`, `provenance`, and taps with `delay_s,power_db,spread_hz,k_db`.
Do **not** label illustrative entries as measurements.

MCP estimate_capture_profile accepts aligned NumPy and SoapyFang-compatible
SigMF recordings, including complex-float and interleaved integer IQ,
while recording source sample rate and center frequency metadata.

For recorded time-varying channels:

```sh
.venv/bin/rfsim estimate aligned_tx.npy aligned_rx.npy profile.npz \
  --fs 48000 --taps 8 --window 2048 \
  --provenance-json '{"kind":"measured","date":"YYYY-MM-DD","frequency_hz":146520000,"capture_id":"example"}'
.venv/bin/rfsim replay profile.npz other_tx.npy other_rx.npy --fs 48000
```

The estimator fits causal integer-sample multipath through regularized
windowed least squares and rejects poorly conditioned reference waveforms.
Time-varying complex tap matrices, delays, time grids and provenance are stored
in compressed NPZ files. The replay interpolates taps over its measured interval.
The TX/RX files **must already be frequency/time aligned** and broadband enough
to estimate taps. Replay does not include noise, receiver nonlinearities, or
RF front-end calibration. Synthetic estimation and replay are regression-tested;
a true measured over-the-air pair has not yet been validated.

## Optional batched IQ CPU/GPU simulation

RFSim also provides a separate [batched IQ API and DORC throughput
benchmark](contrib/experiment-queue/GPU_BACKEND.md). The legacy
`apply_iq` implementation remains the default scientific reference.

`rfsim.batch.apply_iq_batch(..., backend="numpy")` reproduces the current
CPU simulator. Set `backend="torch", device="cuda"` to request an explicitly
installed PyTorch CUDA/ROCm runtime and batched device arithmetic.
Use `iter_iq_batches` for bounded chunked outputs. Both NVIDIA and AMD
PyTorch GPU runtimes use the `cuda` device name.

Torch does **not** silently fall back to CPU. The GPU backend currently
uses seeded CPU-generated fading/noise so numerical results can be checked
against NumPy; it is not guaranteed to be faster. Measurements and backend
limits are in the linked guide. The standard DORC 45-case regression
suite remains CPU-based.

## Reproducible experiment manifests

`rfsim experiment experiments/example.json` (or MCP `run_manifest`) executes
an input `iq_npy` or `audio_wav` across an SNR × seed grid. Supported
interfaces: `iq`, `ssb`, `fm`, `session`, `measured`,
`data2g-hf`, `data2g-fm`. Every output is written with a stable case ID;
the report includes profile, clock, source and manifest SHA256, seeds and
waveform-level metrics. The raw error RMS/NMSE is not BER, PER, or true
post-demodulator SNR. Each client modem must provide its own decoder and
link-layer metrics. Reports include a generated UTC timestamp.

## Agent control (MCP)

Install the `mcp>=1.12,<2` extra; launch the `rfsim-mcp` entrypoint as
a local stdio MCP server. Tools include profile discovery, IQ/audio simulations,
HF/FM Data2G simulations, session experiments, measured-profile estimation,
inspection/replay, manifest execution, and result lookup.

Example agent registration:

```json
{"mcpServers":{"rfsim":{"command":"/home/tyler/dev/rfsim/.venv/bin/rfsim-mcp"}}}
```

MCP passes filenames, never giant base64 waveform payloads. **It executes under
the caller's OS account and can read/write local paths.** Connect trusted
agents only; use a restricted OS user for external clients. No network listener
or physical RF hardware is opened by RFSim.

## External integrations

**PixelQSO:** `~/dev/cardmodem/pixelqso-source/rfsim_sweep.py` generates a
real card with its selected TX backend, passes the waveform through RFSim
(or the full Data2G model), calls that backend's actual decoder, and writes
`pixelqso-rfsim.json` with exact-card verification and coverage:

```sh
cd ~/dev/cardmodem/pixelqso-source
PYTHONPATH="$HOME/dev/rfsim" python rfsim_sweep.py --backend resilient_100 --channel ssb --profile mpp --snr-db 5 10 15 --seeds 0 1 2
```

**Flux/SoapyFang:** `rfsim.soapyfang.RfSimSource` is an RX-only IQ source wrapper
for existing SoapyFang `IqSource.read(n)` implementations. The integration
is currently in the clean `research/soapyfang-modlab-20261124` worktree
(not merged into Flux main). Source specification:

```text
rfsim:{"source":"synthetic","profile":"mobile-urban","duration_s":120,"seed":7,"snr_db":20}
```

It reads the original source, adds continuous channel taps, shadowing, and
noise before the SoapyFang receiver model. For fractional delays it introduces
a common 32-sample FIR latency to maintain block-size independence. It does
not transmit, and does not replace the BK4819 radio model. Available virtual
sources include synthetic, NPY and SigMF.

## Limitations and attribution

This is a development simulator, not a validated propagation prediction
tool or substitute for real hardware trials. Spectrum/PSD matching,
calibration against external SDR captures, complete modem BER/PER suites,
automatic recording alignment and multi-node network scenarios require
further field evidence. Do not infer compliance, receiver sensitivity,
absolute path loss, or antenna gain from synthetic results.

See upstream [Data2G](https://github.com/arodland/Data2G) and
[SSTVAE](https://github.com/arodland/SSTVAE), both Artistic License 2.0,
for their original simulator sources and RF research notes.

## Test evidence (2026-11-25)
- 23 local unit tests passed, including direct upstream Data2G HF/FM wrapper
  parity, measured capture loading, registry, channel timelines and manifests.
- Python MCP client verified 19 tools, registered profiles and full HF/FM
  simulation; additional recording estimation MCP tool was subsequently added.
- PixelQSO real encode/decode integration: 2 pytest tests passed; three
  manually exercised radio paths, two exact cards and one disturbed NVIS miss.
- SoapyFang receiver branch: 18 pytest tests passed including split-block
  deterministic IQ matching. Nothing transmitted over radio hardware.

## Statistical channel diagnostics

Run channel-spectrum and fading diagnostics:
`rfsim validate --profile mobile-urban --carrier-hz 146000000 --duration-s 60`.
The MCP `validate_channel` tool returns the same data, including RMS Doppler,
the nominal maximum frequency spread, fraction of PSD inside the modeled
band, mean transfer power and fading percentiles. This verifies numerical
behavior of the synthetic model, not empirical path accuracy. A 60-second
VHF urban check yielded 4.18 Hz RMS Doppler, with 98.18% of PSD within the
nominal 5.41 Hz maximum plus FFT-bin tolerance.

## License

RFSim original source is licensed under the [MIT License](LICENSE).
The separately installed Data2G and SSTVAE projects retain their own
Artistic License 2.0 terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
