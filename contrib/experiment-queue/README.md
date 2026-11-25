# RFSim as DORC's reusable channel test

RFSim is a shared *simulation backend*, not just a one-off experiment.
The queue runs the pinned source version's
`experiments/dorc_channel_suite.py`, with the test matrix pinned separately
in DORC at `fixtures/rfsim/common_suite.json`.

The suite generates a reproducible complex waveform and runs multiple
built-in HF, VHF, and UHF profiles over a configurable SNR × seed grid.
It checks deterministic replay, finite/nonzero output, and AWGN SNR
accuracy, publishing `report.md`, `metrics.json`, and `cases.csv`. Failure
of a *numerical hypothesis* is reported as a valid `NEGATIVE_FINDING`
rather than misclassified as an infrastructure failure.

The normal run is CPU-only. The Dockerfile pins Python/NumPy/SciPy
dependencies, and the contributor allowlists the local **immutable image ID**.
The experiment itself is executed with networking disabled, pinned
RFSim source read-only, the DORC matrix read-only, and outputs writable.
The container has no need for radio hardware or GPUs.

For future modem experiments, use this suite's profile / SNR / seed
matrix as a common baseline and add decoder-specific BER/PER or
packet-completion evidence. **Raw waveform mismatch is not BER/PER**,
nor does the synthetic simulator establish over-the-air channel accuracy.

Local example, from the RFSim checkout:

```sh
python3 -m experiments.dorc_channel_suite \
  --config /path/to/dorc_experiments/fixtures/rfsim/common_suite.json \
  --output /tmp/dorc-rfsim-common-suite
```

To submit via the distributed queue, use source URL
`https://github.com/tcarwash/RFSim.git` pinned to its full pushed SHA,
with `inputs` referencing the DORC matrix as `repository:"self"`,
also pinned to its own full pushed SHA. See the DORC repository's
`AGENTS.md` and `docs/RFSIM_COMMON_SUITE.md`.
