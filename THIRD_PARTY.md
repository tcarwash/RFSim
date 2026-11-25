# Upstream RF simulation research and optional dependencies

RFSim's first implementation was informed by the openly available source
and experiments in:

- [arodland/Data2G](https://github.com/arodland/Data2G), revision
  `b278852cbc03d32d3c865e4208c17d2700806f57`.
  In particular `data2g/hfchannel.py`, `data2g/fmchannel.py`,
  `data2g/channel_torch.py` and their regression studies.
- [arodland/SSTVAE](https://github.com/arodland/SSTVAE), revision
  `e18700bf8be8f6a13657c9f449d4c55876847568`,
  especially `sstvae/hfchannel.py` and `sstvae/waveform_channel.py`.

Both upstream repositories carry the **Artistic License 2.0**. RFSim's
`data2g_bridge.py` dynamically imports the upstream Data2G package for
full parity with its HF and FM channel models; Data2G itself is never
vendored or redistributed by RFSim. Users should retain upstream
license notices and follow upstream license terms when installing or
distributing Data2G.

RFSim's standalone NumPy channel engine, session model, measured profile
format, experiment runner, MCP interface and adapters are separate
implementations. Its included HF and FM propagation *parameters* are
documented approximations, not claims of independent radio-path
measurements.

The Flux/SoapyFang source integration is maintained in that project's
research branch, which already includes its own upstream MIT
attribution. The PixelQSO decoder and radio-chip models remain in
their respective original projects.
