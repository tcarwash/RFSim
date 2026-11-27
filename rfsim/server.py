"""Local trusted-agent MCP stdio server for RFSim; waveform payloads are files."""
from dataclasses import asdict
from pathlib import Path

def main():
    from mcp.server.fastmcp import FastMCP
    from . import PROFILES, load_profile
    from .session import PATHS, ChannelSession
    from .measured import estimate, save_trace, MeasuredChannel
    from .experiments import run_experiment, list_reports, read_report
    from .cli import simulate
    import numpy as np
    import json

    m=FastMCP("rfsim")

    from .registry import import_profile, list_imported_profiles

    @m.tool()
    def register_profile(profile_path: str, name: str|None=None) -> dict:
        """Import a validated parametric JSON or measured NPZ into the user registry."""
        return import_profile(profile_path,name=name)

    @m.tool()
    def list_imported_profiles() -> list[dict]:
        return list_imported_profiles()

    @m.tool()
    def list_profiles() -> list[str]:
        """Built-in burst IQ and audio channel models."""
        return sorted(set(PROFILES)|{v["name"] for v in list_imported_profiles() if v["kind"]=="taps"})

    @m.tool()
    def get_profile(profile: str) -> list[dict]:
        """Inspect known or custom JSON tap profiles."""
        return [asdict(t) for t in (PROFILES[profile] if profile in PROFILES else load_profile(profile))]

    @m.tool()
    def list_session_profiles() -> list[str]:
        """Time-persistent propagation models including mobile FM conditions."""
        return sorted(set(PROFILES)|set(PATHS)|{v["name"] for v in list_imported_profiles() if v["kind"]=="taps"})

    @m.tool()
    def simulate_iq(input_path: str, output_path: str, fs: float,
                    profile: str="awgn", snr_db: float|None=None,
                    noise_bw_hz: float|None=None,cfo_hz: float=0,
                    clock_ppm: float=0, seed: int=0) -> dict:
        return simulate(input_path,output_path,fs,profile,"iq",snr_db=snr_db,
                        noise_bw_hz=noise_bw_hz,cfo_hz=cfo_hz,clock_ppm=clock_ppm,seed=seed)

    @m.tool()
    def simulate_audio(input_path: str, output_path: str,mode: str="ssb",
                       profile: str="awgn",snr_db: float|None=None,seed: int=0) -> dict:
        return simulate(input_path,output_path,None,profile,mode,snr_db=snr_db,seed=seed)

    @m.tool()
    def run_sweep(input_path: str,output_dir: str,fs: float,profile: str,
                  snrs_db: list[float],seed: int=0) -> list[dict]:
        root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
        return [simulate(input_path,str(root/f"cell-{i}.npy"),fs,profile,"iq",
                         snr_db=s,seed=seed) for i,s in enumerate(snrs_db)]

    @m.tool()
    def inspect_profile_file(path: str) -> list[dict]:
        return [asdict(t) for t in load_profile(path)]

    @m.tool()
    def simulate_session(input_path: str,output_path: str,fs: float,profile: str,
                         duration_s: float=120,start_s: float=0,carrier_hz: float=146e6,
                         seed: int=0,snr_db: float|None=None,noise_seed: int=0,
                         noise_bw_hz: float|None=None) -> dict:
        s=ChannelSession(fs,profile,duration_s=duration_s,carrier_hz=carrier_hz,seed=seed)
        y=s.apply(np.load(input_path,allow_pickle=False),start_s=start_s,
                  snr_db=snr_db,noise_bw_hz=noise_bw_hz,noise_seed=noise_seed)
        with open(output_path,"wb") as f:np.save(f,y)
        result=dict(output=output_path,samples=len(y),sample_rate=fs,profile=profile,
                    duration_s=duration_s,start_s=start_s,carrier_hz=carrier_hz,
                    seed=seed,noise_seed=noise_seed,snr_db=snr_db,noise_bw_hz=noise_bw_hz)
        Path(output_path+".json").write_text(json.dumps(result,indent=2)+"\n")
        return result

    @m.tool()
    def estimate_profile(tx_path: str,rx_path: str,output_path: str,fs: float,
                         provenance: dict,taps: int=8,window: int=2048,
                         hop: int|None=None) -> dict:
        """Estimate synchronized IQ taps; do not supply arbitrary, unaligned captures."""
        t,d,g,stats=estimate(np.load(tx_path,allow_pickle=False),
                             np.load(rx_path,allow_pickle=False),fs,
                             taps=taps,window=window,hop=hop)
        save_trace(output_path,t,d,g,
                   {**provenance,"sample_rate":fs,"tx":tx_path,"rx":rx_path,
                    "window":window,"hop":hop,"taps":taps})
        return dict(output=output_path,start_s=float(t[0]),end_s=float(t[-1]),
                    snapshots=len(t),max_condition=max(stats["condition"]),
                    mean_residual_power=float(np.mean(stats["residual_power"])))

    @m.tool()
    def estimate_capture_profile(tx_capture: str, rx_capture: str, output_path: str,
                                 provenance: dict, fs: float|None=None,
                                 taps: int=8, window: int=2048,
                                 hop: int|None=None) -> dict:
        """Estimate from synchronized .npy or SigMF IQ recordings (no auto alignment)."""
        from .capture import load_iq_capture
        tx,rate,tmeta=load_iq_capture(tx_capture,fs=fs)
        rx,rrate,rmeta=load_iq_capture(rx_capture,fs=fs)
        if abs(rate-rrate)>1e-6:raise ValueError("capture sample rates differ")
        t,d,g,stats=estimate(tx,rx,rate,taps=taps,window=window,hop=hop)
        save_trace(output_path,t,d,g,{**provenance,"tx_capture":tmeta,"rx_capture":rmeta,
                                      "window":window,"hop":hop,"taps":taps})
        return dict(output=output_path,sample_rate=rate,start_s=float(t[0]),
                    end_s=float(t[-1]),snapshots=len(t),
                    mean_residual_power=float(np.mean(stats["residual_power"])),
                    max_condition=float(max(stats["condition"])))

    @m.tool()
    def inspect_measured_profile(path: str) -> dict:
        c=MeasuredChannel(path)
        return dict(provenance=c.provenance,start_s=float(c.times[0]),
                    end_s=float(c.times[-1]),delays_s=c.delays.tolist(),
                    mean_tap_power=np.mean(abs(c.gains)**2,axis=0).tolist())

    @m.tool()
    def replay_measured_profile(input_path: str,output_path: str,
                                profile_path: str,fs: float,start_s: float|None=None) -> dict:
        c=MeasuredChannel(profile_path)
        y=c.apply(np.load(input_path,allow_pickle=False),fs,start_s=start_s)
        with open(output_path,"wb") as f:np.save(f,y)
        return dict(output=output_path,samples=len(y),sample_rate=fs,
                    profile=profile_path,start_s=start_s)

    @m.tool()
    def run_manifest(manifest_path: str) -> dict:
        """Run deterministic offline trials, saving each output plus report.json."""
        return run_experiment(manifest_path)

    @m.tool()
    def list_experiment_reports(directory: str) -> list[str]:
        return list_reports(directory)

    @m.tool()
    def get_experiment_report(path: str) -> dict:
        return read_report(path)

    @m.tool()
    def get_k6_gp0_microphone_profile() -> dict:
        """Measured K6 microphone-to-PCM calibration and its uncertainty."""
        from .k6_mic import k6_gp0_profile
        return k6_gp0_profile()

    @m.tool()
    def simulate_k6_gp0_microphone(input_path: str, output_path: str,
                                   seed: int=0, input_vpp: float=0.1,
                                   response: str="measured", noise_model: str="banded") -> dict:
        """Apply K6 source noise BEFORE encoding, independent of RF channel.

        Input: mono 8 kHz WAV (int16 or float), amplitude 1 = supplied input_vpp.
        Output: new 8 kHz float32 WAV. Never operates the radio or overwrites input.
        """
        from scipy.io import wavfile
        from .k6_mic import k6_gp0_pcm, k6_gp0_profile
        fs, audio = wavfile.read(input_path)
        if audio.ndim != 1 or audio.dtype not in (np.dtype('int16'),np.dtype('float32'),np.dtype('float64')):
            raise ValueError("expected mono float or int16 WAV")
        data = audio.astype(float) / 32768 if audio.dtype == np.dtype('int16') else audio.astype(float)
        result = k6_gp0_pcm(data, fs, seed=seed, input_vpp=input_vpp,
                            response=response, noise_model=noise_model)
        if Path(output_path).resolve() == Path(input_path).resolve():
            raise ValueError("output must not overwrite input")
        with Path(output_path).open('xb') as f:
            from io import BytesIO
            stream=BytesIO();wavfile.write(stream,fs,result.astype(np.float32));f.write(stream.getvalue())
        return dict(output=output_path,samples=len(result),sample_rate=fs,
                    input_vpp=input_vpp,seed=seed,response=response,noise_model=noise_model,
                    noise_rms_fs=k6_gp0_profile()['physical_generator_off_pcm_rms_full_scale'],
                    stage='microphone to PCM, BEFORE codec/RF')

    @m.tool()
    def simulate_data2g_fm(input_path: str,output_path: str,condition: str="static",
                           radio: str="mic",cnr_db: float|None=30,seed: int=0,
                           carrier_mhz: float=146.,duration_s: float=120,
                           start_s: float=0) -> dict:
        from scipy.io import wavfile
        from .experiments import _wav
        from .data2g_bridge import fm_audio
        fs,x=_wav(input_path)
        y=fm_audio(x,fs,condition=condition,radio=radio,cnr_db=cnr_db,seed=seed,
                   carrier_mhz=carrier_mhz,duration_s=duration_s,start_s=start_s)
        wavfile.write(output_path,int(fs),np.asarray(y,dtype=np.float32))
        return dict(output=output_path,samples=len(y),sample_rate=fs,model="Data2G full FM")

    @m.tool()
    def simulate_data2g_hf(input_path: str, output_path: str, preset: str="mpp",
                           snr_db: float|None=None,seed: int=0,
                           cfo_hz: float=0,clock_ppm: float=0) -> dict:
        from scipy.io import wavfile
        from .experiments import _wav
        from .data2g_bridge import hf_audio
        fs,x=_wav(input_path)
        y=hf_audio(x,fs,preset=preset,snr_db=snr_db,seed=seed,
                   cfo_hz=cfo_hz,clock_ppm=clock_ppm)
        wavfile.write(output_path,int(fs),np.asarray(y,dtype=np.float32))
        return dict(output=output_path,samples=len(y),sample_rate=fs,model="Data2G HF")
    @m.tool()
    def validate_channel(profile: str="mobile-urban", fs: float=48000,
                         duration_s: float=60, carrier_hz: float=146e6,
                         seed: int=0) -> dict:
        """Inspect fading power, Doppler PSD and model spectral support."""
        from .validation import analyze_profile
        return analyze_profile(profile,fs=fs,duration_s=duration_s,
                               carrier_hz=carrier_hz,seed=seed)
    m.run()

if __name__=="__main__":
    main()
