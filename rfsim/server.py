from dataclasses import asdict
from pathlib import Path
from . import PROFILES, load_profile
from .cli import simulate

def main():
    from mcp.server.fastmcp import FastMCP
    m = FastMCP("rfsim")
    @m.tool()
    def list_profiles() -> list[str]:
        return list(PROFILES)
    @m.tool()
    def get_profile(profile: str) -> list[dict]:
        return [asdict(t) for t in (PROFILES[profile] if profile in PROFILES else load_profile(profile))]
    @m.tool()
    def simulate_iq(input_path: str, output_path: str, fs: float, profile: str="awgn", snr_db: float|None=None, noise_bw_hz: float|None=None, cfo_hz: float=0, clock_ppm: float=0, seed: int=0) -> dict:
        return simulate(input_path,output_path,fs,profile,"iq",snr_db=snr_db,noise_bw_hz=noise_bw_hz,cfo_hz=cfo_hz,clock_ppm=clock_ppm,seed=seed)
    @m.tool()
    def simulate_audio(input_path: str, output_path: str, mode: str="ssb", profile: str="awgn", snr_db: float|None=None, seed: int=0) -> dict:
        return simulate(input_path,output_path,None,profile,mode,snr_db=snr_db,seed=seed)
    @m.tool()
    def run_sweep(input_path: str, output_dir: str, fs: float, profile: str, snrs_db: list[float], seed: int=0) -> list[dict]:
        root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
        return [simulate(input_path,str(root/f"cell-{i}.npy"),fs,profile,"iq",snr_db=s,seed=seed) for i,s in enumerate(snrs_db)]
    @m.tool()
    def inspect_profile_file(path: str) -> list[dict]:
        return [asdict(t) for t in load_profile(path)]
    m.run()
if __name__ == "__main__": main()
