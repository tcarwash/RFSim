"""Portable user profile registry; copies data only, never executes plugins."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
import re
import shutil

def registry_dir() -> Path:
    return Path.home()/".local"/"share"/"rfsim"/"profiles"

def registered_path(name: str) -> Path:
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",name):
        raise ValueError("registry name must be a short ASCII slug")
    return registry_dir()/(name+".json")

def list_imported_profiles() -> list[dict]:
    root=registry_dir()
    if not root.exists(): return []
    out=[]
    for p in sorted(root.glob("*")):
        if p.suffix not in (".json",".npz") or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",p.stem):
            continue
        out.append(dict(name=p.stem,kind="measured" if p.suffix==".npz" else "taps",path=str(p)))
    return out

def import_profile(path: str, *, name: str|None=None) -> dict:
    src=Path(path).expanduser().resolve()
    if not src.is_file() or src.suffix not in (".json",".npz"):
        raise ValueError("profile must be a .json or .npz file")
    name=src.stem if name is None else name
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",name):
        raise ValueError("invalid profile name")
    from . import PROFILES
    from .session import PATHS
    if name in PROFILES or name in PATHS:
        raise ValueError("profile name conflicts with a built-in model")
    if src.suffix==".json":
        from . import load_profile
        load_profile(src)
    else:
        from .measured import MeasuredChannel
        MeasuredChannel(src)
    root=registry_dir()
    root.mkdir(parents=True,exist_ok=True)
    dst=root/(name+src.suffix)
    h=sha256(src.read_bytes()).hexdigest()
    if dst.exists():
        if sha256(dst.read_bytes()).hexdigest()!=h:
            raise FileExistsError(f"registered profile {name} already exists and differs")
    else: shutil.copyfile(src,dst)
    return dict(name=name,kind="measured" if src.suffix==".npz" else "taps",
                path=str(dst),sha256=h)

def resolve_profile_path(name: str):
    p=Path(name).expanduser()
    if p.is_file(): return p
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",name):
        raise FileNotFoundError(name)
    p=registered_path(name)
    if not p.is_file(): raise FileNotFoundError(name)
    return p
