from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from defi_ai.core.config import get_paths
from defi_ai.utils.json_io import atomic_write_json, read_json

app = FastAPI(title="defi-ai-control")

class ForcePair(BaseModel):
    asset1: str
    asset2: str

class ForceModel(BaseModel):
    path: str

class RetrainRequest(BaseModel):
    checkpoint_path: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/control")
def get_control():
    paths = get_paths()
    return read_json(paths.control_file, default={}) or {}

@app.post("/pause")
def pause():
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl["paused"] = True
    atomic_write_json(paths.control_file, ctl)
    return {"paused": True}

@app.post("/resume")
def resume():
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl["paused"] = False
    atomic_write_json(paths.control_file, ctl)
    return {"paused": False}

@app.post("/force_pair")
def force_pair(req: ForcePair):
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl["force_pair"] = {"asset1": req.asset1, "asset2": req.asset2}
    atomic_write_json(paths.control_file, ctl)
    return ctl

@app.post("/clear_force_pair")
def clear_force_pair():
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl.pop("force_pair", None)
    atomic_write_json(paths.control_file, ctl)
    return ctl

@app.post("/force_model")
def force_model(req: ForceModel):
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl["force_model_path"] = req.path
    atomic_write_json(paths.control_file, ctl)
    return ctl

@app.post("/clear_force_model")
def clear_force_model():
    paths = get_paths()
    ctl = read_json(paths.control_file, default={}) or {}
    ctl.pop("force_model_path", None)
    atomic_write_json(paths.control_file, ctl)
    return ctl

@app.post("/trigger_pair_selection")
def trigger_pair_selection(top_k: int = 5, interval: str = "1h"):
    # For production, delegate to the script so it runs as an isolated job boundary.
    try:
        cmd = ["python", "scripts/run_pair_selection.py", "--top-k", str(top_k), "--interval", interval]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise HTTPException(status_code=500, detail={"stderr": p.stderr, "stdout": p.stdout})
        return {"status": "ok", "stdout": p.stdout}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger_retraining")
def trigger_retraining(req: RetrainRequest):
    try:
        cmd = ["python", "scripts/run_retraining.py"]
        if req.checkpoint_path:
            cmd += ["--checkpoint", req.checkpoint_path]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise HTTPException(status_code=500, detail={"stderr": p.stderr, "stdout": p.stdout})
        return {"status": "ok", "stdout": p.stdout}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
