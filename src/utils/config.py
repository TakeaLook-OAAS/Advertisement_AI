from __future__ import annotations
from typing import Dict, Any
import yaml
import os

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inc = cfg.get("include")
    if inc:
        with open(inc, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)
        cfg.pop("include", None)

    # environment overrides example (optional)
    # e.g. export HEADPOSE_WEIGHTS=...
    hp_w = os.environ.get("HEADPOSE_WEIGHTS")
    if hp_w:
        cfg.setdefault("models", {}).setdefault("headpose", {})["weights_path"] = hp_w

    gaze_xml = os.environ.get("GAZE_MODEL_XML")
    if gaze_xml:
        cfg.setdefault("models", {}).setdefault("gaze", {})["model_xml"] = gaze_xml

    return cfg
