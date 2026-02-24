from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from loguru import logger

def load_config(path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger.info(f"Config loaded: {path}")
    return cfg
