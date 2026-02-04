from __future__ import annotations
from typing import Dict, Any
from loguru import logger
import numpy as np

class HeadPose6DRepNet:
    """6DRepNet wrapper (skeleton).

    Replace the `infer()` implementation with your real 6DRepNet inference:
    - load weights
    - preprocess face crop
    - forward pass
    - postprocess to yaw/pitch/roll degrees
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.weights_path = cfg.get("weights_path", "")
        self.device = cfg.get("device", "cpu")
        # TODO: Load actual model here.
        logger.info(f"[HeadPose] enabled={self.enabled} device={self.device} weights={self.weights_path} (stub)")

    def infer(self, frame_bgr) -> Dict[str, float]:
        # TODO: Replace with real inference.
        # Stub: returns near-zero pose.
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
