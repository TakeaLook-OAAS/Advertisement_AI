from __future__ import annotations
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np

try:
    from openvino.runtime import Core
except Exception:  # pragma: no cover
    Core = None

class GazeOpenVINO:
    """OpenVINO gaze wrapper (skeleton).

    Replace `infer()` with your actual pipeline:
    - load IR (xml/bin)
    - prepare input tensors
    - run compiled_model(...)
    - parse gaze vector output
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.model_xml = cfg.get("model_xml", "")
        self.device = cfg.get("device", "CPU")

        self.core = None
        self.compiled = None

        if Core is None:
            logger.warning("[Gaze] OpenVINO not available in this environment (import failed). Running stub.")
            return

        # Attempt to load if file exists; otherwise stub.
        try:
            self.core = Core()
            if self.model_xml and self.model_xml.endswith(".xml"):
                # It's okay if model files are not present yet; we keep it stub until you add them.
                import os
                if os.path.exists(self.model_xml):
                    model = self.core.read_model(self.model_xml)
                    self.compiled = self.core.compile_model(model, self.device)
                    logger.info(f"[Gaze] Loaded OpenVINO model: {self.model_xml} on {self.device}")
                else:
                    logger.info(f"[Gaze] IR not found at {self.model_xml} (stub until you add files).")
        except Exception as e:
            logger.warning(f"[Gaze] Failed to init OpenVINO ({e}). Running stub.")

    def infer(self, frame_bgr) -> np.ndarray:
        # TODO: Replace with real inference.
        # Stub: forward-facing vector (z+)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
