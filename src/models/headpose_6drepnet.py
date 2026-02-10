from __future__ import annotations
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np
import cv2

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

        if not self.enabled:
            logger.info("[HeadPose] disabled")
            self.model = None
            return

        # (A) face detector: simple & fast
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # (B) model loader (sixdrepnet)
        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()

            # optional: load your custom pth if you want
            if self.weights_path:
                import os, torch
                if os.path.exists(self.weights_path):
                    sd = torch.load(self.weights_path, map_location=self.device)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    self.model.load_state_dict(sd, strict=False)
                    logger.info(f"[HeadPose] loaded weights: {self.weights_path}")
                else:
                    logger.warning(f"[HeadPose] weights not found: {self.weights_path} (using auto weights)")
            logger.info(f"[HeadPose] enabled=True device={self.device}")
        except Exception as e:
            logger.exception(f"[HeadPose] failed to init model: {e}")
            self.model = None

    def _detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return cropped face image (BGR) or None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        # small margin so chin/forehead not clipped
        m = int(0.15 * max(w, h))
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(frame_bgr.shape[1], x + w + m)
        y1 = min(frame_bgr.shape[0], y + h + m)
        return frame_bgr[y0:y1, x0:x1]

    def infer(self, frame_bgr) -> Dict[str, float]:
        if not self.enabled or self.model is None:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        face_bgr = self._detect_largest_face(frame_bgr)
        if face_bgr is None:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # sixdrepnet expects RGB np.ndarray
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        yaw, pitch, roll = self.model.predict(face_rgb)  # degrees
        return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}


    #def infer(self, frame_bgr) -> Dict[str, float]:
        # TODO: Replace with real inference.
        # Stub: returns near-zero pose.
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
