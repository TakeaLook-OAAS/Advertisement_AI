# infer(left_eye, right_eye, headpose) -> Gaze
# gaze_openvino.py 와 동일한 인터페이스

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

from src.models.gaze.gaze_net import GazeNet
from src.utils.types import Gaze, Track


class GazeDetector:
    _ZERO = Gaze(x=0.0, y=0.0, z=0.0)

    def __init__(self, cfg: Dict[str, Any]) -> None:
        weights = cfg.get("weights", "weights/gaze/gaze_pytorch.pth")
        device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu").lower()
        self.device = torch.device(device_str)

        checkpoint = torch.load(weights, map_location=self.device)
        self.model = GazeNet().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        logger.info(f"[GazeDetector] weights={weights}  device={self.device}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        hp = track.headpose
        if hp is None or (hp.yaw == 0.0 and hp.pitch == 0.0 and hp.roll == 0.0):
            track.gaze = self._ZERO
            return track

        left_eye = track.left_eye if track.left_eye is not None else track.bbox
        right_eye = track.right_eye if track.right_eye is not None else track.bbox

        left_crop = frame[left_eye.y1:left_eye.y2, left_eye.x1:left_eye.x2]
        right_crop = frame[right_eye.y1:right_eye.y2, right_eye.x1:right_eye.x2]

        if left_crop.size == 0 or right_crop.size == 0:
            track.gaze = self._ZERO
            return track

        size = GazeNet.EYE_SIZE
        left_t = self._to_tensor(cv2.resize(left_crop, (size, size)))
        right_t = self._to_tensor(cv2.resize(right_crop, (size, size)))
        hp_t = torch.tensor(
            [[hp.yaw / 90.0, hp.pitch / 90.0, hp.roll / 90.0]],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            gaze = self.model(left_t, right_t, hp_t)[0]

        track.gaze = Gaze(x=float(gaze[0]), y=float(gaze[1]), z=float(gaze[2]))
        return track

    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        return [self.detect(frame, t) for t in tracks]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 BGR → (1, 3, H, W) float32 [0,1]"""
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=self.device)
        return t.unsqueeze(0) / 255.0
