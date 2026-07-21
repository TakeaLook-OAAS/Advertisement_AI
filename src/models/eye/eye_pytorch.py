# infer(frame, track) -> track.left_eye, track.right_eye
# eye_openvino.py 와 동일한 인터페이스 (facial-landmarks-35-adas-0002 대체)
# WFLW로 학습: 눈마다 (x1,y1,x2,y2) bbox를 face crop 기준 정규화 좌표로 직접 회귀

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

from src.models.eye.eye_net import EyeNet
from src.utils.types import BBoxXYXY, Track


class EyeDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        weights = cfg.get("weights", "weights/eye_detection/eye_pytorch.pth")
        device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu").lower()
        self.device = torch.device(device_str)
        self.min_eye_size = int(cfg.get("min_eye_size", 10))

        checkpoint = torch.load(weights, map_location=self.device)
        self.model = EyeNet().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        logger.info(f"[EyeDetector(pytorch)] weights={weights}  device={self.device}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        track.crop_bbox를 입력받아서 eye의 좌표를 구하고 track에 넣음
        실패 시 left_eye, right_eye를 None으로 설정
        """
        if track.crop_bbox is None:
            track.left_eye = None
            track.right_eye = None
            return track

        crop_bbox = track.crop_bbox
        crop_h = crop_bbox.h()
        crop_w = crop_bbox.w()

        if crop_h < self.min_eye_size or crop_w < self.min_eye_size:
            track.left_eye = None
            track.right_eye = None
            return track

        face_crop = frame[crop_bbox.y1:crop_bbox.y2, crop_bbox.x1:crop_bbox.x2]
        if face_crop.size == 0:
            track.left_eye = None
            track.right_eye = None
            return track

        size = EyeNet.FACE_SIZE
        resized = cv2.resize(face_crop, (size, size))
        input_t = self._to_tensor(resized)

        with torch.no_grad():
            boxes = self.model(input_t)[0].cpu().numpy().reshape(2, 4)  # [left, right] x [x1,y1,x2,y2]

        track.left_eye = self._to_bbox(boxes[0], crop_bbox, crop_w, crop_h)
        track.right_eye = self._to_bbox(boxes[1], crop_bbox, crop_w, crop_h)
        return track

    @staticmethod
    def _to_bbox(box: np.ndarray, crop_bbox: BBoxXYXY, crop_w: int, crop_h: int) -> BBoxXYXY:
        x1, y1, x2, y2 = box
        return BBoxXYXY(
            x1=int(crop_bbox.x1 + x1 * crop_w),
            y1=int(crop_bbox.y1 + y1 * crop_h),
            x2=int(crop_bbox.x1 + x2 * crop_w),
            y2=int(crop_bbox.y1 + y2 * crop_h),
        )

    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        return [self.detect(frame, t) for t in tracks]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 BGR → (1, 3, H, W) float32 [0,1]"""
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=self.device)
        return t.unsqueeze(0) / 255.0
