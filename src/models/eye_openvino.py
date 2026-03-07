from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import BBoxXYXY, Track


class EyeDetector:
    def __init__(self, cfg: Dict[str, Any]):
        device_str = cfg.get("device", "CPU")
        weights = cfg.get("model", "weights/eye_detection/facial-landmarks-35-adas-0002.xml")

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[EyeDetector] model={weights}  device={device_str}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        track.crop_bbox를 입력받아서 eye의 좌표를 구하고 track에 넣음
        """
        crop_bbox = track.crop_bbox if track.crop_bbox is not None else track.bbox
        
        crop_h = crop_bbox.h()
        crop_w = crop_bbox.w()

        face_crop = frame[crop_bbox.y1:crop_bbox.y2, crop_bbox.x1:crop_bbox.x2]
        
        resized = cv2.resize(face_crop, (60, 60))
        input_image = resized.transpose((2, 0, 1))  # HWC → CHW
        input_image = np.expand_dims(input_image, axis=0)
        # 입력 shape: 1, 3, 60, 60 (B, C, H, W)

        infer_result = self.compiled_model([input_image])   # OVDict: 레이어가 여러 개일 수 있기 때문에 딕셔너리(키-값)로 반환
        results = infer_result[self.output_layer]

        landmarks = results[0]
        # (facial-landmarks-35-adas-0002 기준)
        # p0(왼쪽 눈 안쪽 끝), p1(왼쪽 눈 바깥쪽 끝)
        # p2(오른쪽 눈 안쪽 끝), p3(오른쪽 눈 바깥쪽 끝)
        lx0 = landmarks[0] * crop_w
        ly0 = landmarks[1] * crop_h
        lx1 = landmarks[2] * crop_w
        ly1 = landmarks[3] * crop_h

        rx2 = landmarks[4] * crop_w
        ry2 = landmarks[5] * crop_h
        rx3 = landmarks[6] * crop_w
        ry3 = landmarks[7] * crop_h

        test_left = (lx0, ly0, lx1, ly1)
        test_right = (rx2, ry2, ry3, ry3)

        logger.info(f"p0={lx0, ly0} p1={lx1, ly1} p2={rx2, ry2} p3={rx3, ry3}")
        # 두 코너 점으로부터 눈 영역 bbox 생성
        left_box = self._eye_bbox(lx0, ly0, lx1, ly1, crop_bbox)
        right_box = self._eye_bbox(rx2, ry2, rx3, ry3, crop_bbox)

        track.left_eye = left_box
        track.right_eye = right_box
        return track, test_left, test_right

    @staticmethod
    def _eye_bbox(x0: float, y0: float, x1: float, y1: float,
                  crop_bbox: BBoxXYXY, margin_ratio: float = 0.15) -> BBoxXYXY:
        """
        두 눈 코너 좌표(crop 내 좌표)로부터 정사각형 bbox를 생성
        """
        # eye_w = abs(x1 - x0)
        # cx = (x0 + x1) / 2
        # cy = (y0 + y1) / 2
        # margin = eye_w / 2 + margin_ratio * eye_w

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        margin = crop_bbox.w() * margin_ratio
        
        return BBoxXYXY(
            x1=int(crop_bbox.x1 + cx - margin),
            y1=int(crop_bbox.y1 + cy - margin),
            x2=int(crop_bbox.x1 + cx + margin),
            y2=int(crop_bbox.y1 + cy + margin),
        )

    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        return [self.detect(frame, t) for t in tracks]


















# https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/facial-landmarks-35-adas-0002/FP32/