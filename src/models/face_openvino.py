from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import BBoxXYXY, Track


class FaceDetector:
    def __init__(self, cfg: Dict[str, Any]):
        device_str = cfg.get("device", "CPU")
        weights = cfg.get("model", "weights/face_detection/face-detection-adas-0001.xml")
        self.conf_thresh = float(cfg.get("conf_thresh", 0.5))
        self.min_face_size = int(cfg.get("min_face_size", 30))  # 최소 얼굴 크기: 30px

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[FaceDetector] model={weights}  device={device_str}  conf={self.conf_thresh}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        person bbox 안에서 얼굴을 찾아 track.crop_bbox를 갱신합니다.
        얼굴을 찾지 못하면 crop_bbox를 person bbox(원래 값) 그대로 유지합니다.
        """
        h, w = frame.shape[:2]
        bbox = track.bbox

        # 1) person bbox를 프레임 안으로 clamp 후 crop
        y1 = max(0, bbox.y1)
        y2 = min(h, bbox.y2)
        x1 = max(0, bbox.x1)
        x2 = min(w, bbox.x2)

        crop_h = y2 - y1
        crop_w = x2 - x1
        if crop_h < self.min_face_size or crop_w < self.min_face_size:
            return track  # crop이 너무 작으면 그대로 반환

        person_crop = frame[y1:y2, x1:x2]

        # 2) face detection 모델 입력 준비 (face-detection-adas-0001: 672x384)
        resized = cv2.resize(person_crop, (672, 384))
        input_image = resized.transpose((2, 0, 1))          # HWC → CHW
        input_image = np.expand_dims(input_image, axis=0)    # 배치 차원 추가

        # 3) 추론
        results = self.compiled_model([input_image])[self.output_layer]

        # 4) 가장 confidence 높은 얼굴 하나 선택
        best_conf = 0.0
        best_box = None
        # detection: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        for detection in results[0][0]:
            confidence = float(detection[2])
            if confidence > self.conf_thresh and confidence > best_conf:
                best_conf = confidence
                # detection[3..6]은 person_crop 기준 비율 좌표(0~1)
                fx1 = int(detection[3] * crop_w)
                fy1 = int(detection[4] * crop_h)
                fx2 = int(detection[5] * crop_w)
                fy2 = int(detection[6] * crop_h)
                best_box = (fx1, fy1, fx2, fy2)

        if best_box is None:
            return track  # 얼굴 못 찾음 → crop_bbox 유지

        fx1, fy1, fx2, fy2 = best_box

        # 5) person_crop 좌표 → 원본 프레임 좌표로 변환
        face_bbox = BBoxXYXY(
            x1=x1 + fx1,
            y1=y1 + fy1,
            x2=x1 + fx2,
            y2=y1 + fy2,
        )

        # 6) track의 crop_bbox를 얼굴 좌표로 갱신
        track.crop_bbox = face_bbox
        return track

    # ------------------------------------------------------------------
    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        """
        여러 track에 대해 얼굴 검출 후 crop_bbox를 갱신합니다.

        Returns
        -------
        List[Track]
            crop_bbox가 갱신된 트랙 리스트 (headpose에 그대로 전달 가능)
        """
        return [self.detect(frame, t) for t in tracks]
