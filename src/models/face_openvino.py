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
        weights = cfg.get("weights", "weights/face_detection/face-detection-adas-0001.xml")
        self.conf_thresh = float(cfg.get("conf_thresh", 0.5))
        self.min_face_size = int(cfg.get("min_face_size", 30))  # 최소 얼굴 크기: 30px -> AP(head height > 32px): 84.8%

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[FaceDetector] weights={weights}  device={device_str}  conf={self.conf_thresh}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        person bbox 안에서 얼굴을 찾아 track.crop_bbox를 갱신합니다.
        얼굴을 찾지 못하면 crop_bbox를 person bbox(원래 값) 그대로 유지합니다.
        """
        bbox = track.bbox

        h = bbox.h()
        w = bbox.w()
        if h < self.min_face_size or w < self.min_face_size:
            return track  # crop이 너무 작으면 그대로 반환

        person_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        # 2) face detection 모델 입력 준비 (face-detection-adas-0001: 672x384)
        resized = cv2.resize(person_crop, (672, 384))
        input_image = resized.transpose((2, 0, 1))          # HWC → CHW
        input_image = np.expand_dims(input_image, axis=0)   # 배치 차원 추가
        # 입력 shape: 1, 3, 384, 672 (batch size, number of channels,  image height, image width)
        
        # 3) 추론
        infer_result = self.compiled_model([input_image])   # OVDict: 레이어가 여러 개일 수 있기 때문에 딕셔너리(키-값)로 반환
        results = infer_result[self.output_layer]
        # face-detection-adas-0001은 출력 레이어가 1개라 results = self.compiled_model([input_image])[0] 이래도 됨
        # 출력 shape: 1, 1, N, 7

        # 4) N개 중 confidence 가장 높은 얼굴 하나 선택
        best_conf = 0.0
        best_box = None
        # detection: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        for detection in results[0][0]:
            confidence = float(detection[2])
            if confidence > self.conf_thresh and confidence > best_conf:
                best_conf = confidence
                # x_min, y_min, x_max, y_max은 person_crop 기준 0~1 사이의 비율
                fx1 = int(detection[3] * w)
                fy1 = int(detection[4] * h)
                fx2 = int(detection[5] * w)
                fy2 = int(detection[6] * h)
                best_box = (fx1, fy1, fx2, fy2)

        if best_box is None:
            return track  # 얼굴 못 찾음 → crop_bbox 유지

        # 5) person_crop 좌표 → 원본 프레임 좌표로 변환 (프레임 범위로)
        fh, fw = frame.shape[:2]
        face_bbox = BBoxXYXY(
            x1=max(0, min(bbox.x1 + fx1, fw)),
            y1=max(0, min(bbox.y1 + fy1, fh)),
            x2=max(0, min(bbox.x1 + fx2, fw)),
            y2=max(0, min(bbox.y1 + fy2, fh)),
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
            crop_bbox가 갱신된 트랙 리스트
        """
        return [self.detect(frame, t) for t in tracks]



# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-adas-0001
# https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/