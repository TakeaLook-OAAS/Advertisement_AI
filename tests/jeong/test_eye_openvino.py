# 사진 입력받아서 facial-landmarks-35-adas-0002.xml의 모든 출력 점을 찍음

from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import BBoxXYXY, Track
from src.models.face_openvino import FaceDetector
import os

class EyeDetector:
    def __init__(self, cfg: Dict[str, Any]):
        device_str = cfg.get("device", "CPU")
        weights = cfg.get("weights", "weights/eye_detection/facial-landmarks-35-adas-0002.xml")

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[EyeDetector] weights={weights}  device={device_str}")

    def detect(self, frame: np.ndarray, track: Track) -> list:
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
        for i in range(len(landmarks)):
            if i % 2 == 0:
                landmarks[i] = landmarks[i] * crop_w
            else:
                landmarks[i] = landmarks[i] * crop_h

        return landmarks


INPUT_IMAGE = "data/samples/test_4.jpg"
OUTPUT_DIR = "data/output/"

OUTPUT_IMAGE = "test_4_face.jpg"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 이미지 로드
    frame = cv2.imread(INPUT_IMAGE)
    if frame is None:
        print(f"이미지를 읽을 수 없습니다: {INPUT_IMAGE}")
        return

    h, w = frame.shape[:2]
    print(f"이미지 크기(h, w): {h, w}")

    # 2) face detection
    face_detector = FaceDetector({"device": "CPU"})
    dummy_track = Track(track_id=0, bbox=BBoxXYXY(x1=0, y1=0, x2=w, y2=h))
    track = face_detector.detect(frame, dummy_track)

    if track.crop_bbox is None:
        print("얼굴을 찾지 못했습니다.")
        return

    print(f"얼굴 bbox: {track.crop_bbox}")

    # 얼굴 crop 저장
    cb = track.crop_bbox
    face_crop = frame[cb.y1:cb.y2, cb.x1:cb.x2]

    # 3) eye detection
    eye_detector = EyeDetector({"device": "CPU"})
    landmarks = eye_detector.detect(frame, track)
    
    # 얼굴 crop 위에 모든 랜드마크 점 찍기 (각각 다른 색)
    face_with_pts = face_crop.copy()
    num_points = len(landmarks) // 2
    for i in range(num_points):
        px = landmarks[i * 2]
        py = landmarks[i * 2 + 1]
        # HSV 색상환을 이용해 점마다 고유 색상 생성
        hue = int(180 * i / num_points)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        cv2.circle(face_with_pts, (int(px), int(py)), 3, color, -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_IMAGE), face_with_pts)

    print(f"결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
