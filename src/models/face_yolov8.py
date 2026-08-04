from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
from loguru import logger
from src.utils.types import BBoxXYXY, Track


class FaceDetector:
    """
    YOLOv8-face (WIDER FACE 학습) 기반 얼굴 검출기.
    face_openvino.py의 FaceDetector와 동일한 인터페이스(detect/detect_batch)를 제공합니다.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        weights = cfg.get("weights", "weights/face_detection/yolov8n-face-lindevs.pt")
        self.conf_thresh = float(cfg.get("conf_thresh", 0.5))
        self.iou_thresh = float(cfg.get("iou_thresh", 0.45))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.min_face_size = int(cfg.get("min_face_size", 30))

        from ultralytics import YOLO  # type: ignore
        self.model = YOLO(weights)
        self.model.to(self.device)
        logger.info(f"[FaceDetector(YOLOv8-face)] weights={weights}  device={self.device}  conf={self.conf_thresh}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        person bbox 안에서 얼굴을 찾아 track.crop_bbox를 갱신합니다.
        얼굴을 찾지 못하면 crop_bbox는 기본값 None으로 남습니다.
        """
        bbox = track.bbox

        h = bbox.h()
        w = bbox.w()
        if h < self.min_face_size or w < self.min_face_size:
            return track  # person bbox가 너무 작으면 crop_bbox=None으로 반환

        person_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        results = self.model(
            person_crop,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return track  # 얼굴 못 찾음 → crop_bbox=None으로 반환

        boxes = results[0].boxes.xyxy.cpu().numpy()             # (N, 4), person_crop 기준 픽셀 좌표
        scores = results[0].boxes.conf.cpu().numpy().reshape(-1)  # (N,)

        best_idx = int(np.argmax(scores))
        fx1, fy1, fx2, fy2 = boxes[best_idx]

        # person_crop 좌표 → 원본 프레임 좌표로 변환 (프레임 범위로)
        fh, fw = frame.shape[:2]
        face_bbox = BBoxXYXY(
            x1=max(0, min(bbox.x1 + int(fx1), fw)),
            y1=max(0, min(bbox.y1 + int(fy1), fh)),
            x2=max(0, min(bbox.x1 + int(fx2), fw)),
            y2=max(0, min(bbox.y1 + int(fy2), fh)),
        )

        track.crop_bbox = face_bbox
        return track

    # ------------------------------------------------------------------
    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        """
        여러 track에 대해 얼굴 검출 후 crop_bbox를 갱신합니다.
        person crop들을 모아 한 번의 배치 추론으로 처리합니다.

        Returns
        -------
        List[Track]
            crop_bbox가 갱신된 트랙 리스트
        """
        valid_tracks: List[Track] = []
        crops: List[np.ndarray] = []
        for track in tracks:
            bbox = track.bbox
            if bbox.h() < self.min_face_size or bbox.w() < self.min_face_size:
                continue
            person_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if person_crop.size == 0:
                continue
            valid_tracks.append(track)
            crops.append(person_crop)

        if not crops:
            return tracks

        results = self.model(
            crops,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        fh, fw = frame.shape[:2]
        for track, result in zip(valid_tracks, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy().reshape(-1)

            best_idx = int(np.argmax(scores))
            fx1, fy1, fx2, fy2 = boxes[best_idx]

            bbox = track.bbox
            track.crop_bbox = BBoxXYXY(
                x1=max(0, min(bbox.x1 + int(fx1), fw)),
                y1=max(0, min(bbox.y1 + int(fy1), fh)),
                x2=max(0, min(bbox.x1 + int(fx2), fw)),
                y2=max(0, min(bbox.y1 + int(fy2), fh)),
            )

        return tracks


# https://github.com/lindevs/yolov8-face
