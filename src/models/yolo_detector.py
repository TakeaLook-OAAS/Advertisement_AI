"""
Ultralytics 패키지를 사용하는 YOLO 검출기 래퍼.

아래 상황에서는 빈 검출 결과를 반환:
  - `ultralytics`가 설치되지 않은 경우
  - 모델 파일을 찾을 수 없는 경우
  - 설정에서 검출기가 비활성화된 경우

설정 키 (models.yolo 하위):
  enabled       bool   true
  model_path    str    "yolov8n.pt"   (최초 실행 시 자동 다운로드)
  device        str    "cpu"
  conf_thresh   float  0.25
  iou_thresh    float  0.45           (NMS IoU)
  classes       list   [0]            COCO 클래스 ID (0=사람)
  imgsz         int    640
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger

from src.utils.types import BBoxXYXY, Det


class YoloDetector:

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled     = bool(cfg.get("enabled", True))
        self.model_path  = cfg.get("model_path", "yolov8n.pt")
        self.device      = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = float(cfg.get("conf_thresh", 0.25))
        self.iou_thresh  = float(cfg.get("iou_thresh",  0.45))
        self.classes: List[int] = cfg.get("classes", [0])   # 0 = 사람
        self.imgsz       = int(cfg.get("imgsz", 640))

        self.model = None

        if not self.enabled:
            logger.info("[YOLO] 비활성화")
            return

        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # 워밍업
            self.model(
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
                device=self.device,
                verbose=False,
            )
            logger.info(
                f"[YOLO] model={self.model_path}  device={self.device}  "
                f"conf={self.conf_thresh}  classes={self.classes}"
            )
        except ImportError:
            logger.warning(
                "[YOLO] `ultralytics` 미설치 → 검출 불가. "
                "설치 명령:  pip install ultralytics"
            )
        except Exception as exc:
            logger.warning(f"[YOLO] 모델 로드 실패: {exc} → 스텁 모드")

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> List[Det]:
        """
        BGR 프레임에서 검출 수행 후 List[Det]으로 반환.

        반환값
        ------
        List[Det]   검출된 객체 리스트 (bbox + cls + conf)
                    검출 없을 시 빈 리스트 반환.
        """
        if not self.enabled or self.model is None:
            return []

        results = self.model(
            frame_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes  = results[0].boxes.xyxy.cpu().numpy()                # (N, 4)
        scores = results[0].boxes.conf.cpu().numpy().reshape(-1)    # (N,)

        return [
            Det(
                bbox=BBoxXYXY(
                    x1=int(boxes[i, 0]),
                    y1=int(boxes[i, 1]),
                    x2=int(boxes[i, 2]),
                    y2=int(boxes[i, 3]),
                ),
                cls=0,  # YOLO classes 필터가 이미 적용됨
                conf=float(scores[i]),
            )
            for i in range(len(scores))
        ]
