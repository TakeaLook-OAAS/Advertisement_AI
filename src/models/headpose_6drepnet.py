# infer(frame, track)->HeadPose

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from src.utils.types import HeadPose, Track


class HeadPoseEstimator:
    """
    6DRepNet 어댑터:
      infer(frame, track) -> Optional[HeadPose]

    sixdrepnet 패키지를 사용하여 머리 자세(yaw/pitch/roll)를 추정합니다.
    person bbox에서 얼굴을 찾아 각도를 반환하며,
    얼굴을 찾지 못하거나 crop이 너무 작으면 None을 반환합니다.
    """

    def __init__(self, cfg: Dict[str, Any]):
        from sixdrepnet import SixDRepNet

        device_str = cfg.get("device", "cpu")
        gpu_id = -1 if device_str == "cpu" else 0   # cpu면 -1, gpu면 0
        self.min_face_size = int(cfg.get("min_face_size", 30))  # 최소 얼굴 크기: 30px

        weights = cfg.get("weights", "models/headpose/weights/6DRepNet_300W_LP_AFLW2000.pth")
        logger.info(f"Loading 6DRepNet model (gpu_id={gpu_id}, weights='{weights or 'auto'}')")
        self.model = SixDRepNet(gpu_id=gpu_id, dict_path=weights)   # 가중치 다운로드

    def infer(self, frame, track: Track) -> Tuple[Optional[HeadPose], Optional[str]]:
        """
        Returns:
            (HeadPose, None) on success
            (None, fail_reason) on failure
        """
        bbox = track.bbox
        h, w = frame.shape[:2]  # 프레임의 높이와 너비-> ex: frame.shape = (720, 1280, 3) // frame.shape[:2] = (720, 1280)

        # bbox 경계를 프레임 안으로 clamp
        y1 = max(0, bbox.y1)
        y2 = min(h, bbox.y2)
        x1 = max(0, bbox.x1)
        x2 = min(w, bbox.x2)

        crop_h = y2 - y1
        crop_w = x2 - x1

        if crop_h < self.min_face_size or crop_w < self.min_face_size:
            return None, "face_too_small"

        crop = frame[y1:y2, x1:x2]

        try:
            results = self.model.predict(crop)      # 6DRepNet에 crop 이미지 넣기
        except Exception as e:
            logger.debug(f"6DRepNet predict error: {e}")
            return None, "model_error"

        if results is None or len(results) == 0:    # 얼굴 못 찾음
            return None, "no_face"

        # sixdrepnet predict() 반환: (pitch_array, yaw_array, roll_array)
        # 각각 numpy array([value], dtype=float32)
        if len(results) == 3:
            pitch = float(results[0])
            yaw = float(results[1])
            roll = float(results[2])
        else:
            return None, "parse_error"

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll), None

    def infer_batch(
        self, frame, tracks: List[Track]
        ) -> List[Tuple[int, Optional[HeadPose], Optional[str]]]:
        """
        여러 track에 대해 한 번에 headpose를 추정합니다.

        Returns:
            List of (track_id, headpose_or_none, fail_reason_or_none)
        """
        results = []
        for t in tracks:
            hp, reason = self.infer(frame, t)
            results.append((t.track_id, hp, reason))
        return results