# infer(frame, track)->HeadPose

from __future__ import annotations
from typing import Any, Dict, List
from loguru import logger
from src.utils.types import HeadPose, Track


class HeadPoseEstimator:
    """
    6DRepNet 어댑터:
      infer(frame, track) -> HeadPose

    sixdrepnet 패키지를 사용하여 머리 자세(yaw/pitch/roll)를 추정합니다.
    person bbox에서 얼굴을 찾아 각도를 반환하며,
    얼굴을 찾지 못하거나 crop이 너무 작으면 HeadPose(0, 0, 0)을 반환합니다.
    """

    def __init__(self, cfg: Dict[str, Any]):
        from sixdrepnet import SixDRepNet

        device_str = cfg.get("device", "cpu")
        gpu_id = -1 if device_str == "cpu" else 0   # cpu면 -1, gpu면 0
        self.min_face_size = int(cfg.get("min_face_size", 30))  # 최소 얼굴 크기: 30px

        weights = cfg.get("weights", "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth")
        logger.info(f"Loading 6DRepNet model (gpu_id={gpu_id}, weights='{weights or 'auto'}')")
        self.model = SixDRepNet(gpu_id=gpu_id, dict_path=weights)   # 가중치 다운로드

    _ZERO = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

    def infer(self, frame, track: Track) -> Track:
        """
        track.headpose를 채워서 반환합니다.
        실패 시 HeadPose(0, 0, 0)을 설정합니다.
        """
        crop_bbox = track.crop_bbox if track.crop_bbox is not None else track.bbox
        crop_h = crop_bbox.h()
        crop_w = crop_bbox.w()

        if crop_h < self.min_face_size or crop_w < self.min_face_size:
            track.headpose = self._ZERO
            return track

        crop = frame[crop_bbox.y1:crop_bbox.y2, crop_bbox.x1:crop_bbox.x2]

        try:
            results = self.model.predict(crop)      # 6DRepNet에 crop 이미지 넣기
        except Exception as e:
            logger.debug(f"6DRepNet predict error: {e}")
            track.headpose = self._ZERO
            return track

        if results is None or len(results) == 0:    # 얼굴 못 찾음
            track.headpose = self._ZERO
            return track

        # sixdrepnet predict() 반환: (pitch_array, yaw_array, roll_array)
        # 각각 numpy array([value], dtype=float32)
        if len(results) != 3:
            track.headpose = self._ZERO
            return track

        pitch = float(results[0])
        yaw = float(results[1])
        roll = float(results[2])

        # yaw가 음수면 오른쪽, 양수면 왼쪽 / pitch가 음수면 아래, 양수면 위
        track.headpose = HeadPose(yaw=-yaw, pitch=pitch, roll=roll)
        return track

    def infer_batch(self, frame, tracks: List[Track]) -> List[Track]:
        """
        여러 track에 대해 headpose를 추정하여 track.headpose에 채웁니다.
        """
        return [self.infer(frame, t) for t in tracks]