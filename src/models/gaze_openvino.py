# infer(left_eye, right_eye, headpose)->Gaze

from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import Track, Gaze


class GazeDetector:
    _ZERO = Gaze(x=0.0, y=0.0, z=0.0)

    def __init__(self, cfg: Dict[str, Any]):
        device_str = cfg.get("device", "CPU")
        weights = cfg.get("weights", "weights/gaze/gaze-estimation-adas-0002.xml")

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[EyeDetector] weights={weights}  device={device_str}")

    def detect(self, frame: np.ndarray, track: Track) -> Track:
        """
        track.left_eye, track.right_eye, track.headpose를 사용하여
        gaze 벡터를 추정하고 track.gaze에 설정합니다.
        headpose가 zero이면 추론을 건너뜁니다.
        """
        hp = track.headpose
        if hp is None or (hp.yaw == 0.0 and hp.pitch == 0.0 and hp.roll == 0.0):
            track.gaze = self._ZERO
            return track

        left_eye = track.left_eye if track.left_eye is not None else track.bbox
        right_eye = track.right_eye if track.right_eye is not None else track.bbox
        left_crop = frame[left_eye.y1:left_eye.y2, left_eye.x1:left_eye.x2]
        right_crop = frame[right_eye.y1:right_eye.y2, right_eye.x1:right_eye.x2]

        resized = cv2.resize(left_crop, (60, 60))
        left_image = resized.transpose((2, 0, 1))  # HWC → CHW
        left_image = np.expand_dims(left_image, axis=0)

        resized = cv2.resize(right_crop, (60, 60))
        right_image = resized.transpose((2, 0, 1))  # HWC → CHW
        right_image = np.expand_dims(right_image, axis=0)
        # 입력 shape: 1, 3, 60, 60 (B, C, H, W)

        head_pose_angles = np.array([[hp.yaw, hp.pitch, hp.roll]], dtype=np.float32)  # shape: (1, 3)

        infer_result = self.compiled_model([left_image, right_image, head_pose_angles])
        results = infer_result[self.output_layer]  # shape: (1, 3)

        track.gaze = Gaze(x=float(results[0][0]), y=float(results[0][1]), z=float(results[0][2]))
        return track

    def detect_batch(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        """track.headpose를 사용하여 track.gaze를 채웁니다."""
        return [self.detect(frame, t) for t in tracks]



# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/gaze-estimation-adas-0002
# https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/gaze-estimation-adas-0002/FP32/