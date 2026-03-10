# infer(left_eye, right_eye, headpose)->Gaze

from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import BBoxXYXY, Track, HeadPose, Gaze
from typing import Any, Dict, List, Optional, Tuple


class GazeDetector:
    def __init__(self, cfg: Dict[str, Any]):
        device_str = cfg.get("device", "CPU")
        weights = cfg.get("weights", "weights/gaze/gaze-estimation-adas-0002.xml")

        core = Core()
        model = core.read_model(model=weights)
        self.compiled_model = core.compile_model(model=model, device_name=device_str)
        self.output_layer = self.compiled_model.output(0)
        logger.info(f"[EyeDetector] weights={weights}  device={device_str}")

    def detect(self, frame: np.ndarray, track: Track, headpose: HeadPose) -> Gaze:
        """
        track.left_eye, track.right_eye, headpose(yaw, pitch, roll)를 입력받아서
        방향 벡터를 구하고 gaze(x, y, z) 출력
        """
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

        head_pose_angles = np.array([[headpose.yaw, headpose.pitch, headpose.roll]], dtype=np.float32)  # shape: (1, 3)

        infer_result = self.compiled_model([left_image, right_image, head_pose_angles])
        results = infer_result[self.output_layer]  # shape: (1, 3)

        gaze = Gaze(x=float(results[0][0]), y=float(results[0][1]), z=float(results[0][2]))

        return gaze

    def detect_batch(self, frame: np.ndarray, tracks: List[Track],
                     headpose: List[Tuple[int, Optional[HeadPose], Optional[str]]]) -> List[Gaze]:
        results: List[Gaze] = []

        # track_id -> HeadPose 매핑
        hp_map: Dict[int, HeadPose] = {}
        for tid, hp, _ in headpose:
            if hp is not None:
                hp_map[tid] = hp

        for track in tracks:
            hp = hp_map.get(track.track_id)
            if hp is None:
                results.append(Gaze(x=0.0, y=0.0, z=0.0))
                continue
            results.append(self.detect(frame, track, hp))

        return results



# https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/gaze-estimation-adas-0002
# https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/gaze-estimation-adas-0002/FP32/