# 모든 모델/로직을 들고 있는 컨트롤 타워

# 내부에서 단계별로 호출만 한다:
# 1. detect
# 2. track
# 3. dwell filter
# 4. attributes
# 5. headpose/gaze
# 6. judge(ad look)
# 7. stats update
# 8. log/write

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.yolo_detector import YoloDetector
from src.models.mivolo_attr import MiVOLOAttr
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.models.eye_openvino import EyeDetector
from src.models.gaze_openvino import GazeDetector
from src.logic.stay import StayTracker
from src.logic.look_judge import LookJudge
from src.utils.types import Det, FrameMeta, Track


@dataclass
class OrchestratorOutput:
    dets: List[Det]
    tracks: List[Track]

class Orchestrator:
    """
    컨트롤 타워.
    매 프레임마다: detect -> track -> 구현 아직 x
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        # models
        self.detector = YoloDetector(cfg.get("models", {}).get("yolo", {}))
        self.tracker = ByteTrackTracker(cfg.get("models", {}).get("tracker", {}))
        self.face = FaceDetector(cfg.get("models", {}).get("face", {}))
        self.mivolo = MiVOLOAttr(cfg.get("models", {}).get("mivolo", {}))
        self.headpose = HeadPoseEstimator(cfg.get("models", {}).get("headpose", {}))
        self.eye = EyeDetector(cfg.get("models", {}).get("eye", {}))
        self.gaze = GazeDetector(cfg.get("models", {}).get("gaze", {}))

        # logic
        roi_pts = cfg.get("logic", {}).get("roi", {}).get("polygon", [])
        self.stay_tracker = StayTracker(roi_pts) if roi_pts else None
        self.look_judge = LookJudge(cfg.get("logic", {}).get("attention", {}))

    def process(self, frame, meta: FrameMeta) -> OrchestratorOutput:
        # 1) detect
        dets = self.detector.detect(frame)

        # 2) track
        tracks = self.tracker.update(dets)

        # 3) crop face → track.crop_bbox
        tracks = self.face.detect_batch(frame, tracks)

        # 4) headpose → track.headpose
        tracks = self.headpose.infer_batch(frame, tracks)

        # 5) crop eye → track.left_eye, track.right_eye
        tracks = self.eye.detect_batch(frame, tracks)

        # 6) gaze → track.gaze
        tracks = self.gaze.detect_batch(frame, tracks)

        # 7) ROI 판정 → track.roi
        tracks = self.stay_tracker.update(tracks)

        # 8) 시선 판정 → track.look_result
        tracks = self.look_judge.judge_batch(tracks)

        return OrchestratorOutput(dets=dets, tracks=tracks)
