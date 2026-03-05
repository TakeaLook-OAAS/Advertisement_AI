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
from typing import Any, Dict, List, Optional, Tuple
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.yolo_detector import YoloDetector
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.utils.types import Det, FrameMeta, HeadPose, Track


@dataclass
class OrchestratorOutput:
    dets: List[Det]
    tracks: List[Track]
    hp_results: List[Tuple[int, Optional[HeadPose], Optional[str]]]

class Orchestrator:
    """
    컨트롤 타워.
    매 프레임마다: detect -> track -> 구현 아직 x
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.detector = YoloDetector(cfg.get("models", {}).get("yolo", {}))
        self.tracker = ByteTrackTracker(cfg.get("models", {}).get("tracker", {}))
        self.face = FaceDetector(cfg.get("models", {}).get("face", {}))
        self.headpose = HeadPoseEstimator(cfg.get("models", {}).get("headpose", {}))

    def process(self, frame, meta: FrameMeta) -> OrchestratorOutput:
        # 1) detect
        dets = self.detector.detect(frame)

        # 2) track
        tracks = self.tracker.update(dets)

        # 3) crop face
        tracks = self.face.detect_batch(frame, tracks)

        # 4) headpose
        hp_results = self.headpose.infer_batch(frame, tracks)

        # 5) gaze     (TODO: gaze_openvino 구현 후)
        # 6) attention (TODO: headpose 기반 판정)
        return OrchestratorOutput(dets=dets, tracks=tracks, hp_results=hp_results)