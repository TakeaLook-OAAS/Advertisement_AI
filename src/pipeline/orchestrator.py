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
from src.models.yolo_detector import YoloDetector
from src.utils.types import Det, FrameMeta, Track


@dataclass
class OrchestratorOutput:
    dets: List[Det]
    tracks: List[Track]


class Orchestrator:
    """
    컨트롤 타워.
    매 프레임마다: detect -> track (지금은 여기까지만)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.detector = YoloDetector(cfg.get("models", {}).get("yolo", {}))
        self.tracker = ByteTrackTracker(cfg.get("models", {}).get("tracker", {}))

    def process(self, frame, meta: FrameMeta) -> OrchestratorOutput:
        dets = self.detector.detect(frame)
        tracks = self.tracker.update(dets)
        return OrchestratorOutput(dets=dets, tracks=tracks)
