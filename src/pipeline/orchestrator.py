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
from src.models.bytetrack_tracker import OfficialByteTrackAdapter
from src.models.face_yolov8 import FaceDetector
from src.models.yolo_detector import YoloDetector
from src.models.mivolo_attr import MiVOLOAttr
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.models.eye.eye_pytorch import EyeDetector
from src.models.gaze.gaze_pytorch import GazeDetector
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
        self.face = FaceDetector(cfg.get("models", {}).get("face", {}))
        self.eye = EyeDetector(cfg.get("models", {}).get("eye", {}))
        self.gaze = GazeDetector(cfg.get("models", {}).get("gaze", {}))

        self.detector = YoloDetector(cfg.get("models", {}).get("yolo", {}))

        # tracker.fps는 "원본 영상 fps" 기준. frame_skip으로 실제 update() 호출 빈도가
        # 줄어드는 만큼 나눠서 넘겨야 ByteTrack의 buffer_size(=max_time_lost) 계산이
        # 실제 경과 시간과 맞아떨어진다 (안 그러면 frame_skip>1일 때 트랙이 너무 빨리 삭제됨).
        tracker_cfg = dict(cfg.get("models", {}).get("tracker", {}))
        frame_skip = int(cfg.get("pipeline", {}).get("frame_skip", 1))
        tracker_cfg["fps"] = tracker_cfg.get("fps", 30) / frame_skip
        self.tracker = OfficialByteTrackAdapter(tracker_cfg)
        self.mivolo = MiVOLOAttr(cfg.get("models", {}).get("mivolo", {}))
        self.headpose = HeadPoseEstimator(cfg.get("models", {}).get("headpose", {}))

        # logic
        roi_pts = cfg.get("logic", {}).get("roi", {}).get("polygon", [])
        self.stay_tracker = StayTracker(roi_pts) if roi_pts else None
        self.look_judge = LookJudge(cfg.get("logic", {}).get("attention", {}))

    def process(self, frame) -> OrchestratorOutput:
        # 1) detect
        dets = self.detector.detect(frame)

        # 2) track
        tracks = self.tracker.update(dets)

        # 3) crop face → track.crop_bbox
        tracks = self.face.detect_batch(frame, tracks)

        # 4) mivolo -> track.attr
        tracks = self.mivolo.infer(frame, tracks)

        # 5) headpose → track.headpose
        tracks = self.headpose.infer_batch(frame, tracks)

        # 6) crop eye → track.left_eye, track.right_eye
        tracks = self.eye.detect_batch(frame, tracks)

        # 7) gaze → track.gaze
        tracks = self.gaze.detect_batch(frame, tracks)

        # 8) ROI 판정 → track.roi, none일 경우 에러나서 코드 수정
        if self.stay_tracker is not None:
            tracks = self.stay_tracker.update(tracks)

        # 9) 시선 판정 → track.look_result
        tracks = self.look_judge.judge_batch(tracks)

        return OrchestratorOutput(dets=dets, tracks=tracks)
