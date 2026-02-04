from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.models.headpose_6drepnet import HeadPose6DRepNet
from src.models.gaze_openvino import GazeOpenVINO
from src.logic.attention import is_attending
from src.logic.roi import PolygonROI

@dataclass
class InferenceResult:
    headpose: Optional[Dict[str, float]] = None     # yaw/pitch/roll degrees
    gaze: Optional[np.ndarray] = None               # (3,) vector
    attending: Optional[bool] = None
    meta: Optional[Dict[str, Any]] = None

class Orchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        mcfg = cfg.get("models", {})
        self.headpose = HeadPose6DRepNet(mcfg.get("headpose", {})) if mcfg.get("headpose", {}).get("enabled", True) else None
        self.gaze = GazeOpenVINO(mcfg.get("gaze", {})) if mcfg.get("gaze", {}).get("enabled", True) else None

        roi_cfg = cfg.get("logic", {}).get("roi", {})
        poly = roi_cfg.get("polygon", [])
        self.roi = PolygonROI(poly) if poly else None

        self.att_cfg = cfg.get("logic", {}).get("attention", {})

    def process(self, frame_bgr) -> InferenceResult:
        # In a real project, you'd detect/crop face ROI before calling headpose/gaze.
        # This skeleton keeps it simple.
        head = self.headpose.infer(frame_bgr) if self.headpose else None
        gaze = self.gaze.infer(frame_bgr) if self.gaze else None

        attending = None
        if head is not None:
            attending = is_attending(head, self.att_cfg)

        return InferenceResult(
            headpose=head,
            gaze=gaze,
            attending=attending,
            meta={"roi_polygon": self.roi.points if self.roi else None},
        )
