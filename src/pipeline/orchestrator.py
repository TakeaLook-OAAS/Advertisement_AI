"""
Orchestrator: wires together YOLO detection, ByteTrack, head-pose, and gaze
into a single per-frame inference result.

Pipeline (each frame):
  1. YOLO            → raw detections [x1,y1,x2,y2,score]
  2. ROI filter      → drop detections whose centre is outside the polygon
  3. ByteTrack       → confirmed tracks (list[STrack])
  4. Early exit      → if no tracks, skip heavy models
  5. Per-track crop  → crop person bbox → HeadPose + Gaze
  6. Attention logic → yaw/pitch threshold per track
  7. Return InferenceResult (per-track dict + primary summary)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.models.headpose_6drepnet import HeadPose6DRepNet
from src.models.gaze_openvino import GazeOpenVINO
from src.models.yolo_detector import YOLODetector
from src.tracking.byte_tracker import BYTETracker
from src.tracking.strack import STrack
from src.logic.attention import is_attending
from src.logic.roi import PolygonROI
from src.tracking.strack import TrackState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    # ── Primary (largest / most central) track summary ────────────────
    headpose:  Optional[Dict[str, float]] = None   # yaw / pitch / roll (°)
    gaze:      Optional[np.ndarray]       = None   # (3,) unit vector
    attending: Optional[bool]             = None

    # ── Per-track results ─────────────────────────────────────────────
    # key = track_id  (int)
    headpose_per_track:  Dict[int, Dict[str, float]] = field(default_factory=dict)
    attending_per_track: Dict[int, bool]             = field(default_factory=dict)

    # ── Shared metadata ───────────────────────────────────────────────
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta keys:
    #   "roi_polygon"   list[list[int]]
    #   "tracks"        list[STrack]   – active confirmed tracks
    #   "lost_tracks"   list[STrack]   – occluded / temporarily lost tracks
    #   "raw_dets"      np.ndarray (N,5)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-serialisable representation.
        Converts np.ndarray → list so the result can be sent to a
        Java Spring (or any HTTP) backend without extra marshalling.
        """
        tracks_summary = [
            {
                "track_id":  t.track_id,
                "bbox":      [round(float(v), 1) for v in t.tlbr],
                "score":     round(float(t.score), 3),
                "attending": self.attending_per_track.get(t.track_id),
                "headpose":  self.headpose_per_track.get(t.track_id),
            }
            for t in self.meta.get("tracks", [])
        ]
        return {
            "primary": {
                "headpose":  self.headpose,
                "gaze":      self.gaze.tolist() if self.gaze is not None else None,
                "attending": self.attending,
            },
            "tracks": tracks_summary,
            "track_count": len(tracks_summary),
        }


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def _crop_person(frame_bgr: np.ndarray, tlbr: np.ndarray,
                 margin: float = 0.1) -> np.ndarray:
    """
    Crop the person bounding box from the frame with a small margin.
    The upper portion of a full-body box already contains the head,
    and HeadPose6DRepNet uses an internal Haar cascade to locate the
    face within the crop — so passing the full person crop is correct.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = tlbr
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin

    cx1 = int(max(0, x1 - mx))
    cy1 = int(max(0, y1 - my))
    cx2 = int(min(w, x2 + mx))
    cy2 = int(min(h, y2 + my))
    return frame_bgr[cy1:cy2, cx1:cx2]


def _primary_track(tracks: List[STrack]) -> Optional[STrack]:
    """
    Select the most prominent track: largest bounding-box area
    (proxy for 'closest person to the billboard camera').
    """
    if not tracks:
        return None
    return max(tracks, key=lambda t: (t.tlbr[2] - t.tlbr[0]) * (t.tlbr[3] - t.tlbr[1]))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Creates and runs the full inference pipeline.

    Config sections consumed:
      models.yolo       → YOLODetector
      models.headpose   → HeadPose6DRepNet
      models.gaze       → GazeOpenVINO
      tracking          → BYTETracker
      logic.roi         → PolygonROI  (used for ROI filtering + visualisation)
      logic.attention   → attention thresholds
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        mcfg = cfg.get("models", {})
        tcfg = cfg.get("tracking", {})

        # ── Detection ────────────────────────────────────────────────
        self.yolo = YOLODetector(mcfg.get("yolo", {}))

        # ── Tracking ─────────────────────────────────────────────────
        self.tracker = BYTETracker(tcfg) if tcfg.get("enabled", True) else None

        # ── Head-pose & gaze ─────────────────────────────────────────
        hp_cfg = mcfg.get("headpose", {})
        gz_cfg = mcfg.get("gaze", {})
        self.headpose = HeadPose6DRepNet(hp_cfg) if hp_cfg.get("enabled", True) else None
        self.gaze     = GazeOpenVINO(gz_cfg)     if gz_cfg.get("enabled", True) else None

        # ── ROI & attention ──────────────────────────────────────────
        roi_cfg = cfg.get("logic", {}).get("roi", {})
        poly    = roi_cfg.get("polygon", [])
        self.roi     = PolygonROI(poly) if poly else None
        self.att_cfg = cfg.get("logic", {}).get("attention", {})

    # ------------------------------------------------------------------
    def _filter_by_roi(self, dets: np.ndarray) -> np.ndarray:
        """Remove detections whose centre lies outside the ROI polygon."""
        if self.roi is None or len(dets) == 0:
            return dets
        mask = np.array([self.roi.contains_box(dets[i, :4]) for i in range(len(dets))])
        return dets[mask]

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> InferenceResult:
        """Run the full inference pipeline on one BGR frame."""

        meta: Dict[str, Any] = {
            "roi_polygon": self.roi.points if self.roi else None,
            "tracks":      [],
            "lost_tracks": [],
            "raw_dets":    np.empty((0, 5), dtype=float),
        }

        # ── 1. YOLO ──────────────────────────────────────────────────
        raw_dets = self.yolo.infer(frame_bgr)
        meta["raw_dets"] = raw_dets

        # ── 2. ROI filter ─────────────────────────────────────────────
        roi_dets = self._filter_by_roi(raw_dets)

        # ── 3. ByteTrack ─────────────────────────────────────────────
        tracks: List[STrack] = []
        if self.tracker is not None:
            tracks = self.tracker.update(roi_dets)
            meta["tracks"]      = tracks
            meta["lost_tracks"] = list(self.tracker.lost_stracks)
        else:
            for i in range(len(roi_dets)):
                st = STrack(roi_dets[i, :4], float(roi_dets[i, 4]))
                st.activate(1)
                tracks.append(st)
            meta["tracks"] = tracks

        # ── 4. Early exit — no tracks ─────────────────────────────────
        if not tracks:
            return InferenceResult(meta=meta)

        # ── 5. Per-track HeadPose + Attention ─────────────────────────
        headpose_per_track:  Dict[int, Dict[str, float]] = {}
        attending_per_track: Dict[int, bool]             = {}

        for track in tracks:
            # Lost/occluded tracks are excluded from heavy model inference
            if track.state != TrackState.Tracked:
                continue

            crop = _crop_person(frame_bgr, track.tlbr)
            if crop.size == 0:
                continue

            head = self.headpose.infer(crop) if self.headpose else None
            if head is not None:
                headpose_per_track[track.track_id]  = head
                attending_per_track[track.track_id] = is_attending(head, self.att_cfg)

        # ── 6. Gaze on primary track only (expensive model) ───────────
        primary = _primary_track(tracks)
        primary_head = headpose_per_track.get(primary.track_id) if primary else None
        primary_attending = attending_per_track.get(primary.track_id) if primary else None

        primary_gaze = None
        if primary is not None and self.gaze is not None:
            crop = _crop_person(frame_bgr, primary.tlbr)
            if crop.size > 0:
                primary_gaze = self.gaze.infer(crop)

        return InferenceResult(
            headpose=primary_head,
            gaze=primary_gaze,
            attending=primary_attending,
            headpose_per_track=headpose_per_track,
            attending_per_track=attending_per_track,
            meta=meta,
        )
