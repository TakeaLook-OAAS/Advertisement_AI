"""
Visualization helpers for the inference overlay.

draw_overlay()         – FPS / headpose / gaze arrow / attention / ROI
draw_tracks()          – BBox + track_id + attention colour coding
draw_occlusion_status()– Corner-mark bounding boxes for Lost tracks
"""
import cv2
import numpy as np
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Color palette  (20 distinct BGR colors, cycles on track_id)
# ---------------------------------------------------------------------------
_PALETTE: List[tuple] = [
    (56,  56,  255), (151, 157, 255), (31, 112,  255), (29, 178,  255),
    (49, 210,  207), (10, 249,  72),  (23, 204,  146), (134, 219,  61),
    (52, 147,  26),  (187, 212,  0),  (168, 153,  44), (255, 194,  0),
    (147,  69, 52),  (255, 115,  100),(236,  24,  0),  (255,  56, 132),
    (133,   0, 82),  (255,  56, 203), (200, 149, 255), (199,  55, 255),
]

_COLOR_ATTENDING     = (0,  220,  80)   # bright green
_COLOR_NOT_ATTENDING = (0,   60, 220)   # red
_COLOR_UNKNOWN       = (160, 160, 160)  # grey


def _id_color(track_id: int) -> tuple:
    return _PALETTE[int(track_id) % len(_PALETTE)]


def _attention_style(attending: Optional[bool]):
    """Return (border_color, thickness) based on attention state."""
    if attending is True:
        return _COLOR_ATTENDING, 3
    if attending is False:
        return _COLOR_NOT_ATTENDING, 2
    return _COLOR_UNKNOWN, 1


# ---------------------------------------------------------------------------
# Semi-transparent label helper
# ---------------------------------------------------------------------------

def _put_label(
    vis: np.ndarray,
    text: str,
    origin: tuple,          # (x, y) bottom-left of text
    bg_color: tuple,
    font_scale: float = 0.55,
    alpha: float = 0.65,
) -> None:
    """Draw a semi-transparent filled background, then white text on top."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = origin
    box_x1, box_y1 = x, max(y - th - bl - 2, 0)
    box_x2, box_y2 = x + tw + 4, y + 2

    # Alpha blend: draw on a copy, then merge
    overlay = vis.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    cv2.putText(
        vis, text, (x + 2, y - bl),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Track bounding-box overlay
# ---------------------------------------------------------------------------

def draw_tracks(
    vis: np.ndarray,
    tracks,
    *,
    cfg: dict,
    attending_per_track: Dict[int, bool] = None,
) -> np.ndarray:
    """
    Draw bounding boxes for active STrack objects.

    Border colour & thickness reflect attention state:
      Green (thick)  → attending
      Red            → not attending
      Grey (thin)    → unknown / no headpose result

    Controlled by cfg["display"]:
      draw_tracks      bool   true
      draw_track_id    bool   true
      draw_track_score bool   false
    """
    disp = cfg.get("display", {})
    if not disp.get("draw_tracks", True):
        return vis

    font_scale = float(disp.get("font_scale", 0.6))
    draw_id    = bool(disp.get("draw_track_id",    True))
    draw_score = bool(disp.get("draw_track_score", False))

    if attending_per_track is None:
        attending_per_track = {}

    track_list = list(tracks)

    for track in track_list:
        x1, y1, x2, y2 = (int(v) for v in track.tlbr)
        id_color   = _id_color(track.track_id)
        att_status = attending_per_track.get(track.track_id)
        border_color, border_thick = _attention_style(att_status)

        # Outer attention border
        cv2.rectangle(vis, (x1, y1), (x2, y2), border_color, border_thick)
        # Inner ID-colored thin border
        cv2.rectangle(vis, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), id_color, 1)

        if draw_id or draw_score:
            parts = []
            if draw_id:
                parts.append(f"#{track.track_id}")
            if draw_score:
                parts.append(f"{track.score:.2f}")
            if att_status is True:
                parts.append("LOOK")
            elif att_status is False:
                parts.append("AWAY")
            label = " ".join(parts)
            _put_label(vis, label, (x1, y1), id_color, font_scale)

    # Track count — top right corner
    h, w = vis.shape[:2]
    count_text = f"Tracks: {len(track_list)}"
    (cw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _put_label(vis, count_text, (w - cw - 14, 28), (40, 40, 40), 0.55)

    return vis


# ---------------------------------------------------------------------------
# Occlusion indicator — L-shaped corner marks (fast, clean)
# ---------------------------------------------------------------------------

def draw_occlusion_status(vis: np.ndarray, lost_tracks, *, cfg: dict) -> np.ndarray:
    """
    Draw L-shaped corner marks for Lost (occluded) tracks.
    Much faster than a full dashed rectangle: only 8 cv2.line calls per track.
    """
    disp = cfg.get("display", {})
    if not disp.get("draw_lost_tracks", True):
        return vis

    for track in lost_tracks:
        x1, y1, x2, y2 = (int(v) for v in track.tlbr)
        color  = _id_color(track.track_id)
        arm    = max(10, min(20, (x2 - x1) // 5))  # corner arm length

        # Four L-shaped corners
        corners = [
            ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),   # top-left
            ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),   # top-right
            ((x1, y2 - arm), (x1, y2), (x1 + arm, y2)),   # bottom-left
            ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),   # bottom-right
        ]
        for p0, pivot, p1 in corners:
            cv2.line(vis, p0, pivot, color, 2, cv2.LINE_AA)
            cv2.line(vis, pivot, p1, color, 2, cv2.LINE_AA)

        label = f"#{track.track_id} lost"
        _put_label(vis, label, (x1, y1 - 2), (60, 60, 60), 0.45, alpha=0.5)

    return vis


# ---------------------------------------------------------------------------
# Gaze direction arrow
# ---------------------------------------------------------------------------

def draw_gaze_arrow(
    vis: np.ndarray,
    gaze_vec: np.ndarray,
    primary_tlbr,
    *,
    scale: int = 120,
) -> np.ndarray:
    """
    Draw an arrow from the estimated face centre in the gaze direction.

    gaze_vec  : (3,) [gx, gy, gz]  —  gx>0 right, gy>0 down, gz>0 toward cam
    primary_tlbr : [x1, y1, x2, y2] of the primary tracked person
    scale     : pixels per unit of gaze vector magnitude
    """
    if gaze_vec is None or primary_tlbr is None:
        return vis

    x1, y1, x2, y2 = (int(v) for v in primary_tlbr)

    # Approximate face centre = upper-quarter of the person bbox
    cx = (x1 + x2) // 2
    cy = y1 + (y2 - y1) // 5       # ~20 % from top

    gx, gy = float(gaze_vec[0]), float(gaze_vec[1])
    end_x = int(cx + gx * scale)
    end_y = int(cy + gy * scale)   # gy>0 = looking downward in image

    cv2.arrowedLine(
        vis,
        (cx, cy), (end_x, end_y),
        color=(0, 255, 220),
        thickness=2,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )
    return vis


# ---------------------------------------------------------------------------
# Main overlay (FPS / headpose / attention status / ROI + tracking)
# ---------------------------------------------------------------------------

def draw_overlay(frame_bgr, result, *, fps: float, cfg: dict) -> np.ndarray:
    """
    Composite overlay drawn on a copy of frame_bgr.

    Reads track data and per-track attention from result.meta and
    result.attending_per_track, so no extra arguments are needed.
    """
    disp       = cfg.get("display", {})
    font_scale = float(disp.get("font_scale", 0.7))
    thickness  = int(disp.get("thickness", 2))

    vis = frame_bgr.copy()
    y   = 30

    # FPS
    if disp.get("draw_fps", True):
        cv2.putText(
            vis, f"FPS: {fps:.1f}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 0), thickness, cv2.LINE_AA,
        )
        y += 30

    # Head pose
    if disp.get("draw_headpose", True) and result.headpose:
        hp = result.headpose
        cv2.putText(
            vis,
            f"Yaw:{hp['yaw']:+.1f}  Pitch:{hp['pitch']:+.1f}  Roll:{hp['roll']:+.1f}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        y += 30

    # Gaze — numeric text
    if disp.get("draw_gaze", True) and result.gaze is not None:
        gv = result.gaze
        cv2.putText(
            vis,
            f"Gaze: [{gv[0]:+.2f}, {gv[1]:+.2f}, {gv[2]:+.2f}]",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (200, 200, 255), thickness, cv2.LINE_AA,
        )
        y += 30

    # Global attention status (primary track)
    if result.attending is not None:
        status = "ATTENDING" if result.attending else "NOT ATTENDING"
        color  = _COLOR_ATTENDING if result.attending else _COLOR_NOT_ATTENDING
        cv2.putText(
            vis, status,
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.2, color, thickness, cv2.LINE_AA,
        )

    # ROI polygon
    poly = (result.meta or {}).get("roi_polygon")
    if poly:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 120, 0), thickness=2)

    # ── Gaze arrow (primary track) ────────────────────────────────────
    tracks = (result.meta or {}).get("tracks", [])
    if result.gaze is not None and tracks:
        primary = max(
            tracks,
            key=lambda t: (t.tlbr[2] - t.tlbr[0]) * (t.tlbr[3] - t.tlbr[1]),
        )
        vis = draw_gaze_arrow(vis, result.gaze, primary.tlbr)

    # ── Track bounding boxes ──────────────────────────────────────────
    att_per_track = getattr(result, "attending_per_track", {})
    if tracks:
        vis = draw_tracks(vis, tracks, cfg=cfg, attending_per_track=att_per_track)

    # ── Lost / occluded tracks ────────────────────────────────────────
    lost_tracks = (result.meta or {}).get("lost_tracks", [])
    if lost_tracks:
        vis = draw_occlusion_status(vis, lost_tracks, cfg=cfg)

    return vis
