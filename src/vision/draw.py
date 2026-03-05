from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import math
from src.utils.types import HeadPose, Track


# ── 색상 팔레트 (track_id별 고유 색상, 20색 순환) ──────────────────────
_PALETTE: List[Tuple[int, int, int]] = [
    (56,  56,  255), (151, 157, 255), (31, 112,  255), (29, 178,  255),
    (49, 210,  207), (10, 249,  72),  (23, 204,  146), (134, 219,  61),
    (52, 147,  26),  (187, 212,  0),  (168, 153,  44), (255, 194,  0),
    (147,  69, 52),  (255, 115,  100),(236,  24,  0),  (255,  56, 132),
    (133,   0, 82),  (255,  56, 203), (200, 149, 255), (199,  55, 255),
]


def _id_color(track_id: int) -> Tuple[int, int, int]:
    """track_id에 따라 고유 색상 반환."""
    return _PALETTE[track_id % len(_PALETTE)]


# ── 바운딩 박스 + ID 그리기 ─────────────────────────────────────────────
def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """각 Track의 바운딩 박스와 track_id를 프레임 위에 그린다."""
    for t in tracks:
        color = _id_color(t.track_id)
        x1, y1, x2, y2 = t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2

        cv2.rectangle(
            img=frame, 
            pt1=(x1, y1), 
            pt2=(x2, y2), 
            color=color, 
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
        cv2.putText(
            img=frame,
            text=f"id={t.track_id}",
            org=(x1, max(0, y1 - 5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


# ── FPS 표시 ───────────────────────────────────────────────────────────
def draw_fps(
    frame: np.ndarray,
    fps: float,
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """프레임 왼쪽 상단에 FPS를 표시한다."""
    cv2.putText(
        img=frame,
        text=f"FPS: {fps:.1f}",
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


# ── Headpose 각도 표시 ─────────────────────────────────────────────────
def draw_headpose(
    frame: np.ndarray,
    hp_results: List[Tuple[int, Optional[HeadPose], Optional[str]]],
    tracks: List[Track],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """
    각 Track의 bbox 아래쪽에 headpose 각도(yaw/pitch/roll) 텍스트를 표시한다.
    headpose가 None이면 실패 사유(reason)를 표시한다.
    """
    # track_id -> bbox 빠른 조회용
    bbox_map = {}
    for t in tracks:
        tid = t.track_id   # 사람/트랙 고유 ID
        bbox = t.bbox      # 그 트랙의 박스 좌표
        bbox_map[tid] = bbox

    for track_id, hp, reason in hp_results:
        bbox = bbox_map.get(track_id)
        if bbox is None:
            continue

        if hp is not None:
            text = f"Y:{hp.yaw:+.0f} P:{hp.pitch:+.0f} R:{hp.roll:+.0f}"
            color = _id_color(track_id)
        else:
            text = f"({reason})"
            color = (128, 128, 128)

        cv2.putText(
            img=frame,
            text=text,
            org=(bbox.x2, bbox.y1 - 18),       # bbox 오른쪽 위
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


# ── Headpose 벡터 표시 ─────────────────────────────────────────────────
def draw_headpose_vector(
    frame: np.ndarray,
    hp_results: List[Tuple[int, Optional[HeadPose], Optional[str]]],
    tracks: List[Track],
) -> None:
    # track_id -> bbox 빠른 조회용
    bbox_map = {}
    for t in tracks:
        tid = t.track_id   # 사람/트랙 고유 ID
        bbox = t.bbox      # 그 트랙의 박스 좌표
        bbox_map[tid] = bbox

    for track_id, hp, reason in hp_results:
        bbox = bbox_map.get(track_id)
        if bbox is None:
            continue

        if hp is not None:
            cx, cy = bbox.center()  # bbox 중앙 좌표
            arrow_len = min(bbox.w(), bbox.h()) // 4    # 화살표 길이 (bbox 크기의 1/4)
            dx = int(arrow_len * math.sin(math.radians(hp.yaw)))    # yaw → 좌우(X) 이동량
            dy = int(-arrow_len * math.sin(math.radians(hp.pitch)))   # pitch → 상하(Y) 이동량
            color = _id_color(track_id)
        else:
            continue

        cv2.arrowedLine(
            img=frame, 
            pt1=(cx, cy), 
            pt2=(cx + dx, cy + dy), 
            color=color, 
            thickness=3, 
            line_type=cv2.LINE_AA,
            tipLength=0.3,
        )