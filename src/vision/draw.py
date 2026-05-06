from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
import math
from src.utils.types import Track


# 색상 팔레트 (track_id별 고유 색상, 20색 순환)
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


# bbox + ID 그리기
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


# crop_bbox 그리기
def draw_crop_bbox(
    frame: np.ndarray,
    tracks: List[Track],
    thickness: int = 2,
) -> None:
    """각 Track의 crop_bbox(얼굴 영역)를 프레임 위에 그린다."""
    for t in tracks:
        if t.crop_bbox is None:
            continue
        color = _id_color(t.track_id)
        x1, y1, x2, y2 = t.crop_bbox.x1, t.crop_bbox.y1, t.crop_bbox.x2, t.crop_bbox.y2

        cv2.rectangle(
            img=frame,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )


# FPS 표시
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


# Headpose 각도 + vector 표시
def draw_headpose(
    frame: np.ndarray,
    tracks: List[Track],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """각 Track의 bbox 위에 headpose 각도(yaw/pitch/roll) 텍스트와 vector를 표시한다."""
    for track in tracks:
        hp = track.headpose
        if hp is None:
            continue
        bbox = track.bbox
        crop = track.crop_bbox if track.crop_bbox is not None else track.bbox
        color = _id_color(track.track_id)

        text = f"Y:{hp.yaw:+.0f} P:{hp.pitch:+.0f} R:{hp.roll:+.0f}"
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

        # headpose vector 표시
        cx, cy = crop.center()
        arrow_len = min(crop.w(), crop.h())
        dx = int(arrow_len * math.sin(math.radians(hp.yaw)))
        dy = int(-arrow_len * math.sin(math.radians(hp.pitch)))

        cv2.arrowedLine(
            img=frame,
            pt1=(cx, cy),
            pt2=(cx + dx, cy + dy),
            color=color,
            thickness=3,
            line_type=cv2.LINE_AA,
            tipLength=0.3,
        )


# Gaze 벡터 표시
def draw_gaze(
    frame: np.ndarray,
    tracks: List[Track],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """각 Track의 양쪽 눈 중앙에서 gaze 방향 벡터를 화살표로 표시한다."""
    for track in tracks:
        gaze = track.gaze
        if gaze is None:
            continue
        bbox = track.bbox
        color = _id_color(track.track_id)

        # gaze 수치 텍스트 (headpose 텍스트 위에 표시)
        text = f"G:{gaze.x:+.2f} {gaze.y:+.2f} {gaze.z:+.2f}"
        cv2.putText(
            img=frame,
            text=text,
            org=(bbox.x2, bbox.y1 - 36),       # headpose 텍스트(-18) 위
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        # 양쪽 눈에서 gaze vector 화살표
        for eye_bbox in (track.left_eye, track.right_eye):
            if eye_bbox is None:
                continue
            cx, cy = eye_bbox.center()
            arrow_len = min(eye_bbox.w(), eye_bbox.h()) * 2
            dx = int(arrow_len * gaze.x)
            dy = int(-arrow_len * gaze.y)

            cv2.arrowedLine(
                img=frame,
                pt1=(cx, cy),
                pt2=(cx + dx, cy + dy),
                color=color,
                thickness=2,
                line_type=cv2.LINE_AA,
                tipLength=0.3,
            )


# LookResult 텍스트 표시
def draw_look(
    frame: np.ndarray,
    tracks: List[Track],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """각 Track bbox 위에 LookResult(보고 있는지, 각도)를 표시한다."""
    for track in tracks:
        lr = track.look_result
        if lr is None:
            continue
        color = _id_color(track.track_id)
        cv2.putText(
            img=frame,
            text=f"Look:{lr.is_looking} Degree:{lr.angle_deg:.1f}",
            org=(track.bbox.x2, track.bbox.y1 - 72),   # ROI 텍스트(-54) 위
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


# gender, age_group 표시
def draw_gender_age(
    frame: np.ndarray,
    tracks: List[Track],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """각 Track bbox 위에 gender, age_group를 표시한다. id 텍스트 바로 위."""
    for track in tracks:
        attr = track.attr
        if attr is None:
            continue
        color = _id_color(track.track_id)
        x1, y1 = track.bbox.x1, track.bbox.y1
        cv2.putText(
            img=frame,
            text=f"{attr.gender.value}, {attr.age_group.value}",
            org=(x1, max(0, y1 - 25)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )