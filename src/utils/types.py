# src/utils/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Geometry / Common
# ----------------------------

@dataclass(frozen=True)
class BBoxXYXY:
    """
    픽셀 좌표계에서의 바운딩 박스(xyxy) 형식입니다.
    inclusive/exclusive(경계 포함 여부)는 crop/IOU 계산에서 일관성만 유지하면 됩니다.

    - x1 < x2, y1 < y2 이어야 합니다.
    - 값은 원본 프레임 좌표계 기준 픽셀 값입니다.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.w() * self.h()

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class FrameMeta:
    """
    프레임 메타데이터
    """
    frame_idx: int  # 몇 번째 프레임인지
    ts_ms: int      # 그 프레임의 시간 정보(밀리초)
    fps: float      # 영상의 FPS
    width: int      # 영상 해상도
    height: int     # 영상 해상도


# ----------------------------
# Detection / Tracking
# ----------------------------

@dataclass(frozen=True)
class Det:
    """
    yolo_detector.py의 출력 형식

    - bbox: 원본 프레임 기준 픽셀 좌표(xyxy)
    - cls: 클래스 id (예: YOLO 모델에서 person=0 같은 값)
    """
    bbox: BBoxXYXY
    cls: int


@dataclass
class Track:
    """
    bytetrack_tracker.py의 표준 출력 형식

    - track_id: 지속적으로 유지되는 ID
    - bbox: 원본 프레임 기준 픽셀 좌표(xyxy)
    - age: 처음 등장한 이후 경과한 프레임 수
    - hits: 매칭/확정된 프레임 수

    아래 값들은 보통 ROI/체류 로직에서 채움(트래커 자체가 아니라):
    - in_roi: ROI 내부 여부
    - dwell_frames: ROI 내부에서 체류한 프레임 수
    """
    track_id: int
    bbox: BBoxXYXY

    age: int = 0
    hits: int = 0

    # These are often filled by ROI/dwell logic (not tracker itself)
    in_roi: bool = False
    dwell_frames: int = 0


# ----------------------------
# Attributes (MiVOLO)
# ----------------------------

class Gender(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"


class AgeGroup(str, Enum):
    child = "child"         # 예: 0-12
    teen = "teen"           # 13-19
    young = "young_adult"   # 20-29
    adult = "adult"         # 30-49
    senior = "senior"       # 50+
    unknown = "unknown"


@dataclass(frozen=True)
class PersonAttr:
    """
    track_id별 MiVOLO 출력
    """
    gender: Gender
    age_group: AgeGroup


# ----------------------------
# HeadPose (6DRepNet)
# ----------------------------

@dataclass(frozen=True)
class HeadPose:
    """
    머리 자세 각도(Head pose)이며 단위는 degrees

    컨벤션:
    - yaw: 좌/우 회전
    - pitch: 상/하 회전
    - roll: 기울기(tilt)

    모델마다 부호(sign) 정의가 다를 수 있으므로,
    어댑터 레이어에서 부호/축 정의를 고정해서 맞춰야 함
    """
    yaw: float
    pitch: float
    roll: float


# ----------------------------
# Gaze (OpenVINO)
# ----------------------------

@dataclass(frozen=True)
class Gaze:
    """
    카메라 좌표계 기준 3D 시선 방향 벡터
    강력 추천: 단위벡터(정규화된 방향 벡터)로 저장하세요.
    """
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ----------------------------
# Look judgement / Events / Stats
# ----------------------------

@dataclass(frozen=True)
class LookResult:
    """
    '광고판을 보고 있는지' 판정 단계의 출력입니다.

    - score: 보통 코사인 유사도 [-1, 1]
    - angle_deg: arccos(score)를 '도(deg)'로 변환한 값
    """
    is_looking: bool
    score: float
    angle_deg: float


class EventType(str, Enum):
    # 유동/체류/관심 집계를 위한 최소 이벤트들
    pass_by = "pass_by"             # 유동(ROI 근처/통과) 판정
    enter_roi = "enter_roi"
    exit_roi = "exit_roi"
    dwell_start = "dwell_start"
    dwell_end = "dwell_end"
    look_start = "look_start"
    look_end = "look_end"
    # 추가해야됨


@dataclass(frozen=True)
class Event:
    """
    이벤트 기반 로깅 레코드입니다(JSONL로 저장하기 좋게 설계).

    - payload: 필요한 값(각도, 점수, bbox 등)을 자유롭게 추가할 수 있습니다.
    """
    ts_ms: int                  # 사건이 일어난 시간
    frame_idx: int              # 몇 번째 프레임인지
    track_id: int               # 어떤 사람인지
    type: EventType             # 이벤트 종류(들어온 사건, 머문 사건, 본 사건)  
    payload: Dict[str, object]


# ----------------------------
# Convenience types
# ----------------------------

AttrMap = Dict[int, PersonAttr]   # track_id -> PersonAttr
