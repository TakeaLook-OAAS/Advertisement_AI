# src/utils/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# OpenCV
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

@dataclass
class Track:
    """
    bytetrack_tracker.py의 표준 출력 형식

    - track_id: 지속적으로 유지되는 ID
    - bbox: 원본 프레임 기준 픽셀 좌표(xyxy)
    - crop_bbox: 얼굴 영역 기준 픽셀 좌표(xyxy)
    - left_eye, right_eye: 눈 영역 기준 픽셀 좌표(xyxy)
    - lifetime: yolo에서 처음 검출한 순간부터 지금까지의 프레임 수
    - hits: lifetime 중에서 yolo가 탐지한 프레임 수

    아래 값들은 보통 ROI/체류 로직에서 채움(트래커 자체가 아니라):
    - in_roi: ROI 내부 여부
    - dwell_frames: ROI 내부에서 체류한 프레임 수
    """
    track_id: int
    bbox: BBoxXYXY
    crop_bbox: Optional[BBoxXYXY] = None

    left_eye: Optional[BBoxXYXY] = None
    right_eye: Optional[BBoxXYXY] = None

    headpose: Optional[HeadPose] = None
    gaze: Optional[Gaze] = None
    attr: Optional[PersonAttr] = None
    roi: Optional[ROI] = None
    look_result: Optional[LookResult] = None

    lifetime: int = 0
    hits: int = 0
    conf: float = 0.0    # yolo 검출 신뢰도


# Yolo
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

    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

@dataclass(frozen=True)
class Det:
    """
    yolo_detector.py의 출력 형식

    - bbox: 원본 프레임 기준 픽셀 좌표(xyxy)
    - cls: 클래스 id (예: YOLO 모델에서 person=0 같은 값)
    - conf: 검출 신뢰도 (0.0 ~ 1.0)
    """
    bbox: BBoxXYXY
    cls: int
    conf: float


# 6DRepNet
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


# OpenVINO
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


# MiVOLO
class Gender(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"

class AgeGroup(str, Enum):
    age_0_9 = "0-9"
    age_10_19 = "10-19"
    age_20_29 = "20-29"
    age_30_39 = "30-39"
    age_40_49 = "40-49"
    age_50_59 = "50-59"
    age_60_plus = "60+"
    unknown = "unknown"

@dataclass(frozen=True)
class PersonAttr:
    """
    track_id별 MiVOLO 출력
    """
    gender: Gender
    age_group: AgeGroup

AttrMap = Dict[int, PersonAttr]   # track_id -> PersonAttr


# ROI 체류 판정
@dataclass(frozen=True)
class ROI:
    """
    ROI(관심 영역) 체류 판정 결과

    - in_roi: ROI 내부 여부
    - dwell_frames: ROI 내부에서 연속 체류한 프레임 수
    """
    in_roi: bool
    dwell_frames: int

# Look judgement / Events / Stats
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



# ── 상태 추적용 (status.py에서 사용) ──

@dataclass
class LookInterval:
    """시선 구간 1개. 보기 시작~끝."""
    start_ms: int
    end_ms: int

@dataclass
class PersonState:
    """
    track_id별 누적 상태. 매 프레임 Track 리스트와 비교해서 갱신한다.
    """
    # ── 식별 ──
    track_id: int

    # ── 노출 시간 ──
    first_seen_ms: int          # 처음 나타난 시간
    last_seen_ms: int           # 마지막으로 본 시간 (매 프레임 갱신)
    is_active: bool = True      # 아직 화면에 있는지 (사라지면 False)

    # ── 이전 프레임 상태 (다음 프레임과 비교용) ──
    is_looking: bool = False    # 이전 프레임에서 보고 있었는지
    in_roi: bool = False        # 이전 프레임에서 ROI 안에 있었는지

    # ── 시선 구간 기록 ──
    look_intervals: List[LookInterval] = field(default_factory=list)
    current_look_start_ms: Optional[int] = None   # 보기 시작한 시점 (None이면 안 보는 중)

    # ── 인구통계 ──
    age_group: Optional[str] = None
    gender: Optional[str] = None

@dataclass
class AdSegmentInfo:
    """
    ad_cycle.py에서 사용하는 광고 세그먼트 정보
    """
    segment_index: int      # 0, 1, 2, ... (계속 증가)
    ad_name: str            # "brand_A"
    cycle_index: int        # ads 리스트 내 인덱스 (순환)
    start_ms: int           # 이 세그먼트의 상대 시작 시간
    end_ms: int             # 이 세그먼트의 상대 종료 시간
    wall_start: str         # 세그먼트 절대 시작 시각 (ISO 8601)
