"""
STrack: Kalman 필터 기반 상태 머신을 가진 단일 객체 트랙.
"""
from __future__ import annotations

from enum import IntEnum
from itertools import count
from typing import Optional

import numpy as np

from .kalman_filter import KalmanFilter

# 공유 Kalman 필터 인스턴스 (무상태 수학 연산이므로 공유 안전)
_kf = KalmanFilter()

# 스레드 안전 트랙 ID 생성기
_id_counter = count(1)


def _next_id() -> int:
    return next(_id_counter)


class TrackState(IntEnum):
    New     = 0   # 방금 생성됨, 아직 확정 안 됨
    Tracked = 1   # 현재 프레임에서 활성 매칭 중
    Lost    = 2   # 매칭 실패; 재식별을 위해 유지
    Removed = 3   # max_lost 초과, 완전 제거


class STrack:
    """단일 객체 트랙 (ByteTrack).

    좌표는 tlbr = [x1, y1, x2, y2] 형태로 저장.
    Kalman 상태는 cxcywh = [cx, cy, w, h] 형태 사용.
    """

    __slots__ = (
        "track_id", "tlbr", "score",
        "state", "frame_id", "start_frame",
        "tracklet_len", "is_activated",
        "mean", "covariance",
    )

    def __init__(self, tlbr: np.ndarray, score: float):
        self.track_id: int = _next_id()
        self.tlbr: np.ndarray = tlbr.copy()
        self.score: float = float(score)
        self.state: TrackState = TrackState.New
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.tracklet_len: int = 0
        self.is_activated: bool = False
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    @staticmethod
    def tlbr_to_cxcywh(tlbr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = tlbr
        return np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=float
        )

    @staticmethod
    def cxcywh_to_tlbr(cxcywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = cxcywh
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    # ------------------------------------------------------------------
    # 상태 머신
    # ------------------------------------------------------------------
    def activate(self, frame_id: int) -> None:
        """최초 활성화 (New → Tracked)."""
        meas = self.tlbr_to_cxcywh(self.tlbr)
        self.mean, self.covariance = _kf.initiate(meas)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def predict(self) -> None:
        """Kalman 예측; Lost 트랙은 속도를 0으로 고정."""
        if self.mean is not None:
            m = self.mean.copy()
            if self.state != TrackState.Tracked:
                m[4:] = 0.0  # 속도 추정 동결
            self.mean, self.covariance = _kf.predict(m, self.covariance)
            self.tlbr = self.cxcywh_to_tlbr(self.mean[:4])

    def update(self, det: "STrack", frame_id: int) -> None:
        """매칭된 검출로 업데이트 (Tracked → Tracked)."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        meas = self.tlbr_to_cxcywh(det.tlbr)
        self.mean, self.covariance = _kf.update(self.mean, self.covariance, meas)
        self.tlbr = self.cxcywh_to_tlbr(self.mean[:4])
        self.score = det.score
        self.state = TrackState.Tracked
        self.is_activated = True

    def re_activate(self, det: "STrack", frame_id: int) -> None:
        """Lost 트랙을 검출과 재연결."""
        meas = self.tlbr_to_cxcywh(det.tlbr)
        self.mean, self.covariance = _kf.update(self.mean, self.covariance, meas)
        self.tlbr = self.cxcywh_to_tlbr(self.mean[:4])
        self.tracklet_len = 0
        self.score = det.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    # ------------------------------------------------------------------
    @property
    def age(self) -> int:
        """마지막 매칭 이후 경과 프레임 수."""
        return self.frame_id

    def __repr__(self) -> str:
        return (
            f"STrack(id={self.track_id}, state={self.state.name}, "
            f"score={self.score:.2f}, len={self.tracklet_len})"
        )
