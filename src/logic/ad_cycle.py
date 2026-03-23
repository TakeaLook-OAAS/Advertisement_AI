# 광고 사이클 스케줄러
# YAML의 ads 리스트를 받아서 사이클을 순환하며 경계 시점을 감지한다.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.types import AdSegmentInfo



class AdCycleScheduler:
    """
    광고 사이클을 관리한다.

    ads = [
        {"name": "brand_A", "duration_s": 15},
        {"name": "brand_B", "duration_s": 17},
    ]

    check(ts_ms)를 매 프레임 호출하면,
    경계를 넘었을 때 완료된 세그먼트 정보를 반환한다.
    """

    def __init__(self, ads: List[Dict[str, Any]]) -> None:
        if not ads:
            raise ValueError("ads list must not be empty")

        self._ads = [(a["name"], int(a["duration_s"] * 1000)) for a in ads]
        self._cycle_len = len(self._ads)

        self._segment_index = 0     # 현재 세그먼트 인덱스 (0, 1, 2, 3, 4, 5,...)
        self._cycle_index = 0       # 현재 사이클 인덱스 (0, 1, 2, 0, 1, 2, ...)
        self._segment_start_ms = 0  # 현재 세그먼트 시작 시간 (상대 시간)
        self._next_boundary_ms = self._ads[0][1]    # 다음 경계 시간 (상대 시간)
        self._wall_start = datetime.now(timezone.utc).isoformat()   # 현재 사이클 시작 시간 (절대 시간)

    def current_segment(self) -> AdSegmentInfo:
        name, _ = self._ads[self._cycle_index]
        return AdSegmentInfo(
            segment_index=self._segment_index,
            ad_name=name,
            cycle_index=self._cycle_index,
            start_ms=self._segment_start_ms,
            end_ms=self._next_boundary_ms,
            wall_start=self._wall_start,
        )

    def check(self, ts_ms: int) -> Optional[AdSegmentInfo]:
        """
        ts_ms가 경계를 넘었으면 완료된 세그먼트 정보를 반환하고 다음으로 진행.
        안 넘었으면 None.
        """
        if ts_ms < self._next_boundary_ms:
            return None

        completed = self.current_segment()

        # 다음 세그먼트로 진행
        self._segment_start_ms = self._next_boundary_ms
        self._segment_index += 1
        self._cycle_index = (self._cycle_index + 1) % self._cycle_len
        _, dur = self._ads[self._cycle_index]
        self._next_boundary_ms = self._segment_start_ms + dur
        self._wall_start = datetime.now(timezone.utc).isoformat()

        return completed
