# track_id별 ROI 체류 프레임 수를 누적 관리하는 클래스
# runner.py의 루프에서 매 프레임마다 update()를 호출한다.

from __future__ import annotations
from typing import Dict, List, Sequence
import cv2
import numpy as np
from src.utils.types import ROI, Track


class StayTracker:
    """
    매 프레임마다 tracks를 받아서:
    - ROI 안에 있는 track → in_roi=True, dwell_frames 누적
    - ROI 밖에 있는 track → in_roi=False, dwell_frames 초기화
    """

    def __init__(self, polygon: Sequence[Sequence[int]]):
        self._polygon = np.array(polygon, dtype=np.int32)
        self._dwell: Dict[int, int] = {}   # track_id → 누적 체류 프레임

    def _contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._polygon, (float(x), float(y)), measureDist=False) >= 0

    def update(self, tracks: List[Track]) -> List[Track]:
        """track.roi를 채워서 반환합니다."""
        active_ids = set()

        for t in tracks:
            active_ids.add(t.track_id)
            cx, cy = t.bbox.center()

            if self._contains(cx, cy):
                self._dwell[t.track_id] = self._dwell.get(t.track_id, 0) + 1
                t.roi = ROI(in_roi=True, dwell_frames=self._dwell[t.track_id])
            else:
                self._dwell.pop(t.track_id, None)
                t.roi = ROI(in_roi=False, dwell_frames=0)

        # 사라진 track 정리
        for tid in list(self._dwell):
            if tid not in active_ids:
                del self._dwell[tid]

        return tracks
