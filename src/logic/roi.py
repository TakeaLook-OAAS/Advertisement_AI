from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class PolygonROI:
    points: List[Tuple[int, int]]

    def __init__(self, pts: Sequence[Sequence[int]]):
        self.points = [(int(x), int(y)) for x, y in pts]
        self._pts_np = np.array(self.points, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        """(x, y) 좌표가 다각형 내부 또는 경계에 있으면 True 반환."""
        return cv2.pointPolygonTest(self._pts_np, (float(x), float(y)), measureDist=False) >= 0

    def contains_box(self, tlbr: Sequence[float]) -> bool:
        """Bounding Box 중심점이 다각형 안에 있으면 True 반환."""
        x1, y1, x2, y2 = tlbr
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.contains_point(cx, cy)
