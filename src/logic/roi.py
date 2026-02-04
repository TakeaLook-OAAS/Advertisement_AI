from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

@dataclass
class PolygonROI:
    points: List[Tuple[int, int]]

    def __init__(self, pts: Sequence[Sequence[int]]):
        self.points = [(int(x), int(y)) for x, y in pts]
