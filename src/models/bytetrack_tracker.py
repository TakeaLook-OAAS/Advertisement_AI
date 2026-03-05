# update(dets: List[Det]) -> List[Track]
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.utils.types import BBoxXYXY, Det, Track


@dataclass
class _T:
    """내부 트랙 상태."""
    id:   int
    box:  np.ndarray   # [x1, y1, x2, y2]
    conf: float
    hits: int = 0
    age:  int = 0
    lost: int = 0


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4)  b:(M,4)  →  IoU 행렬 (N,M)"""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match(tracks: List[_T], boxes: np.ndarray, thresh: float):
    """greedy IoU 매칭. 반환: (매칭쌍, 미매칭 트랙 idx, 미매칭 검출 idx)"""
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    iou = _iou(np.array([t.box for t in tracks]), boxes)
    used_t, used_d, matched = set(), set(), []
    for ti, di in sorted(np.argwhere(iou >= thresh).tolist(), key=lambda x: -iou[x[0], x[1]]):
        if ti not in used_t and di not in used_d:
            matched.append((ti, di)); used_t.add(ti); used_d.add(di)
    return (matched,
            [i for i in range(len(tracks)) if i not in used_t],
            [j for j in range(len(boxes)) if j not in used_d])


class ByteTrackTracker:
    """
    ByteTrack 스타일 트래커 (독립 구현).
    입력 List[Det] → 출력 List[Track]

    1. 고신뢰도(≥track_thresh) / 저신뢰도 검출 분리
    2. 1차 매칭: 전체 트랙 ↔ 고신뢰도
    3. 2차 매칭: 미매칭 트랙 ↔ 저신뢰도
    4. 미매칭 검출 → 새 트랙,  미매칭 트랙 → lost++
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.hi       = float(cfg.get("track_thresh",   0.5))
        self.lo       = float(cfg.get("low_thresh",     0.1))
        self.iou_th   = float(cfg.get("match_thresh",   0.8))
        self.max_lost = int(cfg.get("max_lost_frames", 30))
        self.min_hits = int(cfg.get("min_hits",         1))
        self._tracks: List[_T] = []
        self._next_id: int = 1

    @staticmethod
    def _boxes(dets: List[Det]) -> np.ndarray:
        return np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
                         for d in dets], dtype=float).reshape(-1, 4)

    def update(self, dets: List[Det]) -> List[Track]:
        hi = [d for d in dets if d.conf >= self.hi]
        lo = [d for d in dets if self.lo <= d.conf < self.hi]
        hi_b, lo_b = self._boxes(hi), self._boxes(lo)

        # 1차 매칭: 전체 트랙 ↔ 고신뢰도
        m1, unm_t, unm_d1 = _match(self._tracks, hi_b, self.iou_th)
        for ti, di in m1:
            t = self._tracks[ti]
            t.box, t.conf, t.hits, t.age, t.lost = hi_b[di], hi[di].conf, t.hits+1, t.age+1, 0

        # 2차 매칭: 미매칭 트랙 ↔ 저신뢰도
        pool = [self._tracks[i] for i in unm_t]
        m2, unm_t2, _ = _match(pool, lo_b, self.iou_th)
        for ti, di in m2:
            t = pool[ti]
            t.box, t.conf, t.hits, t.age, t.lost = lo_b[di], lo[di].conf, t.hits+1, t.age+1, 0

        # 미매칭 트랙 → lost++
        for i in unm_t2:
            pool[i].lost += 1
            pool[i].age  += 1

        # 미매칭 고신뢰도 검출 → 새 트랙
        for di in unm_d1:
            self._tracks.append(_T(id=self._next_id, box=hi_b[di], conf=hi[di].conf, hits=1, age=1))
            self._next_id += 1

        # max_lost 초과 트랙 제거
        self._tracks = [t for t in self._tracks if t.lost <= self.max_lost]

        # min_hits 충족 + 현재 프레임 매칭된 트랙만 반환
        return [
            Track(
                track_id=t.id,
                bbox=BBoxXYXY(int(t.box[0]), int(t.box[1]), int(t.box[2]), int(t.box[3])),
                age=t.age,
                hits=t.hits,
                conf=t.conf,
            )
            for t in self._tracks if t.hits >= self.min_hits and t.lost == 0
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
