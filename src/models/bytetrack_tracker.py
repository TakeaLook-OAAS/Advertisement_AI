# update(dets) -> List[Track]

from __future__ import annotations
from typing import Any, Dict, List

from src.utils.types import BBoxXYXY, Det, Track


def _iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    """두 BBoxXYXY 간 IoU(Intersection over Union) 계산."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = a.w() * a.h()
    area_b = b.w() * b.h()
    return inter / (area_a + area_b - inter)


class ByteTrackTracker:
    """
    단순 IoU 기반 트래커.

    알고리즘 (매 프레임):
      1. 이전 프레임의 트랙들과 새 검출(Det)을 IoU로 매칭
      2. IoU가 threshold 이상인 쌍을 greedy 매칭 (가장 높은 IoU 우선)
      3. 매칭된 트랙 → bbox/conf 갱신, hits +1
      4. 매칭 안 된 검출 → 새 트랙 생성
      5. 매칭 안 된 트랙 → age +1, max_lost 초과 시 제거

    입출력:
      update(dets: List[Det]) -> List[Track]
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.match_thresh: float = float(cfg.get("match_thresh", 0.3))
        self.max_lost: int = int(cfg.get("max_lost_frames", 30))

        self._tracks: List[Track] = []       # 현재 활성 트랙
        self._lost_age: Dict[int, int] = {}  # track_id -> 매칭 안 된 연속 프레임 수
        self._next_id: int = 1

    def update(self, dets: List[Det]) -> List[Track]:
        """
        한 프레임의 검출 결과를 받아 트래킹 수행.

        1) 기존 트랙 vs 새 검출 IoU 매칭
        2) 매칭된 트랙 갱신
        3) 매칭 안 된 검출 → 새 트랙
        4) 매칭 안 된 트랙 → lost 처리
        """
        # ── 1. IoU 매칭 (greedy: IoU 큰 순서대로) ──────────────────────
        pairs = []  # (iou, track_idx, det_idx)
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(dets):
                score = _iou(t.bbox, d.bbox)
                if score >= self.match_thresh:
                    pairs.append((score, ti, di))

        pairs.sort(reverse=True)  # IoU 높은 쌍 우선

        matched_t = set()
        matched_d = set()
        matches = []  # (track_idx, det_idx)

        for _, ti, di in pairs:
            if ti not in matched_t and di not in matched_d:
                matches.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        # ── 2. 매칭된 트랙 갱신 ────────────────────────────────────────
        new_tracks: List[Track] = []

        for ti, di in matches:
            old = self._tracks[ti]
            d = dets[di]
            new_tracks.append(Track(
                track_id=old.track_id,
                bbox=d.bbox,
                age=old.age + 1,
                hits=old.hits + 1,
                conf=d.conf,
                in_roi=old.in_roi,
                dwell_frames=old.dwell_frames,
            ))
            self._lost_age.pop(old.track_id, None)

        # ── 3. 매칭 안 된 검출 → 새 트랙 생성 ─────────────────────────
        for di, d in enumerate(dets):
            if di not in matched_d:
                new_tracks.append(Track(
                    track_id=self._next_id,
                    bbox=d.bbox,
                    age=0,
                    hits=1,
                    conf=d.conf,
                ))
                self._next_id += 1

        # ── 4. 매칭 안 된 기존 트랙 → lost 처리 ───────────────────────
        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                lost_count = self._lost_age.get(t.track_id, 0) + 1
                if lost_count <= self.max_lost:
                    # 아직 제거 안 함 — 위치 유지, conf 감소
                    new_tracks.append(Track(
                        track_id=t.track_id,
                        bbox=t.bbox,
                        age=t.age + 1,
                        hits=t.hits,
                        conf=t.conf * 0.9,
                        in_roi=t.in_roi,
                        dwell_frames=t.dwell_frames,
                    ))
                    self._lost_age[t.track_id] = lost_count
                else:
                    # max_lost 초과 → 완전 제거
                    self._lost_age.pop(t.track_id, None)

        self._tracks = new_tracks
        return list(self._tracks)

    def reset(self) -> None:
        """트래커 상태 초기화."""
        self._tracks.clear()
        self._lost_age.clear()
        self._next_id = 1
