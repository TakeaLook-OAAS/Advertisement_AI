# update(dets: List[Det]) -> List[Track]
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np

from src.utils.types import BBoxXYXY, Det, Track


# ── 공식 OC-SORT 어댑터 ──────────────────────────────────────

class OCSortAdapter:
    """
    공식 OC-SORT(OCSort)를 감싸서
    OfficialByteTrackAdapter와 동일한 인터페이스로 사용할 수 있는 어댑터.

    입력: List[Det]  →  출력: List[Track]
    """

    def __init__(self, cfg: Dict[str, Any]):
        from src.models.trackers.ocsort.ocsort import OCSort

        self._cfg = dict(cfg)
        self._tracker = OCSort(
            det_thresh=float(cfg.get("det_thresh", 0.6)),
            max_age=int(cfg.get("max_age", 30)),
            min_hits=int(cfg.get("min_hits", 3)),
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            delta_t=int(cfg.get("delta_t", 3)),
            asso_func=cfg.get("asso_func", "iou"),
            inertia=float(cfg.get("inertia", 0.2)),
            use_byte=bool(cfg.get("use_byte", False)),
        )
        self._img_h = int(cfg.get("img_h", 1080))
        self._img_w = int(cfg.get("img_w", 1920))

    def update(self, dets: List[Det]) -> List[Track]:
        if len(dets) == 0:
            output_results = np.empty((0, 5), dtype=np.float32)
        else:
            output_results = np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.conf] for d in dets],
                dtype=np.float32,
            )

        # img_info == img_size → 스케일링 없이 원본 좌표 그대로 사용 (ByteTrack 어댑터와 동일)
        img_info = [self._img_h, self._img_w]
        img_size = [self._img_h, self._img_w]

        rows = self._tracker.update(output_results, img_info, img_size)

        # OC-SORT는 [x1,y1,x2,y2,track_id]만 반환(신뢰도 컬럼 없음) → conf는 0.0 고정
        trackers_by_id = {int(t.id) + 1: t for t in self._tracker.trackers}

        tracks: List[Track] = []
        for row in rows:
            x1 = max(0, min(int(row[0]), self._img_w))
            y1 = max(0, min(int(row[1]), self._img_h))
            x2 = max(0, min(int(row[2]), self._img_w))
            y2 = max(0, min(int(row[3]), self._img_h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            track_id = int(row[4])
            trk = trackers_by_id.get(track_id)
            tracks.append(Track(
                track_id=track_id,
                bbox=BBoxXYXY(x1, y1, x2, y2),
                conf=0.0,
                lifetime=trk.age if trk is not None else 0,
                hits=trk.hits if trk is not None else 0,
            ))
        return tracks

    def reset(self) -> None:
        from src.models.trackers.ocsort.ocsort import OCSort

        self._tracker = OCSort(
            det_thresh=float(self._cfg.get("det_thresh", 0.6)),
            max_age=int(self._cfg.get("max_age", 30)),
            min_hits=int(self._cfg.get("min_hits", 3)),
            iou_threshold=float(self._cfg.get("iou_threshold", 0.3)),
            delta_t=int(self._cfg.get("delta_t", 3)),
            asso_func=self._cfg.get("asso_func", "iou"),
            inertia=float(self._cfg.get("inertia", 0.2)),
            use_byte=bool(self._cfg.get("use_byte", False)),
        )
