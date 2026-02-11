# update(dets)->List[Track]

from __future__ import annotations
from typing import Any, Dict, List
from src.utils.types import Det, Track


class ByteTrackTracker:
    """
    ByteTrack 어댑터(계약서):
      update(dets) -> List[Track]

    지금은 stub.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._next_id = 1

    def update(self, dets: List[Det]) -> List[Track]:
        # TODO: 실제 ByteTrack 붙이면 dets를 트래커 입력으로 변환하고
        # 트래커 출력 -> Track으로 정규화해서 반환
        #
        # 지금은 dets를 그냥 "각 det를 하나의 트랙"이라고 가정한 더미 예시
        # (YOLO 연결 후 화면에 track_id라도 보이게 하려고)
        tracks: List[Track] = []
        for d in dets:
            tracks.append(Track(track_id=self._next_id, bbox=d.bbox, conf=d.conf, age=1, hits=1))
            self._next_id += 1
        return tracks
