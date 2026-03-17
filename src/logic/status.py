# 프레임 간 상태 변화를 감지해서 이벤트(Event)를 발행하고
# track_id 기준 최종 JSON 통계를 만드는 클래스
# runner.py에서 매 프레임 update()를 호출한다.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.utils.types import Event, EventType, FrameMeta, Track


@dataclass
class _LookInterval:
    start_ms: int
    end_ms: Optional[int] = None

    def close(self, end_ms: int) -> None:
        self.end_ms = end_ms

    def to_dict(self) -> Dict[str, int]:
        end_ms = self.end_ms if self.end_ms is not None else self.start_ms
        return {
            "start_ms": self.start_ms,
            "end_ms": end_ms,
            "duration_ms": max(0, end_ms - self.start_ms),
        }


@dataclass
class _TrackState:
    track_id: int
    first_seen_ms: int
    first_seen_frame: int
    last_seen_ms: int
    last_seen_frame: int

    in_roi: bool = False
    is_looking: bool = False

    current_look_start_ms: Optional[int] = None
    look_times: List[_LookInterval] = field(default_factory=list)

    age_group: Optional[str] = None
    gender: Optional[str] = None

    closed: bool = False
    exposure_end_ms: Optional[int] = None

    def update_attr(self, track: Track) -> None:
        if track.attr is None:
            return

        age_group_value = getattr(track.attr, "age_group", None)
        if age_group_value is not None:
            self.age_group = getattr(age_group_value, "value", str(age_group_value))

        gender_value = getattr(track.attr, "gender", None)
        if gender_value is not None:
            self.gender = getattr(gender_value, "value", str(gender_value))

    def close_exposure(self, end_ms: int) -> None:
        self.closed = True
        self.exposure_end_ms = end_ms

    def start_look(self, ts_ms: int) -> None:
        if self.current_look_start_ms is None:
            self.current_look_start_ms = ts_ms

    def end_look(self, ts_ms: int) -> None:
        if self.current_look_start_ms is None:
            return

        interval = _LookInterval(start_ms=self.current_look_start_ms, end_ms=ts_ms)
        self.look_times.append(interval)
        self.current_look_start_ms = None

    def ensure_open_look_closed(self, end_ms: int) -> None:
        if self.current_look_start_ms is not None:
            self.end_look(end_ms)

    def to_summary(self) -> Dict[str, Any]:
        exposure_end_ms = (
            self.exposure_end_ms
            if self.exposure_end_ms is not None
            else self.last_seen_ms
        )

        look_times_dict = [x.to_dict() for x in self.look_times]
        total_look_duration_ms = sum(x["duration_ms"] for x in look_times_dict)

        return {
            "track_id": self.track_id,
            "exposure": {
                "start_ms": self.first_seen_ms,
                "end_ms": exposure_end_ms,
                "duration_ms": max(0, exposure_end_ms - self.first_seen_ms),
            },
            "look_times": look_times_dict,
            "total_look_duration_ms": total_look_duration_ms,
            "age_group": self.age_group,
            "gender": self.gender,
        }


class StatusTracker:
    """
    매 프레임 tracks를 받아서 track_id별 상태를 추적한다.

    최종 목표:
    - exposure: 영상에 들어왔다가 나간 시간
    - look_times: 본 시간 구간들
    - total_look_duration_ms
    - age
    - gender
    """

    def __init__(self) -> None:
        self._states: Dict[int, _TrackState] = {}
        self._frame_interval_ms: int = 33  # 기본값(약 30fps)

    def _update_frame_interval(self, meta: FrameMeta) -> None:
        if meta.fps and meta.fps > 0:
            self._frame_interval_ms = max(1, int(round(1000.0 / meta.fps)))

    def update(self, meta: FrameMeta, tracks: List[Track]) -> List[Event]:
        self._update_frame_interval(meta)
        events: List[Event] = []

        active_ids = set()

        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)

            if tid not in self._states:
                self._states[tid] = _TrackState(
                    track_id=tid,
                    first_seen_ms=meta.ts_ms,
                    first_seen_frame=meta.frame_idx,
                    last_seen_ms=meta.ts_ms,
                    last_seen_frame=meta.frame_idx,
                )

            state = self._states[tid]
            state.last_seen_ms = meta.ts_ms + self._frame_interval_ms
            state.last_seen_frame = meta.frame_idx
            state.update_attr(track)

            # ROI 이벤트
            now_in_roi = bool(track.roi and track.roi.in_roi)
            if now_in_roi and not state.in_roi:
                events.append(
                    Event(
                        ts_ms=meta.ts_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.enter_roi,
                        payload={},
                    )
                )
            elif (not now_in_roi) and state.in_roi:
                events.append(
                    Event(
                        ts_ms=meta.ts_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.exit_roi,
                        payload={},
                    )
                )
            state.in_roi = now_in_roi

            # look 이벤트
            now_looking = bool(track.look_result and track.look_result.is_looking)

            if now_looking and not state.is_looking:
                state.start_look(meta.ts_ms)
                events.append(
                    Event(
                        ts_ms=meta.ts_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.look_start,
                        payload={},
                    )
                )

            elif (not now_looking) and state.is_looking:
                look_end_ms = meta.ts_ms
                state.end_look(look_end_ms)
                events.append(
                    Event(
                        ts_ms=look_end_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.look_end,
                        payload={},
                    )
                )

            state.is_looking = now_looking

        # 이번 프레임에 사라진 track들 종료 처리
        vanished_ids = [tid for tid in self._states.keys() if tid not in active_ids and not self._states[tid].closed]

        for tid in vanished_ids:
            state = self._states[tid]
            end_ms = state.last_seen_ms

            if state.is_looking:
                state.end_look(end_ms)
                events.append(
                    Event(
                        ts_ms=end_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.look_end,
                        payload={"reason": "track_disappeared"},
                    )
                )
                state.is_looking = False

            if state.in_roi:
                events.append(
                    Event(
                        ts_ms=end_ms,
                        frame_idx=meta.frame_idx,
                        track_id=tid,
                        type=EventType.exit_roi,
                        payload={"reason": "track_disappeared"},
                    )
                )
                state.in_roi = False

            state.close_exposure(end_ms)

        return events

    def finalize(self) -> None:
        """
        스트림 종료 시 아직 열린 look interval을 정리한다.
        """
        for state in self._states.values():
            if state.closed:
                state.ensure_open_look_closed(state.exposure_end_ms or state.last_seen_ms)
                continue

            end_ms = state.last_seen_ms
            state.ensure_open_look_closed(end_ms)
            state.close_exposure(end_ms)

    def get_results(self) -> List[Dict[str, Any]]:
        results = [state.to_summary() for _, state in sorted(self._states.items(), key=lambda x: x[0])]
        return results

    def save_json(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        payload = {
            "tracks": self.get_results()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


'''
JSON 결과 예시

[{
  'track_id': 1,
  'exposure': {'start_ms': 0, 'end_ms': 3033, 'duration_ms': 3033},
  'look_times': [{'start_ms': 1000, 'end_ms': 3000, 'duration_ms': 2000}],
  'total_look_duration_ms': 2000,
  'age_group': 'young_adult',
  'gender': 'female'
}]
'''