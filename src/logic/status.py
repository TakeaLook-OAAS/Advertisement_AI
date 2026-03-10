# 프레임 간 상태 변화를 감지해서 이벤트(Event)를 발행하는 클래스
# runner.py에서 매 프레임 update()를 호출한다.

from __future__ import annotations

from typing import Dict, List, Set

from loguru import logger

from src.pipeline.orchestrator import OrchestratorOutput
from src.utils.types import Event, EventType, FrameMeta, LookResult


class StatusTracker:
    """
    OrchestratorOutput + FrameMeta를 받아서
    이전 프레임 대비 상태 변화를 감지 → Event 발행.

    감지하는 이벤트:
    - enter_roi / exit_roi  (in_roi 변화)
    - look_start / look_end (is_looking 변화)
    - pass_by               (track이 사라졌을 때)
    """

    def __init__(self) -> None:
        # 이전 프레임 상태 기억
        self._prev_in_roi: Dict[int, bool] = {}          # track_id → in_roi
        self._prev_looking: Dict[int, bool] = {}          # track_id → is_looking
        self._prev_dwell: Dict[int, int] = {}             # track_id → dwell_frames (exit 시 기록용)
        self._seen_ids: Set[int] = set()                  # 한 번이라도 등장한 track_id
        self.events: List[Event] = []                     # 누적 이벤트 로그

    def update(self, out: OrchestratorOutput, meta: FrameMeta) -> List[Event]:
        """상태 변화 감지 → 이벤트 발행. 이번 프레임에서 발생한 이벤트 리스트 반환."""
        new_events: List[Event] = []
        active_ids: Set[int] = set()

        # look_results와 tracks는 같은 인덱스 대응
        look_map: Dict[int, LookResult] = {}
        for i, track in enumerate(out.tracks):
            if i < len(out.look_results):
                look_map[track.track_id] = out.look_results[i]

        for track in out.tracks:
            tid = track.track_id
            active_ids.add(tid)
            self._seen_ids.add(tid)

            # ── ROI 상태 변화 ──
            was_in_roi = self._prev_in_roi.get(tid, False)
            now_in_roi = track.in_roi

            if not was_in_roi and now_in_roi:
                new_events.append(Event(
                    ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                    track_id=tid, type=EventType.enter_roi, payload={},
                ))
            elif was_in_roi and not now_in_roi:
                new_events.append(Event(
                    ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                    track_id=tid, type=EventType.exit_roi,
                    payload={"dwell_frames": self._prev_dwell.get(tid, 0)},
                ))

            self._prev_in_roi[tid] = now_in_roi
            self._prev_dwell[tid] = track.dwell_frames

            # ── 시선 상태 변화 ──
            lr = look_map.get(tid)
            if lr is not None:
                was_looking = self._prev_looking.get(tid, False)
                now_looking = lr.is_looking

                if not was_looking and now_looking:
                    new_events.append(Event(
                        ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                        track_id=tid, type=EventType.look_start,
                        payload={"score": lr.score, "angle_deg": lr.angle_deg},
                    ))
                elif was_looking and not now_looking:
                    new_events.append(Event(
                        ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                        track_id=tid, type=EventType.look_end,
                        payload={"score": lr.score, "angle_deg": lr.angle_deg},
                    ))

                self._prev_looking[tid] = now_looking

        # ── 사라진 track 처리 ──
        disappeared = set(self._prev_in_roi.keys()) - active_ids
        for tid in disappeared:
            # ROI 안에 있다가 사라진 경우 exit_roi 발행
            if self._prev_in_roi.get(tid, False):
                new_events.append(Event(
                    ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                    track_id=tid, type=EventType.exit_roi,
                    payload={"dwell_frames": self._prev_dwell.get(tid, 0)},
                ))
            # 보고 있다가 사라진 경우 look_end 발행
            if self._prev_looking.get(tid, False):
                new_events.append(Event(
                    ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                    track_id=tid, type=EventType.look_end, payload={},
                ))
            # pass_by 발행
            new_events.append(Event(
                ts_ms=meta.ts_ms, frame_idx=meta.frame_idx,
                track_id=tid, type=EventType.pass_by, payload={},
            ))
            # 정리
            self._prev_in_roi.pop(tid, None)
            self._prev_looking.pop(tid, None)
            self._prev_dwell.pop(tid, None)

        self.events.extend(new_events)
        return new_events

    def summary(self) -> Dict[str, object]:
        """영상 종료 후 최종 통계 반환."""
        total_tracks = len(self._seen_ids)
        enter_count = sum(1 for e in self.events if e.type == EventType.enter_roi)
        look_count = sum(1 for e in self.events if e.type == EventType.look_start)

        # 체류 시간 분포 (프레임 단위)
        dwell_list = [
            e.payload.get("dwell_frames", 0)
            for e in self.events if e.type == EventType.exit_roi
        ]

        return {
            "total_tracks": total_tracks,
            "enter_roi_count": enter_count,
            "look_count": look_count,
            "dwell_frames_list": dwell_list,
            "stay_rate": enter_count / total_tracks if total_tracks > 0 else 0.0,
            "attention_rate": look_count / total_tracks if total_tracks > 0 else 0.0,
        }
