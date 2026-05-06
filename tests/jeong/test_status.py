# 프레임 간 상태 변화를 감지하고
# track_id 기준 세그먼트별 JSON 통계를 만드는 클래스
# runner.py에서 매 프레임 update()를 호출한다.

from __future__ import annotations

import json
import os
from typing import Dict, List, Any

from src.utils.types import (
    AdSegmentInfo,
    FrameMeta,
    LookInterval,
    PersonState,
    Track,
)


class Test_StatusTracker:
    """
    매 프레임 tracks를 받아서 track_id별 상태를 추적한다.

    최종 목표:
    - exposure: 영상에 들어왔다가 나간 시간
    - look_times: 본 시간 구간들
    - age_group
    - gender
    """

    def __init__(self) -> None:
        self._states: Dict[int, PersonState] = {}
        self._frame_interval_ms: int = 33  # 기본값(약 30fps)
        self._segment_start_ms: int = 0    # 현재 세그먼트의 시작 시간
        self._device_id: str = ""          # 카메라 식별자

    def set_device_id(self, device_id: str) -> None:
        self._device_id = device_id

    def _update_frame_interval(self, meta: FrameMeta) -> None:
        if meta.fps > 0:
            self._frame_interval_ms = max(1, int(round(1000.0 / meta.fps)))

    # ──────────────────────────────────────────────
    # 매 프레임 호출
    # ──────────────────────────────────────────────

    def update(self, meta: FrameMeta, tracks: List[Track]) -> None:
        self._update_frame_interval(meta)

        active_ids = set()

        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)

            # 새로운 사람이면 PersonState 생성
            if tid not in self._states:
                self._states[tid] = PersonState(
                    track_id=tid,
                    first_seen_ms=meta.ts_ms,
                    last_seen_ms=meta.ts_ms,
                )

            state = self._states[tid]

            # 1) 시간 갱신
            state.last_seen_ms = meta.ts_ms + self._frame_interval_ms

            # 2) 인구통계 갱신
            if track.attr is not None:
                age_group_value = getattr(track.attr, "age_group", None)
                if age_group_value is not None:
                    state.age_group = getattr(age_group_value, "value", str(age_group_value))

                gender_value = getattr(track.attr, "gender", None)
                if gender_value is not None:
                    state.gender = getattr(gender_value, "value", str(gender_value))

            # 3) bbox center 기록
            bbox_center = track.bbox.center()

            # 4) 시선 변화 감지
            now_looking = bool(track.look_result and track.look_result.is_looking)

            if now_looking and not state.is_looking:
                # False → True: 보기 시작
                state.current_look_start_ms = meta.ts_ms
                state._look_start_center = bbox_center  # type: ignore[attr-defined]

            elif (not now_looking) and state.is_looking:
                # True → False: 보기 끝남 → 구간 저장
                if state.current_look_start_ms is not None:
                    state.look_intervals.append(
                        LookInterval(
                            start_ms=state.current_look_start_ms,
                            end_ms=meta.ts_ms,
                            start_center=getattr(state, '_look_start_center', (0, 0)),
                            end_center=bbox_center,
                        )
                    )
                    state.current_look_start_ms = None

            state.is_looking = now_looking

        # 5) 사라진 사람 처리
        vanished_ids = []
        for tid in self._states:
            if tid not in active_ids and self._states[tid].is_active:
                vanished_ids.append(tid)

        for tid in vanished_ids:
            state = self._states[tid]
            end_ms = state.last_seen_ms

            # 보다가 사라짐 → 시선 구간 강제 종료 (마지막 bbox center 사용)
            if state.is_looking and state.current_look_start_ms is not None:
                last_center = getattr(state, '_look_start_center', (0, 0))
                state.look_intervals.append(
                    LookInterval(
                        start_ms=state.current_look_start_ms,
                        end_ms=end_ms,
                        start_center=last_center,
                        end_center=last_center,  # 사라진 시점이라 마지막 known center 사용
                    )
                )
                state.current_look_start_ms = None

            state.is_looking = False
            state.is_active = False

    # ──────────────────────────────────────────────
    # 세그먼트 경계에서 호출
    # ──────────────────────────────────────────────

    def flush_segment(self, segment_info: AdSegmentInfo) -> Dict[str, Any]:
        """
        광고 경계에서 호출. 현재까지의 상태를 세그먼트 JSON으로 반환하고,
        활성 track은 다음 세그먼트로 이월한다.
        """
        boundary_ms = segment_info.end_ms
        base_ms = self._segment_start_ms
        summaries = []

        carry_forward: List[Dict[str, Any]] = []

        for tid in sorted(self._states.keys()):
            state = self._states[tid]

            # 열린 시선 구간 → boundary에서 강제 종료
            if state.current_look_start_ms is not None:
                last_center = getattr(state, '_look_start_center', (0, 0))
                state.look_intervals.append(
                    LookInterval(
                        start_ms=state.current_look_start_ms,
                        end_ms=boundary_ms,
                        start_center=last_center,
                        end_center=last_center,
                    )
                )
                state.current_look_start_ms = None

            was_active = state.is_active

            # summary 생성
            summary = self._to_summary(state, base_ms)
            summaries.append(summary)

            # 활성 track은 다음 세그먼트로 이월
            if was_active:
                carry_forward.append({
                    "track_id": tid,
                    "age_group": state.age_group,
                    "gender": state.gender,
                    "in_roi": state.in_roi,
                    "is_looking": state.is_looking,
                })

        # _states 초기화 후 이월 track 재생성
        self._states.clear()
        for info in carry_forward:
            tid = info["track_id"]
            new_state = PersonState(
                track_id=tid,
                first_seen_ms=boundary_ms,
                last_seen_ms=boundary_ms,
                in_roi=info["in_roi"],
                is_looking=info["is_looking"],
                age_group=info["age_group"],
                gender=info["gender"],
            )
            if info["is_looking"]:
                new_state.current_look_start_ms = boundary_ms
            self._states[tid] = new_state

        self._segment_start_ms = boundary_ms

        return {
            "segment": {
                "device_id": self._device_id,
                "index": segment_info.segment_index,
                "cycle_index": segment_info.cycle_index,
                "timestamp": segment_info.wall_start,
                "duration_ms": segment_info.end_ms - segment_info.start_ms,
            },
            "tracks": summaries,
        }

    # ──────────────────────────────────────────────
    # 영상 종료 시 호출
    # ──────────────────────────────────────────────

    def finalize(self) -> None:
        """아직 열린 시선 구간과 활성 track을 정리한다."""
        for state in self._states.values():
            if not state.is_active:
                continue
            end_ms = state.last_seen_ms
            if state.current_look_start_ms is not None:
                last_center = getattr(state, '_look_start_center', (0, 0))
                state.look_intervals.append(
                    LookInterval(
                        start_ms=state.current_look_start_ms,
                        end_ms=end_ms,
                        start_center=last_center,
                        end_center=last_center,
                    )
                )
                state.current_look_start_ms = None
            state.is_active = False

    # ──────────────────────────────────────────────
    # JSON 변환
    # ──────────────────────────────────────────────

    def _to_summary(self, state: PersonState, base_ms: int) -> Dict[str, Any]:
        """PersonState → JSON dict. 타임스탬프는 base_ms 기준 상대값으로 변환."""
        start = max(0, state.first_seen_ms - base_ms)
        end = max(0, state.last_seen_ms - base_ms)

        # 500ms 미만의 시선 구간은 노이즈로 제외
        look_times = []
        for interval in state.look_intervals:
            s = max(0, interval.start_ms - base_ms)
            e = max(0, interval.end_ms - base_ms)
            if e - s >= 500:  # look_judge가 True인 기간이 0.5초 이상일 때만 기록
                look_times.append({
                    "start_ms": s,
                    "end_ms": e,
                    "start_center": list(interval.start_center),
                    "end_center": list(interval.end_center),
                })

        return {
            "track_id": state.track_id,
            "exposure": {
                "start_ms": start,
                "end_ms": end,
            },
            "look_times": look_times,
            "total_look_duration_ms": sum(lt["end_ms"] - lt["start_ms"] for lt in look_times),
            "age_group": state.age_group,
            "gender": state.gender,
        }

    @staticmethod
    def save_segment_json(output_path: str, segment_data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segment_data, f, ensure_ascii=False, indent=2)


'''
세그먼트 JSON 결과 예시

{
  "segment": {
    "device_id": "cam_01",
    "index": 0,
    "cycle_index": 0,
    "timestamp": "2026-03-21T14:30:00+00:00",
    "duration_ms": 5000
  },
  "tracks": [
    {
      "track_id": 1,
      "exposure": {"start_ms": 0, "end_ms": 5000},
      "look_times": [
        {"start_ms": 200, "end_ms": 800, "start_center": [500, 300], "end_center": [520, 310]},
        {"start_ms": 1500, "end_ms": 3000, "start_center": [480, 290], "end_center": [510, 305]}
      ],
      "total_look_duration_ms": 2100,
      "age_group": "20-29",
      "gender": "female"
    },
    {
      "track_id": 2,
      "exposure": {"start_ms": 1000, "end_ms": 4000},
      "look_times": [
        {"start_ms": 1200, "end_ms": 1800, "start_center": [620, 410], "end_center": [600, 400]},
        {"start_ms": 2500, "end_ms": 3500, "start_center": [580, 390], "end_center": [590, 380]}
      ],
      "total_look_duration_ms": 1600,
      "age_group": "30-39",
      "gender": "male"
    }
  ]
}
'''
