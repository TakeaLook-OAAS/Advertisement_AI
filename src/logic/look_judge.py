# gaze 벡터로 카메라 정면(화면 정중앙)을 보고 있는지 판정
# 코사인 유사도 기반: gaze 벡터와 카메라 정면 벡터 [0, 0, -1]의 각도 비교

from __future__ import annotations
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional
from src.utils.types import Gaze, LookResult, Track


# 카메라 정면 방향 (OpenVINO 좌표계: z가 카메라에서 멀어지는 방향)
_FRONT = (0.0, 0.0, -1.0)


class LookJudge:
    """
    gaze 벡터와 카메라 정면의 각도를 비교해서 '보고 있는지' 판정.

    distance_adaptive가 활성화되면, 얼굴 crop 높이(px)로 거리를 추정하고
    광고판 실제 폭(ad_width_m)이 그 거리에서 차지하는 각도로 threshold_deg를 매 프레임 재계산한다.
    비활성화(기본값)면 기존과 동일하게 고정 threshold_deg를 사용한다.

    또한 얼굴 bbox가 화면 중앙에서 얼마나 벗어나 있는지(카메라 화각 기준 좌우/상하 각도)를 계산해서,
    비교 기준 방향을 카메라 정면(0,0,-1) 대신 "그 사람 위치에서 광고판을 보는 방향"으로 이동시킨다.
    (카메라 정면만 기준으로 삼으면, 화면 가장자리에 있는 사람은 실제로 광고판을 보고 있어도
    광고판이 있는 방향과 어긋나 오판정될 수 있음)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.threshold_deg = float(cfg.get("threshold_deg", 30.0))

        da_cfg = cfg.get("distance_adaptive", {})
        self.distance_adaptive_enabled = bool(da_cfg.get("enabled", False))
        self.ad_width_m = float(da_cfg.get("ad_width_m", 1.0))
        self.ad_height_m = float(da_cfg.get("ad_height_m", 1.0))
        self.ref_distance_m = float(da_cfg.get("ref_distance_m", 2.0))
        self.ref_face_height_px = float(da_cfg.get("ref_face_height_px", 30.0))
        self.min_threshold_deg = float(da_cfg.get("min_threshold_deg", 5.0))
        self.max_threshold_deg = float(da_cfg.get("max_threshold_deg", 45.0))
        # bbox 중심 픽셀 위치 -> 좌우/상하 offset 각도로 변환할 때 쓰는 카메라 화각
        self.h_fov_deg = float(da_cfg.get("h_fov_deg", 70.0))
        self.v_fov_deg = float(da_cfg.get("v_fov_deg", 50.0))

        # 히스테리시스: 프레임 단위 raw 판정의 노이즈를 완충 (진입/이탈에 각각 다른 연속 프레임 수 요구)
        hyst_cfg = cfg.get("hysteresis", {})
        self.hysteresis_enabled = bool(hyst_cfg.get("enabled", True))
        self.enter_frames = int(hyst_cfg.get("enter_frames", 3))
        self.exit_frames = int(hyst_cfg.get("exit_frames", 8))
        self._hysteresis_state: Dict[int, Dict[str, Any]] = {}  # track_id -> {smoothed, pending, streak}

    def _estimate_distance_m(self, face_height_px: float) -> float:
        """얼굴 crop 높이(px)로 거리(m)를 추정. 1점 보정(ref_distance_m/ref_face_height_px) 기반 역비례."""
        if face_height_px <= 0:
            return self.ref_distance_m
        return self.ref_distance_m * self.ref_face_height_px / face_height_px

    def _thresholds_for_distance(self, distance_m: float) -> tuple[float, float]:
        """
        광고판 실제 폭/높이가 해당 거리에서 차지하는 각도를 (수평, 수직) threshold로 사용.
        정사각형이 아닌 광고판(가로/세로 비율이 다른 경우)을 위해 따로 계산 (min/max로 각각 clamp).
        """
        h_deg = math.degrees(math.atan((self.ad_width_m / 2.0) / distance_m))
        v_deg = math.degrees(math.atan((self.ad_height_m / 2.0) / distance_m))
        h_deg = max(self.min_threshold_deg, min(self.max_threshold_deg, h_deg))
        v_deg = max(self.min_threshold_deg, min(self.max_threshold_deg, v_deg))
        return h_deg, v_deg

    def _offset_deg_from_center(
        self, cx: float, cy: float, frame_w: int, frame_h: int
    ) -> tuple[float, float]:
        """
        얼굴 bbox 중심 픽셀 좌표 -> 카메라 광축 기준 좌우/상하 offset 각도(부호 있음).
        offset_deg_x: 화면 오른쪽이면 +, 왼쪽이면 -
        offset_deg_y: 화면 아래쪽이면 +, 위쪽이면 -
        (화면 중앙이면 둘 다 0)
        """
        norm_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
        norm_y = (cy - frame_h / 2.0) / (frame_h / 2.0)
        offset_deg_x = norm_x * (self.h_fov_deg / 2.0)
        offset_deg_y = norm_y * (self.v_fov_deg / 2.0)
        return offset_deg_x, offset_deg_y

    def judge(
        self,
        gaze: Gaze,
        face_height_px: Optional[float] = None,
        offset_deg_x: float = 0.0,
        offset_deg_y: float = 0.0,
    ) -> LookResult:
        gx, gy, gz = gaze.x, gaze.y, gaze.z
        fx, fy, fz = _FRONT

        dot = gx * fx + gy * fy + gz * fz
        mag_g = math.sqrt(gx * gx + gy * gy + gz * gz)

        if mag_g < 1e-9:
            return LookResult(is_looking=False, score=0.0, angle_deg=180.0, threshold_deg_used=self.threshold_deg)

        cos_sim = dot / mag_g                      # mag_f = 1.0
        cos_sim = max(-1.0, min(1.0, cos_sim))     # 부동소수점 오차 방지

        angle_deg = math.degrees(math.acos(cos_sim))  # 진단/표시용 (원뿔 기준 종합 각도, 판정 방식과 무관하게 항상 계산)

        if self.distance_adaptive_enabled and face_height_px is not None:
            # 거리 적응형: 가로/세로를 분해해서 사각형 허용 범위로 판정 (원뿔이 아니라 사각뿔)
            distance_m = self._estimate_distance_m(face_height_px)
            h_threshold, v_threshold = self._thresholds_for_distance(distance_m)

            horizontal_deg = math.degrees(math.atan2(gx, -gz))
            vertical_deg = math.degrees(math.atan2(gy, -gz))

            # 카메라 정면(0,0,-1)이 아니라, 이 사람 위치에서 광고판을 보는 방향을 기준으로 비교.
            # 화면 오른쪽에 있는 사람(offset_deg_x>0)은 광고판이 상대적으로 왼쪽에 있으므로
            # target_h_deg가 음수 쪽으로 이동 (왼쪽을 봐야 "본다"로 판정됨).
            target_h_deg = -offset_deg_x
            target_v_deg = offset_deg_y

            is_looking = (
                abs(horizontal_deg - target_h_deg) <= h_threshold
                and abs(vertical_deg - target_v_deg) <= v_threshold
            )

            return LookResult(
                is_looking=is_looking,
                score=cos_sim,
                angle_deg=angle_deg,
                threshold_deg_used=h_threshold,
                threshold_deg_vertical_used=v_threshold,
                distance_m=distance_m,
            )

        return LookResult(
            is_looking=angle_deg <= self.threshold_deg,
            score=cos_sim,
            angle_deg=angle_deg,
            threshold_deg_used=self.threshold_deg,
        )

    def _apply_hysteresis(self, track_id: int, raw_is_looking: bool) -> bool:
        """
        raw 판정을 track_id별 상태로 완충.
        상태 전환에는 raw가 enter_frames(진입)/exit_frames(이탈)번 연속돼야 함.
        """
        state = self._hysteresis_state.setdefault(
            track_id, {"smoothed": raw_is_looking, "pending": None, "streak": 0}
        )

        if raw_is_looking == state["smoothed"]:
            state["pending"] = None
            state["streak"] = 0
            return state["smoothed"]

        if state["pending"] == raw_is_looking:
            state["streak"] += 1
        else:
            state["pending"] = raw_is_looking
            state["streak"] = 1

        needed = self.enter_frames if raw_is_looking else self.exit_frames
        if state["streak"] >= needed:
            state["smoothed"] = raw_is_looking
            state["pending"] = None
            state["streak"] = 0

        return state["smoothed"]

    def judge_track(self, track: Track, frame_w: int = 0, frame_h: int = 0) -> Track:
        """track.gaze로 판정하여 track.look_result에 채웁니다. gaze가 없으면(측정 실패) None으로 둡니다."""
        if track.gaze is None:
            track.look_result = None
            return track

        face_height_px = track.crop_bbox.h() if track.crop_bbox is not None else None

        offset_deg_x, offset_deg_y = 0.0, 0.0
        if self.distance_adaptive_enabled and track.crop_bbox is not None and frame_w > 0 and frame_h > 0:
            cx, cy = track.crop_bbox.center()
            offset_deg_x, offset_deg_y = self._offset_deg_from_center(cx, cy, frame_w, frame_h)

        result = self.judge(track.gaze, face_height_px, offset_deg_x, offset_deg_y)

        if self.hysteresis_enabled:
            smoothed = self._apply_hysteresis(track.track_id, result.is_looking)
            if smoothed != result.is_looking:
                result = replace(result, is_looking=smoothed, raw_is_looking=result.is_looking)
            else:
                result = replace(result, raw_is_looking=result.is_looking)

        track.look_result = result
        return track

    def judge_batch(self, tracks: List[Track], frame_w: int = 0, frame_h: int = 0) -> List[Track]:
        """각 track.gaze로 판정하여 track.look_result에 채웁니다."""
        result = [self.judge_track(t, frame_w, frame_h) for t in tracks]

        if self.hysteresis_enabled:
            active_ids = {t.track_id for t in tracks}
            for stale_id in list(self._hysteresis_state.keys()):
                if stale_id not in active_ids:
                    del self._hysteresis_state[stale_id]

        return result
