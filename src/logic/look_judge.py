# gaze 벡터로 카메라 정면(화면 정중앙)을 보고 있는지 판정
# 코사인 유사도 기반: gaze 벡터와 카메라 정면 벡터 [0, 0, -1]의 각도 비교

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from loguru import logger
from src.utils.types import Gaze, LookResult, Track


# 카메라 정면 방향 (OpenVINO 좌표계: z가 카메라에서 멀어지는 방향)
_FRONT = (0.0, 0.0, -1.0)


class LookJudge:
    """
    gaze 벡터와 카메라 정면의 각도를 비교해서 '보고 있는지' 판정.

    distance_adaptive가 활성화되면, 얼굴 crop 높이(px)로 거리를 추정하고
    광고판 실제 폭(ad_width_m)이 그 거리에서 차지하는 각도로 threshold_deg를 매 프레임 재계산한다.
    비활성화(기본값)면 기존과 동일하게 고정 threshold_deg를 사용한다.
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

    def judge(self, gaze: Gaze, face_height_px: Optional[float] = None) -> LookResult:
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
            is_looking = abs(horizontal_deg) <= h_threshold and abs(vertical_deg) <= v_threshold

            logger.debug(
                f"[LookJudge] distance_m={distance_m:.2f} h_threshold={h_threshold:.1f} "
                f"v_threshold={v_threshold:.1f} horizontal_deg={horizontal_deg:.1f} "
                f"vertical_deg={vertical_deg:.1f} is_looking={is_looking}"
            )  # TEMP: distance_adaptive 검증용

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

    def judge_track(self, track: Track) -> Track:
        """track.gaze로 판정하여 track.look_result에 채웁니다. gaze가 없으면(측정 실패) None으로 둡니다."""
        if track.gaze is None:
            track.look_result = None
        else:
            face_height_px = track.crop_bbox.h() if track.crop_bbox is not None else None
            track.look_result = self.judge(track.gaze, face_height_px)
        return track

    def judge_batch(self, tracks: List[Track]) -> List[Track]:
        """각 track.gaze로 판정하여 track.look_result에 채웁니다."""
        return [self.judge_track(t) for t in tracks]
