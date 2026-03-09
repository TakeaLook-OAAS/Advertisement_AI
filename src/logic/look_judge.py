# gaze 벡터로 카메라 정면(화면 정중앙)을 보고 있는지 판정
# 코사인 유사도 기반: gaze 벡터와 카메라 정면 벡터 [0, 0, -1]의 각도 비교

from __future__ import annotations
import math
from typing import Any, Dict, List
from src.utils.types import Gaze, LookResult


# 카메라 정면 방향 (OpenVINO 좌표계: z가 카메라에서 멀어지는 방향)
_FRONT = (0.0, 0.0, -1.0)


class LookJudge:
    """
    gaze 벡터와 카메라 정면의 각도를 비교해서 '보고 있는지' 판정.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.threshold_deg = float(cfg.get("threshold_deg", 30.0))

    def judge(self, gaze: Gaze) -> LookResult:
        gx, gy, gz = gaze.x, gaze.y, gaze.z
        fx, fy, fz = _FRONT

        dot = gx * fx + gy * fy + gz * fz
        mag_g = math.sqrt(gx * gx + gy * gy + gz * gz)

        if mag_g < 1e-9:
            return LookResult(is_looking=False, score=0.0, angle_deg=180.0)

        cos_sim = dot / mag_g                      # mag_f = 1.0
        cos_sim = max(-1.0, min(1.0, cos_sim))     # 부동소수점 오차 방지

        angle_deg = math.degrees(math.acos(cos_sim))

        return LookResult(
            is_looking=angle_deg <= self.threshold_deg,
            score=cos_sim,
            angle_deg=angle_deg,
        )

    def judge_batch(self, gazes: List[Gaze]) -> List[LookResult]:
        """List[Gaze] → List[LookResult]. gazes[i]는 tracks[i]와 같은 순서."""
        return [self.judge(g) for g in gazes]
