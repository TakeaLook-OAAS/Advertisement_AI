# 관심도(Attention) 판정 로직
# yaw/pitch 임계값 기반으로 광고판을 보고 있는지 판정

from __future__ import annotations
from typing import Any, Dict


def is_attending(headpose: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    """
    head pose의 yaw/pitch 값이 임계값 이내이면 '광고를 보고 있다'고 판정합니다.

    Args:
        headpose: {"yaw": float, "pitch": float} (degrees)
        cfg: {"max_yaw_abs_deg": float, "max_pitch_abs_deg": float}

    Returns:
        True면 광고를 보고 있음, False면 안 보고 있음
    """
    max_yaw = float(cfg.get("max_yaw_abs_deg", 25))
    max_pitch = float(cfg.get("max_pitch_abs_deg", 20))

    yaw = abs(float(headpose.get("yaw", 0)))
    pitch = abs(float(headpose.get("pitch", 0)))

    return yaw <= max_yaw and pitch <= max_pitch
