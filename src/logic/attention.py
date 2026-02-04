from typing import Dict, Any

def is_attending(headpose: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    """Example attention rule using head pose thresholds only.

    Later you can combine:
    - gaze vector direction
    - head pose
    - ROI projection / screen geometry
    """
    yaw = float(headpose.get("yaw", 0.0))
    pitch = float(headpose.get("pitch", 0.0))

    max_yaw = float(cfg.get("max_yaw_abs_deg", 25))
    max_pitch = float(cfg.get("max_pitch_abs_deg", 20))

    return (abs(yaw) <= max_yaw) and (abs(pitch) <= max_pitch)
