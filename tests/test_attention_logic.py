from src.logic.attention import is_attending

def test_attending_thresholds():
    cfg = {"max_yaw_abs_deg": 10, "max_pitch_abs_deg": 10}
    assert is_attending({"yaw": 0, "pitch": 0}, cfg) is True
    assert is_attending({"yaw": 20, "pitch": 0}, cfg) is False
