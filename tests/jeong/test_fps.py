from src.utils.fps import FPSMeter
import time

def test_fps_meter_runs():
    m = FPSMeter(alpha=0.5)
    m.update()
    time.sleep(0.01)
    fps = m.update()
    assert fps >= 0
