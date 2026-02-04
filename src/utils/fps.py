import time

class FPSMeter:
    """Exponential moving average FPS."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.last_t = None
        self.fps = 0.0

    def reset(self):
        self.last_t = None
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self.last_t is None:
            self.last_t = now
            return self.fps
        dt = now - self.last_t
        self.last_t = now
        inst = (1.0 / dt) if dt > 0 else 0.0
        if self.fps <= 0:
            self.fps = inst
        else:
            self.fps = (1 - self.alpha) * self.fps + self.alpha * inst
        return self.fps
