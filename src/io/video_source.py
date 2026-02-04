import cv2
from loguru import logger

class VideoSource:
    def __init__(self, source=0, width=1280, height=720, flip_horizontal=False):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self.flip_horizontal = bool(flip_horizontal)
        logger.info(f"VideoSource opened: {source} ({width}x{height}) flip={self.flip_horizontal}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
