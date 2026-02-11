# OpenCV cv2.VideoCapture 래핑
# read()로 (ret, frame, timestamp_ms, frame_idx) 반환
# FPS/해상도/총프레임 같은 메타도 제공

from __future__ import annotations
import time
from typing import Iterator, Tuple, Union
import cv2
from src.utils.types import FrameMeta


class VideoSource:
    """
    OpenCV VideoCapture 래퍼.
    프레임과 함께 FrameMeta를 같이 뱉어주기 위해 존재.
    """

    def __init__(self, source: Union[int, str]):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._frame_idx = 0
        self._start_time = time.time()

    def read(self) -> Tuple[bool, "cv2.Mat", FrameMeta]:
        ok, frame = self.cap.read()
        if not ok:
            meta = FrameMeta(
                frame_idx=self._frame_idx,
                ts_ms=int((time.time() - self._start_time) * 1000),
                fps=self.fps if self.fps > 0 else 0.0,
                width=self.width,
                height=self.height,
            )
            return False, frame, meta

        ts_ms = int((time.time() - self._start_time) * 1000)
        meta = FrameMeta(
            frame_idx=self._frame_idx,
            ts_ms=ts_ms,
            fps=self.fps if self.fps > 0 else 0.0,
            width=frame.shape[1],
            height=frame.shape[0],
        )
        self._frame_idx += 1
        return True, frame, meta

    def release(self) -> None:
        self.cap.release()
