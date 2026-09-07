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
        self.cap = cv2.VideoCapture(source) #int or str(경로)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # 라이브 카메라(source가 정수)는 CAP_PROP_POS_MSEC가 실시간과 안 맞으므로
        # 벽시계 기준 경과시간을 ts_ms로 사용한다. 파일 입력은 원래 타임스탬프 유지.
        self.is_live = isinstance(source, int)
        self._t0 = time.perf_counter()

        self._frame_idx = 0

    def read(self) -> Tuple[bool, "cv2.Mat", FrameMeta]:
        ok, frame = self.cap.read()     # VideoCapture의 read함수의 반환값은 ret(bool)과 frame(numpy배열->ex: 1280x720x3)
        if self.is_live:
            ts_ms = int((time.perf_counter() - self._t0) * 1000)  # 벽시계 경과시간 (ms)
        else:
            ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))  # 영상 자체 타임스탬프 (ms)
        if not ok:
            meta = FrameMeta(
                frame_idx=self._frame_idx,
                ts_ms=ts_ms,
                fps=self.fps if self.fps > 0 else 0.0,
                width=self.width,
                height=self.height,
            )
            return False, frame, meta
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
