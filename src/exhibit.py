# 전시 체험용 실행 파일 (python -m src.exhibit)
# 대기 화면 → SPACE → N초 녹화(창에는 실시간 분석 화면) → 원본 전체 프레임 분석(진행률 표시) → 결과 영상 재생 → 대기 화면
# 한 세션마다 원본 / 실시간 분석 / 최종 분석 영상 3개를 output_dir에 저장한다.
# 종료: q 또는 ESC, 창 닫기

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from src.pipeline.orchestrator import Orchestrator
from src.utils.config import load_config
from src.vision.draw import draw_tracks, draw_crop_bbox, draw_headpose, draw_gaze, draw_look, draw_gender_age

WINDOW = "Advertisement AI - Exhibit"
KEY_SPACE = 32


class _Quit(Exception):
    """q/ESC/창 닫기로 프로그램 종료."""


class Camera:
    """
    웹캠을 별도 스레드에서 계속 읽는다.
    분석 루프가 느려도 녹화 중에는 카메라가 보낸 프레임이 하나도 빠지지 않고 쌓인다.
    """

    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {index}")

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0                       # 새 프레임이 들어올 때마다 +1
        self._recording = False
        self._frames: List[np.ndarray] = []
        self._ts: List[float] = []

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.perf_counter()
            with self._lock:
                self._latest = frame
                self._seq += 1
                if self._recording:
                    self._frames.append(frame)
                    self._ts.append(t)

    def latest(self) -> Tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._latest

    def start_recording(self) -> None:
        with self._lock:
            self._frames, self._ts = [], []
            self._recording = True

    def stop_recording(self) -> Tuple[List[np.ndarray], List[float]]:
        with self._lock:
            self._recording = False
            frames, ts = self._frames, self._ts
            self._frames, self._ts = [], []
        return frames, ts

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ── 화면/키 유틸 ──────────────────────────────────────────────

def _check_key(delay_ms: int = 1) -> int:
    key = cv2.waitKey(delay_ms) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        raise _Quit
    return key


def _label(img: np.ndarray, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """화면 상단에 검은 띠 + 안내 문구 (화면 표시용, 저장 영상에는 안 들어감)."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _progress_bar(img: np.ndarray, ratio: float) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 10, h - 30, w - 10, h - 10
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(img, (x1, y1), (x1 + int((x2 - x1) * ratio), y2), (0, 200, 0), -1)


def _draw_overlays(frame: np.ndarray, tracks, disp_cfg: Dict[str, Any]) -> None:
    """runner.py와 같은 display.draw_* 설정으로 분석 결과를 그린다."""
    font_scale = float(disp_cfg.get("font_scale", 0.7))
    thickness = int(disp_cfg.get("thickness", 2))
    if disp_cfg.get("draw_bbox", True):
        draw_tracks(frame, tracks, font_scale, thickness)
    if disp_cfg.get("draw_crop_bbox", True):
        draw_crop_bbox(frame, tracks, thickness)
    if disp_cfg.get("draw_headpose", True):
        draw_headpose(frame, tracks, font_scale, thickness)
    if disp_cfg.get("draw_gaze", True):
        draw_gaze(frame, tracks, font_scale, thickness)
    if disp_cfg.get("draw_look", True):
        draw_look(frame, tracks, font_scale, thickness)
    if disp_cfg.get("draw_gender_age", True):
        draw_gender_age(frame, tracks, font_scale, thickness)


def _measured_fps(ts: List[float], default: float = 30.0) -> float:
    """실제 프레임 타임스탬프 간격으로 fps 계산 → 저장 영상 길이가 실제 시간과 맞는다."""
    if len(ts) < 2 or ts[-1] <= ts[0]:
        return default
    return (len(ts) - 1) / (ts[-1] - ts[0])


def _save_video(path: str, frames: List[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    logger.info(f"Saved: {path} ({len(frames)} frames, fps={fps:.2f})")


# ── 단계별 화면 ──────────────────────────────────────────────

def wait_for_start(cam: Camera) -> None:
    """대기 화면: 웹캠 원본 + 안내 문구. SPACE를 누르면 반환."""
    while True:
        _, frame = cam.latest()
        if frame is not None:
            disp = frame.copy()
            _label(disp, "Press SPACE to start")
            cv2.imshow(WINDOW, disp)
        if _check_key(30) == KEY_SPACE:
            return


def record_session(
    cam: Camera, orch: Orchestrator, disp_cfg: Dict[str, Any], seconds: float
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    """
    N초 녹화. 캡처 스레드는 원본 프레임을 전부 쌓고,
    여기서는 가장 최근 프레임만 가져와 실시간 분석 → 창 표시 + 실시간 분석 영상용으로 보관.
    """
    orch.reset()
    cam.start_recording()
    t0 = time.perf_counter()
    last_seq = -1
    live_frames: List[np.ndarray] = []
    live_ts: List[float] = []

    while (elapsed := time.perf_counter() - t0) < seconds:
        seq, frame = cam.latest()
        if frame is None or seq == last_seq:
            _check_key(1)
            continue
        last_seq = seq

        vis = frame.copy()      # 원본 프레임은 캡처 스레드가 보관 중이므로 복사본에 그린다
        out = orch.process(vis)
        _draw_overlays(vis, out.tracks, disp_cfg)
        live_frames.append(vis)
        live_ts.append(time.perf_counter())

        disp = vis.copy()
        _label(disp, f"REC  {max(0.0, seconds - elapsed):.0f}s", (0, 0, 255))
        cv2.imshow(WINDOW, disp)
        _check_key(1)

    raw_frames, raw_ts = cam.stop_recording()
    return raw_frames, raw_ts, live_frames, live_ts


def analyze_full(
    frames: List[np.ndarray], fps: float, orch: Orchestrator, disp_cfg: Dict[str, Any], out_path: str
) -> None:
    """녹화된 원본을 한 프레임도 빠짐없이 분석해서 결과 영상으로 저장. 창에는 진행률 표시."""
    orch.reset()
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = len(frames)
    try:
        for i, frame in enumerate(frames):
            vis = frame.copy()
            out = orch.process(vis)
            _draw_overlays(vis, out.tracks, disp_cfg)
            writer.write(vis)

            disp = vis.copy()
            _label(disp, f"Analyzing... {(i + 1) * 100 // n}%", (0, 255, 255))
            _progress_bar(disp, (i + 1) / n)
            cv2.imshow(WINDOW, disp)
            _check_key(1)
    finally:
        writer.release()
    logger.info(f"Saved: {out_path} ({n} frames, fps={fps:.2f})")


def play_result(path: str) -> None:
    """결과 영상을 원래 속도로 재생. SPACE로 건너뛰기."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    next_t = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _label(frame, "RESULT  (SPACE: skip)", (0, 255, 0))
            cv2.imshow(WINDOW, frame)
            next_t += 1.0 / fps
            wait_ms = max(1, int((next_t - time.perf_counter()) * 1000))
            if _check_key(wait_ms) == KEY_SPACE:
                break
    finally:
        cap.release()


# ── main ──────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config("configs/dev.yaml")
    ex_cfg = cfg.get("exhibit", {})
    disp_cfg = cfg.get("display", {})
    camera = int(ex_cfg.get("camera", 0))
    seconds = float(ex_cfg.get("record_seconds", 10))
    output_dir = ex_cfg.get("output_dir", "data/exhibit/")
    os.makedirs(output_dir, exist_ok=True)

    orch = Orchestrator(cfg)
    cam = Camera(camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    logger.info(f"Exhibit ready: camera={camera}, record={seconds}s, output_dir={output_dir}")

    try:
        while True:
            wait_for_start(cam)

            raw_frames, raw_ts, live_frames, live_ts = record_session(cam, orch, disp_cfg, seconds)
            if not raw_frames:
                logger.warning("No frames recorded. Back to waiting.")
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_fps = _measured_fps(raw_ts)
            _save_video(os.path.join(output_dir, f"{stamp}_raw.mp4"), raw_frames, raw_fps)
            if live_frames:
                _save_video(os.path.join(output_dir, f"{stamp}_live.mp4"), live_frames, _measured_fps(live_ts))

            result_path = os.path.join(output_dir, f"{stamp}_result.mp4")
            analyze_full(raw_frames, raw_fps, orch, disp_cfg, result_path)
            del raw_frames, live_frames     # 다음 세션 전에 메모리 해제

            play_result(result_path)
    except _Quit:
        logger.info("Exhibit closed by user.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
