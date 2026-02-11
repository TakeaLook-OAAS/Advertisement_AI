# run_loop(cfg, src, orch, show_window) 구현
# 여기서만 while 루프 돌고, 매 프레임 orch.process(frame, meta) 호출

from __future__ import annotations
import time
from typing import Any, Dict, Union
import cv2
from loguru import logger
from src.io.video_source import VideoSource
from src.utils.types import Track


def _draw_tracks(frame, tracks: list[Track], font_scale: float, thickness: int) -> None:
    for t in tracks:
        x1, y1, x2, y2 = t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            frame,
            f"id={t.track_id}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )


def run_loop(cfg: Dict[str, Any], source: Union[int, str], orch, show_window: bool = True) -> None:
    vs = VideoSource(source)

    io_cfg = cfg.get("io", {})
    disp_cfg = cfg.get("display", {})
    window_name = disp_cfg.get("window_name", "AI Demo")
    font_scale = float(disp_cfg.get("font_scale", 0.7))
    thickness = int(disp_cfg.get("thickness", 2))
    draw_fps = bool(disp_cfg.get("draw_fps", True))

    target_fps = float(io_cfg.get("target_fps", 0) or 0)
    warmup_frames = int(io_cfg.get("warmup_frames", 0) or 0)
    flip_horizontal = bool(io_cfg.get("flip_horizontal", False))

    last = time.time()
    fps_est = 0.0

    frame_count = 0
    logger.info(f"VideoSource opened: fps={vs.fps:.2f} size=({vs.width}x{vs.height})")

    try:
        while True:
            ok, frame, meta = vs.read()
            if not ok:
                logger.info("End of stream.")
                break

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            out = orch.process(frame, meta)
            
            ########################## 50프레임마다 로그 출력
            if meta.frame_idx % 50 == 0:
                logger.info(f"frame={meta.frame_idx} ts_ms={meta.ts_ms} dets={len(out.dets)} tracks={len(out.tracks)}")
            ########################## 나중에 지우셔
            
            # FPS estimate
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est > 0 else (1.0 / dt)

            # draw
            _draw_tracks(frame, out.tracks, font_scale, thickness)

            if draw_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps_est:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            if show_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    logger.info("Quit requested.")
                    break

            # optional throttle
            if target_fps > 0:
                sleep_s = max(0.0, (1.0 / target_fps) - (time.time() - now))
                if sleep_s > 0:
                    time.sleep(sleep_s)

            frame_count += 1
            if warmup_frames and frame_count == warmup_frames:
                logger.info("Warmup frames passed.")

    finally:
        vs.release()
        if show_window:
            cv2.destroyAllWindows()
