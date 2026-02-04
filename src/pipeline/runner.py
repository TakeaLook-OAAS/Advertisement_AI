import time
import cv2
from loguru import logger

from src.io.video_source import VideoSource
from src.utils.fps import FPSMeter
from src.vision.draw import draw_overlay

def run_loop(cfg, source, orchestrator, *, show_window: bool = True):
    io_cfg = cfg.get("io", {})
    disp_cfg = cfg.get("display", {})
    window_name = disp_cfg.get("window_name", "AI Demo")

    vs = VideoSource(
        source=source,
        width=io_cfg.get("width", 1280),
        height=io_cfg.get("height", 720),
        flip_horizontal=io_cfg.get("flip_horizontal", False),
    )
    fps = FPSMeter()

    warmup = int(io_cfg.get("warmup_frames", 5))
    target_fps = float(io_cfg.get("target_fps", 0))  # 0=no throttle
    min_dt = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

    frame_idx = 0
    last_t = time.perf_counter()

    logger.info("Starting frame loop. Press 'q' to quit (window mode).")

    while True:
        ok, frame = vs.read()
        if not ok:
            logger.warning("No more frames / failed to read.")
            break

        frame_idx += 1
        if frame_idx <= warmup:
            fps.reset()

        t0 = time.perf_counter()
        result = orchestrator.process(frame)
        fps_val = fps.update()

        if show_window:
            vis = draw_overlay(frame, result, fps=fps_val, cfg=cfg)
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # optional throttle
        if min_dt > 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < min_dt:
                time.sleep(min_dt - dt)

        # occasional log
        now = time.perf_counter()
        if now - last_t > 5.0:
            last_t = now
            logger.info(f"FPS(EMA): {fps_val:.2f}")

    vs.release()
    if show_window:
        cv2.destroyAllWindows()
