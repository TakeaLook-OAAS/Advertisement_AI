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

    disp_cfg = cfg.get("display", {})

    window_name = disp_cfg.get("window_name", "AI Demo")
    font_scale = float(disp_cfg.get("font_scale", 0.7))             # 폰트 크기
    thickness = int(disp_cfg.get("thickness", 2))                   # 폰트 두께
    draw_fps = bool(disp_cfg.get("draw_fps", True))                 # FPS 표시

    last = time.time()  # FPS 계산용 타이머
    fps = 0.0            # 현재 FPS

    logger.info(f"VideoSource opened: fps={vs.fps:.2f} size=({vs.width}x{vs.height})")

    try:
        while True:
            ok, frame, meta = vs.read()        # 프레임 1장 읽기
            if not ok:
                logger.info("End of stream.")
                break

            out = orch.process(frame, meta)
            
            ########################## 50프레임마다 로그 출력
            if meta.frame_idx % 50 == 0:
                logger.info(f"frame={meta.frame_idx} ts_ms={meta.ts_ms} dets={len(out.dets)} tracks={len(out.tracks)}")
            ########################## 나중에 지우셔   

            # FPS 계산
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 1.0 / dt

            # draw: 바운딩 박스 + ID 그리기
            _draw_tracks(frame, out.tracks, font_scale, thickness)

            if draw_fps:    # FPS 숫자를 화면 왼쪽 위에 표시
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            if show_window:     # --no-window가 아니면
                cv2.imshow(window_name, frame)      # 창에 프레임 표시
                key = cv2.waitKey(1) & 0xFF         # 키 입력 대기 (1ms)
                if key == 27 or key == ord("q"):    # ESC 또는 Q 누르면
                    logger.info("Quit requested.") 
                    break                           # 루프 탈출

    finally:
        vs.release()    # 동영상 파일 닫기
        if show_window:
            cv2.destroyAllWindows() # 창 닫기