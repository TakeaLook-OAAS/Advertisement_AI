# run_loop(cfg, src, orch) 구현
# 여기서만 while 루프 돌고, 매 프레임 orch.process(frame, meta) 호출

from __future__ import annotations

import os
import time
from typing import Any, Dict, Union

import cv2
from loguru import logger

from src.io.video_source import VideoSource
from src.logic.ad_cycle import AdCycleScheduler
from src.logic.status import StatusTracker
from src.vision.draw import draw_tracks, draw_crop_bbox, draw_fps, draw_headpose, draw_gaze, draw_roi, draw_look, draw_gender_age


def run_loop(cfg: Dict[str, Any], source: Union[int, str], orch) -> None:
    vs = VideoSource(source)
    status = StatusTracker()
    status.set_device_id(cfg.get("device_id", ""))
    status.set_roi_polygon(cfg.get("logic", {}).get("roi", {}).get("polygon", []))

    # ── display ──────────────────────────────────────────────────
    disp_cfg = cfg.get("display", {})
    font_scale = float(disp_cfg.get("font_scale", 0.7))             # 폰트 크기
    thickness = int(disp_cfg.get("thickness", 2))                   # 폰트 두께
    show_bbox = bool(disp_cfg.get("draw_bbox", True))               # bbox 표시
    show_crop_bbox = bool(disp_cfg.get("draw_crop_bbox", True))     # crop_bbox 표시
    show_fps = bool(disp_cfg.get("draw_fps", True))                 # FPS 표시
    show_headpose = bool(disp_cfg.get("draw_headpose", True))       # headpose + headpose vector표시
    show_gaze = bool(disp_cfg.get("draw_gaze", True))               # gaze + gaze vector 표시
    show_roi = bool(disp_cfg.get("draw_roi", True))                 # ROI 폴리곤 + in_roi 표시
    show_look = bool(disp_cfg.get("draw_look", True))               # LookResult 표시
    show_gender_age = bool(disp_cfg.get("draw_gender_age", True))   # gender, age_group 표시
    roi_pts = cfg.get("logic", {}).get("roi", {}).get("polygon", [])

    # ── 비디오 출력 설정(output) ──────────────────────────────────────────
    out_cfg = cfg.get("output", {})
    
    output_video = bool(disp_cfg.get("output_video", True))
    output_path = disp_cfg.get("output_video_path", "data/output/output.mp4")
    
    # ── 광고 사이클 설정 (항상 활성화) ──────────────────────────────
    ad_cycle_cfg = out_cfg.get("ad_cycle", {})
    json_dir = out_cfg.get("json_dir", "data/output/segments/")

    durations_s = ad_cycle_cfg["durations_s"]
    scheduler = AdCycleScheduler(durations_s)
    os.makedirs(json_dir, exist_ok=True)
    logger.info(f"Ad cycle: {len(durations_s)} segments, json_dir={json_dir}")

    writer = None
    if output_video:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)    # 출력 폴더 자동 생성
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, vs.fps, (vs.width, vs.height))
        logger.info(f"Video output enabled: {output_path}")

    last = time.time()  # FPS 계산용 타이머
    fps = 0.0           # 현재 FPS

    logger.info(f"VideoSource opened: fps={vs.fps:.2f} size=({vs.width}x{vs.height})")

    try:
        while True:
            ok, frame, meta = vs.read()        # 프레임 1장 읽기
            
            if not ok:
                logger.info("End of stream.")
                break

            out = orch.process(frame)

            # 상태 추적 업데이트
            status.update(meta, out.tracks)

            # 광고 경계 체크 → 세그먼트 JSON 내보내기
            while True:
                completed = scheduler.check(meta.ts_ms)
                if completed is None:
                    break
                segment_data = status.flush_segment(completed)
                seg_path = os.path.join(
                    json_dir,
                    f"segment_{completed.segment_index:03d}.json",
                )
                status.save_segment_json(seg_path, segment_data)
                logger.info(f"Ad segment exported: {seg_path}")

            ########################## 60프레임마다 로그 출력
            # 필요하면 디버그 로그   
            if meta.frame_idx % 60 == 0:
                logger.info(
                    f"\n"
                    f"frame={meta.frame_idx}\n"
                    f"ts_ms={meta.ts_ms}\n"
                    f"dets={out.dets}\n"
                    f"tracks={out.tracks}"
                )
            ########################## 나중에 지우셔   

            # FPS 계산
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 1.0 / dt

            # draw
            if show_bbox:           # bbox + ID
                draw_tracks(frame, out.tracks, font_scale, thickness)
            if show_crop_bbox:      # face bbox
                draw_crop_bbox(frame, out.tracks, thickness)
            if show_fps:            # FPS
                draw_fps(frame, fps, font_scale, thickness)
            if show_headpose:       # headpose + headpose vector
                draw_headpose(frame, out.tracks, font_scale, thickness)
            if show_gaze:           # gaze + gaze vector
                draw_gaze(frame, out.tracks, font_scale, thickness)
            if show_roi:            # ROI 폴리곤 + in_roi
                draw_roi(frame, out.tracks, roi_pts, font_scale, thickness)
            if show_look:           # LookResult
                draw_look(frame, out.tracks, font_scale, thickness)
            if show_gender_age:     # gender, age_group
                draw_gender_age(frame, out.tracks, font_scale, thickness)

            # 비디오 파일로 기록
            if writer is not None:
                writer.write(frame)

    finally:
        # 스트림 종료 시 마지막 상태 마감
        status.finalize()

        # 마지막 미완료 세그먼트도 내보내기
        final = scheduler.current_segment()
        segment_data = status.flush_segment(final)
        seg_path = os.path.join(
            json_dir,
            f"segment_{final.segment_index:03d}.json",
        )
        status.save_segment_json(seg_path, segment_data)
        logger.info(f"Final ad segment exported: {seg_path}")

        if writer is not None:
            writer.release()
            logger.info(f"Output video saved: {output_path}")

        vs.release()    # 동영상 파일 닫기