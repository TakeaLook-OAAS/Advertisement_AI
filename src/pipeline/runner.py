# run_loop(cfg, src, orch) 구현
# 여기서만 while 루프 돌고, 매 프레임 orch.process(frame, meta) 호출

from __future__ import annotations

import os
import time
from typing import Any, Dict, Union

import cv2
from loguru import logger

from src.io.video_source import VideoSource
from src.io.api_sender import send_segment
from src.logic.ad_cycle import AdCycleScheduler
from src.logic.status import StatusTracker
from src.vision.draw import draw_tracks, draw_crop_bbox, draw_fps, draw_headpose, draw_gaze, draw_look, draw_gender_age


def run_loop(cfg: Dict[str, Any], source: Union[int, str], orch) -> None:
    vs = VideoSource(source)
    status = StatusTracker()
    status.set_min_hits(int(cfg.get("logic", {}).get("presence", {}).get("min_hits", 3)))

    # ── display ──────────────────────────────────────────────────
    disp_cfg = cfg.get("display", {})
    font_scale = float(disp_cfg.get("font_scale", 0.7))             # 폰트 크기
    thickness = int(disp_cfg.get("thickness", 2))                   # 폰트 두께
    show_bbox = bool(disp_cfg.get("draw_bbox", True))               # bbox 표시
    show_crop_bbox = bool(disp_cfg.get("draw_crop_bbox", True))     # crop_bbox 표시
    show_fps = bool(disp_cfg.get("draw_fps", True))                 # FPS 표시
    show_headpose = bool(disp_cfg.get("draw_headpose", True))       # headpose + headpose vector표시
    show_gaze = bool(disp_cfg.get("draw_gaze", True))               # gaze + gaze vector 표시
    show_look = bool(disp_cfg.get("draw_look", True))               # LookResult 표시
    show_gender_age = bool(disp_cfg.get("draw_gender_age", True))   # gender, age_group 표시
    show_window = bool(disp_cfg.get("show_window", False))          # 실시간 미리보기 창

    # ── 비디오 출력 설정(output) ──────────────────────────────────────────
    out_cfg = cfg.get("output", {})
    
    output_video = bool(disp_cfg.get("output_video", True))
    output_path = disp_cfg.get("output_video_path", "data/output/output.mp4")
    
    # ── 프레임 스킵 설정 ──────────────────────────────────────────
    frame_skip = int(cfg.get("pipeline", {}).get("frame_skip", 1))
    logger.info(f"Frame skip: every {frame_skip} frame(s)")
    # ── 백엔드 전송 설정 ──────────────────────────────────────────
    backend_url: str | None = cfg.get("backend", {}).get("url")

    # ── 광고 사이클 설정 (항상 활성화) ──────────────────────────────
    ad_cycle_cfg = out_cfg.get("ad_cycle", {})
    json_dir = out_cfg.get("json_dir", "data/output/segments/")
    
    device_id = cfg.get("device_id")
    status.set_device_id(device_id)
    durations_s = ad_cycle_cfg["durations_s"]
    scheduler = AdCycleScheduler(durations_s)
    os.makedirs(json_dir, exist_ok=True)
    logger.info(f"Ad cycle: {len(durations_s)} segments, json_dir={json_dir}")

    writer = None
    writer_pending = False       # 라이브 소스: 실제 처리 FPS 측정 후 writer 생성
    measure_ts: list[float] = []
    if output_video:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)    # 출력 폴더 자동 생성
        if vs.is_live:
            # 웹캠은 vs.fps(30)로 저장하면 처리 속도가 느려 빨리감기 영상이 됨.
            # 처음 몇 초로 실제 처리 FPS를 재고 그 값으로 writer를 만든다.
            writer_pending = True
            logger.info("Video output: measuring real FPS before recording...")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, vs.fps, (vs.width, vs.height))
            logger.info(f"Video output enabled: {output_path}")

    start_time = time.time()  # 전체 처리 시간 측정용
    last = time.time()  # FPS 계산용 타이머
    fps = 0.0           # 현재 FPS

    logger.info(f"VideoSource opened: fps={vs.fps:.2f} size=({vs.width}x{vs.height})")

    try:
        while True:
            ok, frame, meta = vs.read()        # 프레임 1장 읽기
            
            if not ok:
                logger.info("End of stream.")
                break

            if meta.frame_idx % frame_skip != 0:
                # 광고 경계 체크만 수행 (처리 스킵)
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
                continue

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
                if backend_url:
                    send_segment(segment_data, backend_url)

            if meta.frame_idx % 60 == 0:
            #    looking = sum(1 for t in out.tracks if t.look_result and t.look_result.is_looking)
            #    logger.info(
            #        f"frame={meta.frame_idx} | ts={meta.ts_ms}ms | "
            #        f"dets={len(out.dets)} | tracks={len(out.tracks)} | looking={looking}"
            #    )
            # ########################## 60프레임마다 로그 출력
                logger.info(
                    f"frame={meta.frame_idx} ts_ms={meta.ts_ms}"
                    #f"dets={out.dets}\n"
                    #f"tracks={out.tracks}"
                )
            # ########################## 나중에 지우셔

            # FPS 계산
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 1.0 / dt

            # 라이브 소스: 처음 ~3초로 실제 처리 FPS를 재고 그 값으로 writer 생성
            if writer_pending:
                measure_ts.append(now)
                span = measure_ts[-1] - measure_ts[0]
                if span >= 3.0 and len(measure_ts) >= 10:
                    real_fps = max(1.0, min((len(measure_ts) - 1) / span, 60.0))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, real_fps, (vs.width, vs.height))
                    writer_pending = False
                    logger.info(f"Video output enabled: {output_path} (measured fps={real_fps:.2f})")

            # draw (writer 또는 preview 중 하나라도 켜지면 수행)
            if writer is not None or show_window:
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
                if show_look:           # LookResult
                    draw_look(frame, out.tracks, font_scale, thickness)
                if show_gender_age:     # gender, age_group
                    draw_gender_age(frame, out.tracks, font_scale, thickness)

            if writer is not None:
                writer.write(frame)

            if show_window:
                cv2.imshow("Advertisement AI", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q 또는 ESC 로 종료
                    logger.info("Preview closed by user.")
                    break

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
        if backend_url:
            send_segment(segment_data, backend_url)

        if writer is not None:
            writer.release()
            logger.info(f"Output video saved: {output_path}")

        if show_window:
            cv2.destroyAllWindows()

        vs.release()    # 동영상 파일 닫기

        elapsed = time.time() - start_time
        logger.info(f"총 처리 시간: {elapsed:.1f}초")