# tests/sue/test_orchestrator_video.py
# 실행방법: PYTHONPATH=. python tests/sue/test_orchestrator_video.py

from __future__ import annotations

import os
import cv2
import yaml

from src.pipeline.orchestrator import Orchestrator
from src.utils.types import FrameMeta


def _draw_text_lines(
    frame,
    lines,
    x,
    y,
    font_scale=0.8,
    thickness=2,
    line_gap=28,
    color=(0, 255, 255),
):
    """
    여러 줄 텍스트를 위에서 아래로 출력
    """
    for i, line in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            frame,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_track_info(frame, track):
    """
    track 안의 각종 결과(attr, headpose, gaze, roi, look_result)를 시각화
    """
    # -------------------------
    # bbox
    # -------------------------
    pb = track.bbox
    cv2.rectangle(frame, (pb.x1, pb.y1), (pb.x2, pb.y2), (0, 255, 0), 2)  # person

    if track.crop_bbox is not None:
        fb = track.crop_bbox
        cv2.rectangle(frame, (fb.x1, fb.y1), (fb.x2, fb.y2), (255, 0, 0), 2)  # face

    if track.left_eye is not None:
        eb = track.left_eye
        cv2.rectangle(frame, (eb.x1, eb.y1), (eb.x2, eb.y2), (0, 0, 255), 2)  # left eye

    if track.right_eye is not None:
        eb = track.right_eye
        cv2.rectangle(frame, (eb.x1, eb.y1), (eb.x2, eb.y2), (0, 165, 255), 2)  # right eye

    # -------------------------
    # text lines
    # -------------------------
    lines = []

    # line 1: id + attr
    if track.attr is not None:
        lines.append(
            f"ID {track.track_id} | {track.attr.gender.value} | {track.attr.age_group.value}"
        )
    else:
        lines.append(f"ID {track.track_id} | unknown | unknown")

    # line 2: headpose
    if track.headpose is not None:
        hp = track.headpose
        lines.append(f"HP y:{hp.yaw:.1f} p:{hp.pitch:.1f} r:{hp.roll:.1f}")

    # line 3: gaze
    if track.gaze is not None:
        gz = track.gaze
        lines.append(f"Gaze x:{gz.x:.2f} y:{gz.y:.2f} z:{gz.z:.2f}")

    # line 4: ROI
    if getattr(track, "roi", None) is not None:
        roi = track.roi
        lines.append(f"ROI in:{roi.in_roi} dwell:{roi.dwell_frames}")

    # line 5: look_result
    if track.look_result is not None:
        lr = track.look_result
        lines.append(
            f"Look {lr.is_looking} score:{lr.score:.2f} angle:{lr.angle_deg:.1f}"
        )

    # -------------------------
    # text position
    # -------------------------
    text_x = pb.x1
    text_y = max(30, pb.y1 - 10 - (len(lines) - 1) * 30)

    _draw_text_lines(
        frame,
        lines,
        text_x,
        text_y,
        font_scale=0.8,
        thickness=2,
        line_gap=28,
        color=(0, 255, 255),
    )

    return frame


def main():
    # -------------------------
    # paths
    # -------------------------
    config_path = "configs/dev.yaml"
    video_path = "data/samples/test2.mp4"
    output_path = "data/output/test_orchestrator_video_result.mp4"

    # -------------------------
    # load config
    # -------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -------------------------
    # init orchestrator
    # -------------------------
    orchestrator = Orchestrator(cfg)

    # -------------------------
    # open video
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    # -------------------------
    # frame loop
    # -------------------------
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        meta = FrameMeta(
            frame_idx=frame_idx,
            ts_ms=ts_ms,
            fps=fps,
            width=width,
            height=height,
        )

        output = orchestrator.process(frame, meta)

        vis = frame.copy()

        for track in output.tracks:
            draw_track_info(vis, track)

        # 좌상단에 프레임 정보
        cv2.putText(
            vis,
            f"frame={frame_idx}  ts={ts_ms}ms  tracks={len(output.tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

        frame_idx += 1
        print(f"[TEST] processed frame {frame_idx}")

    cap.release()
    writer.release()

    print(f"[TEST] result saved to: {output_path}")


if __name__ == "__main__":
    main()