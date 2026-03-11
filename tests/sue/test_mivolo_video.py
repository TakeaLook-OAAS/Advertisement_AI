# tests/sue/test_mivolo_video.py
# 실행방법: PYTHONPATH=. python tests/sue/test_mivolo_video.py
# headpose, gaze, roi, look_result 안 넣음 attr 표시 확인용

from __future__ import annotations

import os
import cv2
import yaml

from src.models.yolo_detector import YoloDetector
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.mivolo_attr import MiVOLOAttr


def draw_track_info(frame, track):
    # person bbox
    pb = track.bbox
    cv2.rectangle(frame, (pb.x1, pb.y1), (pb.x2, pb.y2), (0, 255, 0), 2)

    # face bbox
    if track.crop_bbox is not None:
        fb = track.crop_bbox
        cv2.rectangle(frame, (fb.x1, fb.y1), (fb.x2, fb.y2), (255, 0, 0), 2)

    # attr text
    if track.attr is not None:
        text = f"ID {track.track_id} | {track.attr.gender.value} | {track.attr.age_group.value}"
    else:
        text = f"ID {track.track_id} | unknown | unknown"

    text_x = pb.x1
    text_y = max(25, pb.y1 - 10)

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def main():
    # -------------------------
    # paths
    # -------------------------
    config_path = "configs/dev.yaml"
    video_path = "data/samples/test2.mp4"
    output_path = "data/output/test_mivolo_video_result.mp4"

    # -------------------------
    # load config
    # -------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -------------------------
    # init models
    # -------------------------
    yolo = YoloDetector(cfg.get("models", {}).get("yolo", {}))
    tracker = ByteTrackTracker(cfg.get("models", {}).get("tracker", {}))
    face = FaceDetector(cfg.get("models", {}).get("face", {}))
    mivolo = MiVOLOAttr(cfg.get("models", {}).get("mivolo", {}))

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

        frame_idx += 1
        print(f"[TEST] processing frame {frame_idx}")

        # 1) detect
        dets = yolo.detect(frame)

        # 2) track
        tracks = tracker.update(dets)

        # 3) face
        tracks = face.detect_batch(frame, tracks)

        # 4) mivolo
        tracks = mivolo.infer(frame, tracks)

        # 5) draw
        vis = frame.copy()
        for track in tracks:
            draw_track_info(vis, track)

        # 6) write
        writer.write(vis)

    cap.release()
    writer.release()

    print(f"[TEST] result saved to: {output_path}")


if __name__ == "__main__":
    main()