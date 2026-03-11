# tests/sue/test_mivolo_image.py
# 실행방법: PYTHONPATH=. python tests/sue/test_mivolo_image.py

from __future__ import annotations

import os
import cv2
import yaml

from src.models.yolo_detector import YoloDetector
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.mivolo_attr import MiVOLOAttr


def draw_track_info(frame, track, attr=None):
    # person bbox
    pb = track.bbox
    cv2.rectangle(frame, (pb.x1, pb.y1), (pb.x2, pb.y2), (0, 255, 0), 2)

    # face bbox
    if track.crop_bbox is not None:
        fb = track.crop_bbox
        cv2.rectangle(frame, (fb.x1, fb.y1), (fb.x2, fb.y2), (255, 0, 0), 2)

    # text
    if attr is not None:
        text = f"ID {track.track_id} | {attr.gender.value} | {attr.age_group.value}"
    else:
        text = f"ID {track.track_id} | unknown | unknown"

    text_x = pb.x1
    text_y = max(20, pb.y1 - 10)

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,            # 글자 크기
        (0, 255, 255),
        3,              # 글자 두께
        cv2.LINE_AA,
    )

    return frame


def main():
    # -------------------------
    # paths
    # -------------------------
    config_path = "configs/dev.yaml"
    image_path = "data/samples/test_images/test1.jpg"
    output_path = "data/output/test_mivolo_image_result.jpg"

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
    # read image
    # -------------------------
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # -------------------------
    # pipeline
    # -------------------------
    dets = yolo.detect(frame)
    print(f"[TEST] dets={len(dets)}")

    tracks = tracker.update(dets)
    print(f"[TEST] tracks={len(tracks)}")

    tracks = face.detect_batch(frame, tracks)
    attrs = mivolo.infer(frame, tracks)

    print("[TEST] attrs:")
    for track_id, attr in attrs.items():
        print(
            f"  track_id={track_id}, "
            f"gender={attr.gender.value}, age_group={attr.age_group.value}"
        )

    # -------------------------
    # draw
    # -------------------------
    vis = frame.copy()
    for track in tracks:
        attr = attrs.get(track.track_id)
        draw_track_info(vis, track, attr)

    # -------------------------
    # save
    # -------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)

    print(f"[TEST] result saved to: {output_path}")


if __name__ == "__main__":
    main()