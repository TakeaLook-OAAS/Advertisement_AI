"""
실제 영상으로 YOLOv8 + ByteTrack + Draw 파이프라인 테스트.
기존 Haar cascade 테스트(test_video_pipeline.py)와 비교용.

출력: data/output/yolo_track_test.mp4
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from loguru import logger

from src.tracking.byte_tracker import BYTETracker
from src.models.yolo_detector import YOLODetector
from src.vision.draw import draw_tracks, draw_occlusion_status

# ── 설정 ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/samples/test.mp4"
OUTPUT_PATH = "data/output/yolo_track_test.mp4"

cfg = {
    "display": {
        "draw_tracks":      True,
        "draw_track_id":    True,
        "draw_track_score": True,
        "draw_lost_tracks": True,
        "font_scale":       0.65,
        "track_thickness":  2,
    }
}

tracker = BYTETracker({
    "track_thresh":    0.4,
    "low_thresh":      0.1,
    "match_thresh":    0.65,
    "max_lost_frames": 30,
    "min_hits":        2,
})

# ── YOLODetector 초기화 ──────────────────────────────────────────────────────
yolo = YOLODetector({
    "enabled":     True,
    "model_path":  "yolov8n.pt",
    "device":      "cpu",
    "conf_thresh": 0.25,
    "iou_thresh":  0.45,
    "classes":     [0],    # 0 = 사람
    "imgsz":       640,
})

# ── 영상 처리 ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_PATH)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

logger.info(f"입력: {INPUT_PATH}  ({W}x{H}, {FPS:.0f}fps, {TOTAL}프레임)")
logger.info(f"출력: {OUTPUT_PATH}")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # ── YOLO 검출 ────────────────────────────────────────────────────────
    dets = yolo.infer(frame)

    # ── ByteTrack ────────────────────────────────────────────────────────
    tracks = tracker.update(dets)

    # ── 가상 attention (bbox 넓이 기준 — 실제는 HeadPose 결과 사용) ─────
    attending_map = {}
    if tracks:
        areas = [(t, (t.tlbr[2]-t.tlbr[0]) * (t.tlbr[3]-t.tlbr[1])) for t in tracks]
        max_area = max(a for _, a in areas)
        for t, a in areas:
            attending_map[t.track_id] = (a >= max_area * 0.8)

    # ── 시각화 ──────────────────────────────────────────────────────────
    vis = frame.copy()
    vis = draw_tracks(vis, tracks, cfg=cfg, attending_per_track=attending_map)
    vis = draw_occlusion_status(vis, tracker.lost_stracks, cfg=cfg)

    # 프레임 정보 오버레이
    cv2.putText(vis, f"[YOLO] Frame {frame_idx}/{TOTAL}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Dets:{len(dets)}  Tracked:{len(tracks)}  Lost:{len(tracker.lost_stracks)}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    out.write(vis)

    if frame_idx % 50 == 0:
        logger.info(f"  {frame_idx}/{TOTAL}  tracks={[t.track_id for t in tracks]}")

cap.release()
out.release()
logger.success(f"완료 → {OUTPUT_PATH}")
