"""
Visual smoke-test for ByteTrackTracker — 실제 영상 버전.

tracking/ 폴더를 참조하지 않고 bytetrack_tracker.py 단독으로 동작을 검증.

YOLO로 사람을 실제 검출 → ByteTrackTracker로 추적 → 결과 영상 저장.

입력: data/samples/test.mp4
출력: data/output/tracking_test.mp4

사용법:
    python tests/kwon/test_tracking_visual.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np

from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.yolo_detector import YoloDetector
from src.utils.types import Track
from src.logic.roi import PolygonROI


# ── 설정 ────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "data/samples/test.mp4"
OUTPUT_VIDEO = "data/output/tracking_test.mp4"

YOLO_CFG = {
    "enabled":     True,
    "model_path":  "models/yolo/yolov8n.pt",
    "device":      "cpu",
    "conf_thresh": 0.25,
    "iou_thresh":  0.45,
    "classes":     [0],   # 0 = person
    "imgsz":       640,
}

TRACKER_CFG = {
    "track_thresh":    0.5,
    "low_thresh":      0.1,
    "match_thresh":    0.65,
    "max_lost_frames": 30,
    "min_hits":        1,
}

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
]

def _id_color(track_id: int) -> tuple:
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ── 시각화 ───────────────────────────────────────────────────────────────────
def draw_tracks(vis: np.ndarray, tracks: list, font_scale: float = 0.6) -> np.ndarray:
    """Track(types.py) 기반으로 bbox + track_id + conf를 그린다."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2
        color = _id_color(t.track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id}  conf={t.conf:.2f}  age={t.age}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_info(vis: np.ndarray, frame_idx: int, n_tracks: int, fps: float) -> np.ndarray:
    lines = [
        f"Frame: {frame_idx}",
        f"Tracks: {n_tracks}",
        f"FPS: {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    # 입력 영상 열기
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {INPUT_VIDEO}")
        sys.exit(1)

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 입력: {INPUT_VIDEO}  ({src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames}frames)")

    # 출력 영상 설정
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, src_fps, (src_w, src_h))

    # 모델 초기화
    print("[INFO] YOLO 모델 로딩...")
    detector = YoloDetector(YOLO_CFG)
    tracker  = ByteTrackTracker(TRACKER_CFG)
    print("[INFO] 추적 시작")

    import time
    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 검출 → 추적
        dets   = detector.detect(frame)
        tracks = tracker.update(dets)

        # FPS 계산
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-9)
        t_prev = now

        # 시각화
        vis = frame.copy()
        vis = draw_tracks(vis, tracks)
        vis = draw_info(vis, frame_idx, len(tracks), fps)

        writer.write(vis)

        # 100 프레임마다 콘솔 출력
        if frame_idx % 100 == 0:
            ids = [t.track_id for t in tracks]
            print(f"  frame={frame_idx:>4}  tracks={ids}  fps={fps:.1f}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n[DONE] {frame_idx} frames 처리 완료")
    print(f"[DONE] 출력 영상 저장: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
