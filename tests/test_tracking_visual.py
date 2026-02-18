"""
Visual smoke-test for the tracking + draw pipeline.

Runs WITHOUT real models (YOLO / HeadPose / Gaze).
Uses synthetic bounding boxes to simulate:
  - 3 people walking across the frame
  - One person temporarily occluded (frames 15-25)
  - Attention states fed manually

Output: /app/tests/output_tracking_test.png
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from src.tracking.byte_tracker import BYTETracker
from src.tracking.strack import TrackState
from src.logic.roi import PolygonROI
from src.vision.draw import draw_tracks, draw_occlusion_status, draw_gaze_arrow
from src.pipeline.orchestrator import InferenceResult


# ── 합성 시나리오 설정 ──────────────────────────────────────────────────────
W, H      = 1280, 720
N_FRAMES  = 40
FPS_SIM   = 15

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
    "track_thresh":    0.5,
    "low_thresh":      0.1,
    "match_thresh":    0.65,
    "max_lost_frames": 30,
    "min_hits":        1,
})

roi = PolygonROI([[200, 150], [1080, 150], [1080, 600], [200, 600]])

# ── 사람 3명의 이동 경로 정의 ───────────────────────────────────────────────
def make_box(cx, cy, w=100, h=200):
    return [cx - w//2, cy - h//2, cx + w//2, cy + h//2]

def person_boxes(frame_idx):
    """프레임 번호에 따라 3명의 bbox 생성. person2는 15~25 프레임 occluded."""
    dets = []

    # person1: 왼→오 이동, 항상 보임
    cx1 = 250 + frame_idx * 18
    dets.append(make_box(cx1, 350) + [0.92])

    # person2: 오→왼, 15~25 프레임 occluded (낮은 신뢰도)
    cx2 = 1050 - frame_idx * 15
    score2 = 0.08 if 15 <= frame_idx <= 25 else 0.85
    dets.append(make_box(cx2, 400, 110, 220) + [score2])

    # person3: 중앙, 처음부터 끝까지 거의 제자리 (광고 주시 중)
    cx3 = 620 + int(10 * np.sin(frame_idx * 0.3))
    dets.append(make_box(cx3, 300, 90, 180) + [0.78])

    return np.array(dets, dtype=float)


# ── ROI 테스트 ───────────────────────────────────────────────────────────────
def test_roi():
    inside  = roi.contains_point(640, 400)
    outside = roi.contains_point(100, 100)
    assert inside,  "ROI: 중심점이 내부로 판별되어야 함"
    assert not outside, "ROI: 외부 점이 외부로 판별되어야 함"
    print("[PASS] ROI contains_point")

    box_in  = roi.contains_box([500, 200, 700, 500])
    box_out = roi.contains_box([10, 10, 80, 80])
    assert box_in,      "ROI: 내부 박스 통과해야 함"
    assert not box_out, "ROI: 외부 박스 필터링되어야 함"
    print("[PASS] ROI contains_box")


# ── Kalman 연속 추적 테스트 ──────────────────────────────────────────────────
def test_kalman_continuity():
    from src.tracking.kalman_filter import KalmanFilter
    kf = KalmanFilter()
    meas = np.array([300.0, 200.0, 100.0, 200.0])
    mean, cov = kf.initiate(meas)

    for _ in range(10):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, meas + np.random.randn(4) * 2)

    assert not np.any(np.isnan(mean)), "Kalman: mean에 NaN 없어야 함"
    assert not np.any(np.isnan(cov)),  "Kalman: cov에 NaN 없어야 함"
    # 정규화 후 수치 안정성 확인
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > -1e-6), f"Kalman: cov 음의 고유값 발생: {eigvals.min():.2e}"
    print("[PASS] KalmanFilter 수치 안정성")


# ── ByteTracker 전체 시뮬레이션 ──────────────────────────────────────────────
def run_simulation():
    """N_FRAMES 동안 추적 실행, 마지막 프레임 시각화 반환"""
    last_frame_vis = None

    for frame_idx in range(N_FRAMES):
        dets = person_boxes(frame_idx)

        # ROI 필터링
        roi_dets = np.array(
            [d for d in dets if roi.contains_box(d[:4])],
            dtype=float
        )
        if len(roi_dets) == 0:
            roi_dets = np.empty((0, 5), dtype=float)

        tracks = tracker.update(roi_dets)

        # 마지막 프레임 시각화
        if frame_idx == N_FRAMES - 1:
            frame = np.full((H, W, 3), 40, dtype=np.uint8)

            # ROI 폴리곤
            pts = np.array(roi.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (255, 120, 0), 2)
            cv2.putText(frame, "ROI", (205, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 1)

            # 가상 attention 상태 (track 3이 가장 크면 주시)
            attending_map = {}
            for t in tracks:
                w = t.tlbr[2] - t.tlbr[0]
                attending_map[t.track_id] = bool(w > 100)

            # 트랙 그리기
            frame = draw_tracks(
                frame, tracks, cfg=cfg,
                attending_per_track=attending_map
            )

            # Lost(occluded) 트랙 코너 마크
            frame = draw_occlusion_status(
                frame, tracker.lost_stracks, cfg=cfg
            )

            # Gaze 화살표 (가장 큰 트랙 기준)
            if tracks:
                primary = max(
                    tracks,
                    key=lambda t: (t.tlbr[2]-t.tlbr[0]) * (t.tlbr[3]-t.tlbr[1])
                )
                fake_gaze = np.array([0.3, 0.1, 0.95])  # 약간 오른쪽 아래를 봄
                frame = draw_gaze_arrow(frame, fake_gaze, primary.tlbr, scale=100)

            # 정보 텍스트
            y = 30
            lines = [
                f"Frame: {frame_idx+1}/{N_FRAMES}",
                f"Active tracks : {len(tracks)}",
                f"Lost tracks   : {len(tracker.lost_stracks)}",
                f"Scipy (Hungarian): {'YES' if _scipy_available() else 'NO (greedy fallback)'}",
            ]
            for line in lines:
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                y += 26

            # 범례
            lx, ly = W - 230, H - 100
            cv2.putText(frame, "Legend:", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.rectangle(frame, (lx, ly+8), (lx+18, ly+22), (0,220,80), 3)
            cv2.putText(frame, "Attending",  (lx+24, ly+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.rectangle(frame, (lx, ly+30), (lx+18, ly+44), (0,60,220), 2)
            cv2.putText(frame, "Not attending", (lx+24, ly+44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, "L = Lost(occluded)", (lx, ly+66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            last_frame_vis = frame

        # 중간 통계 출력
        if frame_idx in (0, 14, 15, 25, 26, N_FRAMES - 1):
            lost_ids = [t.track_id for t in tracker.lost_stracks]
            active_ids = [t.track_id for t in tracks]
            print(
                f"  Frame {frame_idx+1:>2} | "
                f"active={active_ids} | "
                f"lost={lost_ids}"
            )

    return last_frame_vis


def _scipy_available():
    try:
        import scipy
        return True
    except ImportError:
        return False


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== ROI 테스트 ===")
    test_roi()

    print("\n=== Kalman Filter 수치 안정성 테스트 ===")
    test_kalman_continuity()

    print("\n=== ByteTracker 시뮬레이션 (40 프레임) ===")
    vis = run_simulation()

    out_path = os.path.join(os.path.dirname(__file__), "output_tracking_test.png")
    cv2.imwrite(out_path, vis)
    print(f"\n시각화 이미지 저장: {out_path}")

    print(f"\n최종 상태:")
    print(f"  Active tracks : {len([t for t in tracker.tracked_stracks])}")
    print(f"  Lost tracks   : {len(tracker.lost_stracks)}")
    print(f"  Removed tracks: {len(tracker.removed_stracks)}")
    print("\n[ALL PASS]")
