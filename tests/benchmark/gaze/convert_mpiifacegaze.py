"""
MPIIFaceGaze (p00) → labels.json 변환 스크립트

어노테이션 파일(p00.txt) 형식 (총 28 컬럼, 데이터로 검증 완료):
  cols[0]      : 이미지 경로
  cols[1:3]    : 화면 gaze 타겟 (픽셀)
  cols[3:15]   : 2D landmark 6점  (0,1=좌눈 / 2,3=우눈 / 4,5=입)
  cols[15:18]  : head rotation vector (rodrigues) — 미사용
  cols[18:21]  : face center 3D (카메라 좌표계 mm, gaze 원점)
  cols[21:24]  : 보조 3D point — 미사용
  cols[24:27]  : 3D gaze target (카메라 좌표계 mm)
  cols[27]     : which eye

gaze GT:
  normalize(gaze_target_3D - face_center_3D)
  = normalize(cols[24:27] - cols[18:21])
  → 0002 출력과 같은 카메라 좌표계 3D 단위벡터

eye bbox:
  2D landmark 눈꼬리 점에서 직접 추출 (0,1=좌눈 / 2,3=우눈).
  3D 투영이 아니라 이미지 평면 좌표를 그대로 쓴다.

headpose:
  MPIIFaceGaze 어노테이션에 yaw/pitch/roll 오일러각이 없으므로
  6DRepNet(HeadPoseEstimator)으로 각 이미지에서 직접 추정한다.

사용법:
    python -m tests.benchmark.gaze.convert_mpiifacegaze
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.models.headpose_6drepnet import HeadPoseEstimator
from src.utils.types import BBoxXYXY, Track

# ── 설정 ──────────────────────────────────────────────────────
MPIIFACEGAZE_DIR = "data/benchmark/gaze/MPIIFaceGaze"
SUBJECT = "p00"
OUTPUT_PATH = "data/benchmark/gaze/labels.json"

HEADPOSE_WEIGHTS = "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"

# eye bbox: 눈꼬리 두 점 기준 패딩(px), 최소 half 크기
EYE_PAD = 12
EYE_MIN_HALF = 18

# p00 전체 ~3700장 중 샘플링 간격 (1이면 전체)
SAMPLE_EVERY = 7  # ~500장


def make_eye_bbox(
    p_a: tuple[float, float],
    p_b: tuple[float, float],
    img_w: int,
    img_h: int,
) -> dict:
    """눈꼬리 두 점(p_a, p_b)으로 정사각형에 가까운 bbox 생성."""
    cx = (p_a[0] + p_b[0]) / 2.0
    cy = (p_a[1] + p_b[1]) / 2.0
    half = max(abs(p_a[0] - p_b[0]) / 2.0 + EYE_PAD, EYE_MIN_HALF)
    return {
        "x1": int(max(0, cx - half)),
        "y1": int(max(0, cy - half)),
        "x2": int(min(img_w, cx + half)),
        "y2": int(min(img_h, cy + half)),
    }


def estimate_headpose(
    estimator: HeadPoseEstimator,
    frame: np.ndarray,
    face_pts_2d: list[tuple[int, int]],
) -> Optional[dict]:
    """6DRepNet으로 headpose 추정. face_pts_2d에서 face bbox를 만든다."""
    xs = [p[0] for p in face_pts_2d]
    ys = [p[1] for p in face_pts_2d]
    h, w = frame.shape[:2]
    pad_x = int((max(xs) - min(xs)) * 0.3)
    pad_y = int((max(ys) - min(ys)) * 0.3)
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)

    if x2 - x1 < 30 or y2 - y1 < 30:
        return None

    face_bbox = BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)
    track = Track(track_id=0, bbox=face_bbox, crop_bbox=face_bbox)
    track = estimator.infer(frame, track)

    hp = track.headpose
    if hp is None or (hp.yaw == 0.0 and hp.pitch == 0.0 and hp.roll == 0.0):
        return None

    return {"yaw": hp.yaw, "pitch": hp.pitch, "roll": hp.roll}


def main() -> None:
    subject_dir = os.path.join(MPIIFACEGAZE_DIR, SUBJECT)
    annotation_path = os.path.join(subject_dir, f"{SUBJECT}.txt")

    estimator = HeadPoseEstimator({"weights": HEADPOSE_WEIGHTS, "device": "cpu"})

    with open(annotation_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    labels = []
    skipped = 0

    for i, line in enumerate(lines):
        if i % SAMPLE_EVERY != 0:
            continue

        cols = line.split()

        img_rel = cols[0]                          # e.g. day01/0005.jpg
        img_path = os.path.join(subject_dir, img_rel)

        frame = cv2.imread(img_path)
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]

        # ── 2D landmark 6점 (0,1=좌눈 / 2,3=우눈 / 4,5=입) ──────
        lm = [(float(cols[3 + j * 2]), float(cols[4 + j * 2])) for j in range(6)]

        # ── gaze GT: face center 3D → gaze target 3D (카메라 좌표계) ──
        # cols[18:21] = face center 3D (gaze 원점), cols[24:27] = gaze target 3D.
        # 둘의 차이를 정규화하면 0002 출력과 같은 카메라 좌표계 단위벡터가 된다.
        face_center_3d = np.array(
            [float(cols[18]), float(cols[19]), float(cols[20])]
        )
        target_3d = np.array(
            [float(cols[24]), float(cols[25]), float(cols[26])]
        )
        gaze_raw = target_3d - face_center_3d
        gaze_raw[1] = -gaze_raw[1]
        gaze_norm = np.linalg.norm(gaze_raw)
        if gaze_norm < 1e-8:
            skipped += 1
            continue
        gaze_unit = gaze_raw / gaze_norm

        # ── eye bbox: 2D landmark 눈꼬리에서 직접 추출 ───────────
        left_eye = make_eye_bbox(lm[0], lm[1], w, h)
        right_eye = make_eye_bbox(lm[2], lm[3], w, h)

        # ── headpose: 6DRepNet 추정 ─────────────────────────────
        face_pts = [(int(x), int(y)) for (x, y) in lm]
        headpose = estimate_headpose(estimator, frame, face_pts)
        if headpose is None:
            skipped += 1
            continue

        labels.append({
            "image": f"{SUBJECT}/{img_rel}",
            "left_eye": left_eye,
            "right_eye": right_eye,
            "headpose": headpose,
            "gaze": {
                "x": float(gaze_unit[0]),
                "y": float(gaze_unit[1]),
                "z": float(gaze_unit[2]),
            },
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"  총 샘플: {len(labels)}")
    print(f"  제외: {skipped}")


if __name__ == "__main__":
    main()