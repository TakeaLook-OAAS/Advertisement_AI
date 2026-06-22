"""
MPIIFaceGaze 학습용 라벨 생성 (p01-p14).

convert_mpiifacegaze.py 와 동일한 방식으로:
  - eye bbox  : 2D landmark 눈꼬리 두 점에서 추출
  - headpose  : 6DRepNet 추정 (추론 파이프라인과 동일 형식)
  - gaze GT   : normalize(gaze_target_3D - face_center_3D), y축 반전

p00 은 test set 이므로 제외한다.

사용법 (프로젝트 루트에서):
    python -m scripts.gaze.gaze_generate_labels
    python -m scripts.gaze.gaze_generate_labels --every 1   # 전체 데이터
    python -m scripts.gaze.gaze_generate_labels --subjects p01 p02 p03

출력: data/benchmark/gaze/labels_train.json
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from src.models.headpose_6drepnet import HeadPoseEstimator
from src.utils.types import BBoxXYXY, Track

# ── 기본 설정 ──────────────────────────────────────────────────
MPIIFACEGAZE_DIR = "data/benchmark/gaze/MPIIFaceGaze"
OUTPUT_PATH = "data/benchmark/gaze/labels_train.json"
HEADPOSE_WEIGHTS = "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"

# p00 은 test set → 제외
ALL_TRAIN_SUBJECTS = [f"p{i:02d}" for i in range(1, 15)]

EYE_PAD = 12
EYE_MIN_HALF = 18


# ── 유틸 함수 (convert_mpiifacegaze.py 와 동일 로직) ─────────

def _make_eye_bbox(p_a: tuple, p_b: tuple, img_w: int, img_h: int) -> dict:
    cx = (p_a[0] + p_b[0]) / 2.0
    cy = (p_a[1] + p_b[1]) / 2.0
    half = max(abs(p_a[0] - p_b[0]) / 2.0 + EYE_PAD, EYE_MIN_HALF)
    return {
        "x1": int(max(0, cx - half)),
        "y1": int(max(0, cy - half)),
        "x2": int(min(img_w, cx + half)),
        "y2": int(min(img_h, cy + half)),
    }


def _estimate_headpose(estimator: HeadPoseEstimator, frame: np.ndarray, face_pts: list) -> dict | None:
    xs = [p[0] for p in face_pts]
    ys = [p[1] for p in face_pts]
    h, w = frame.shape[:2]
    pad_x = int((max(xs) - min(xs)) * 0.3)
    pad_y = int((max(ys) - min(ys)) * 0.3)
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)
    if x2 - x1 < 30 or y2 - y1 < 30:
        return None

    track = Track(
        track_id=0,
        bbox=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
        crop_bbox=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
    )
    track = estimator.infer(frame, track)
    hp = track.headpose
    if hp is None or (hp.yaw == 0.0 and hp.pitch == 0.0 and hp.roll == 0.0):
        return None
    return {"yaw": hp.yaw, "pitch": hp.pitch, "roll": hp.roll}


def _process_subject(subject: str, estimator: HeadPoseEstimator, every: int) -> tuple[list, int]:
    subject_dir = os.path.join(MPIIFACEGAZE_DIR, subject)
    ann_path = os.path.join(subject_dir, f"{subject}.txt")

    with open(ann_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    results, skipped = [], 0

    for i, line in enumerate(lines):
        if i % every != 0:
            continue

        cols = line.split()
        if len(cols) < 27:
            skipped += 1
            continue

        img_path = os.path.join(subject_dir, cols[0])
        frame = cv2.imread(img_path)
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]

        # 2D landmark 6점 (0,1=좌눈 / 2,3=우눈 / 4,5=입)
        lm = [(float(cols[3 + j * 2]), float(cols[4 + j * 2])) for j in range(6)]

        # gaze GT: face_center=cols[18:21], target=cols[24:27], y축 반전 후 정규화
        face_center = np.array([float(cols[18]), float(cols[19]), float(cols[20])])
        target = np.array([float(cols[24]), float(cols[25]), float(cols[26])])
        gaze_raw = target - face_center
        gaze_raw[1] = -gaze_raw[1]
        norm = np.linalg.norm(gaze_raw)
        if norm < 1e-8:
            skipped += 1
            continue
        gaze_unit = gaze_raw / norm

        left_eye = _make_eye_bbox(lm[0], lm[1], w, h)
        right_eye = _make_eye_bbox(lm[2], lm[3], w, h)

        face_pts = [(int(x), int(y)) for (x, y) in lm]
        headpose = _estimate_headpose(estimator, frame, face_pts)
        if headpose is None:
            skipped += 1
            continue

        results.append({
            "image": f"{subject}/{cols[0]}",
            "left_eye": left_eye,
            "right_eye": right_eye,
            "headpose": headpose,
            "gaze": {
                "x": float(gaze_unit[0]),
                "y": float(gaze_unit[1]),
                "z": float(gaze_unit[2]),
            },
        })

    return results, skipped


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MPIIFaceGaze 학습 라벨 생성")
    parser.add_argument(
        "--subjects", nargs="+", default=ALL_TRAIN_SUBJECTS,
        help="처리할 참여자 ID (기본: p01-p14)",
    )
    parser.add_argument(
        "--every", type=int, default=3,
        help="샘플링 간격 (기본 3 → 약 1/3 샘플, 1이면 전체)",
    )
    parser.add_argument("--out", default=OUTPUT_PATH, help="출력 JSON 경로")
    args = parser.parse_args()

    estimator = HeadPoseEstimator({"weights": HEADPOSE_WEIGHTS, "device": "cpu"})

    all_labels: list = []
    total_skipped = 0

    for subject in args.subjects:
        print(f"[{subject}] 처리 중...")
        labels, skipped = _process_subject(subject, estimator, args.every)
        all_labels.extend(labels)
        total_skipped += skipped
        print(f"  완료: {len(labels)}개 수집, {skipped}개 제외")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {args.out}")
    print(f"총 샘플: {len(all_labels)} | 제외: {total_skipped}")


if __name__ == "__main__":
    main()
