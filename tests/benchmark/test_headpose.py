"""
Headpose 모델 벤치마크: 6DRepNet (yaw, pitch, roll)
두 가중치를 같은 테스트 데이터로 평가하여 MAE(Mean Absolute Error)를 비교한다.

사용법:
    python -m tests.benchmark.test_headpose

데이터 구조:
    data/benchmark/headpose/
    ├── images/         # 테스트 이미지
    └── labels.json     # 정답 라벨
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from src.utils.types import BBoxXYXY, Track


# ── 설정 ──────────────────────────────────────────────────────
DATA_DIR = "data/benchmark/headpose"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

HEADPOSE_WEIGHTS_A = "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"
HEADPOSE_WEIGHTS_B = "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"


# ── 메트릭 함수 ──────────────────────────────────────────────

def compute_mae(
    preds: List[Tuple[float, float, float]],
    gts: List[Tuple[float, float, float]],
) -> Tuple[float, float, float, float]:
    """
    MAE를 yaw, pitch, roll 각각 + 전체 평균으로 계산한다.
    Returns: (mae_yaw, mae_pitch, mae_roll, mae_mean)
    """
    if not preds:
        return 0.0, 0.0, 0.0, 0.0

    yaw_errors = []
    pitch_errors = []
    roll_errors = []

    for (py, pp, pr), (gy, gp, gr) in zip(preds, gts):
        yaw_errors.append(abs(py - gy))
        pitch_errors.append(abs(pp - gp))
        roll_errors.append(abs(pr - gr))

    mae_yaw = float(np.mean(yaw_errors))
    mae_pitch = float(np.mean(pitch_errors))
    mae_roll = float(np.mean(roll_errors))
    mae_mean = (mae_yaw + mae_pitch + mae_roll) / 3.0

    return mae_yaw, mae_pitch, mae_roll, mae_mean


# ── 데이터 로드 ──────────────────────────────────────────────

def load_labels() -> List[Dict[str, Any]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── Headpose 벤치마크 ────────────────────────────────────────

def bench_headpose(
    weights: str, labels: List[Dict[str, Any]]
) -> Tuple[float, float, float, float]:
    """
    HeadPoseEstimator 모델 하나를 평가한다.
    Returns: (mae_yaw, mae_pitch, mae_roll, mae_mean)
    """
    from src.models.headpose_6drepnet import HeadPoseEstimator

    cfg = {"weights": weights, "device": "cpu"}
    estimator = HeadPoseEstimator(cfg)

    preds: List[Tuple[float, float, float]] = []
    gts: List[Tuple[float, float, float]] = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        face_bbox = parse_bbox(item["face"])
        track = Track(track_id=0, bbox=face_bbox, crop_bbox=face_bbox)

        track = estimator.infer(frame, track)

        if track.headpose is None:
            continue

        preds.append((track.headpose.yaw, track.headpose.pitch, track.headpose.roll))
        gts.append((item["yaw"], item["pitch"], item["roll"]))

    return compute_mae(preds, gts)


# ── 결과 출력 ────────────────────────────────────────────────

def print_comparison(
    result_a: Tuple[float, float, float, float],
    result_b: Tuple[float, float, float, float],
) -> None:
    print("\n=== Headpose MAE (degrees) ===")
    print(f"           {'Yaw':>8}  {'Pitch':>8}  {'Roll':>8}  {'Mean':>8}")
    print(
        f"  Model A: {result_a[0]:8.2f}  {result_a[1]:8.2f}  "
        f"{result_a[2]:8.2f}  {result_a[3]:8.2f}"
    )
    print(
        f"  Model B: {result_b[0]:8.2f}  {result_b[1]:8.2f}  "
        f"{result_b[2]:8.2f}  {result_b[3]:8.2f}"
    )


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일이 없습니다: {LABELS_PATH}")
        logger.info("data/benchmark/headpose/labels.json 을 먼저 준비하세요.")
        return

    labels = load_labels()
    logger.info(f"테스트 이미지 수: {len(labels)}")

    logger.info("Headpose Model A 평가 중...")
    result_a = bench_headpose(HEADPOSE_WEIGHTS_A, labels)
    logger.info("Headpose Model B 평가 중...")
    result_b = bench_headpose(HEADPOSE_WEIGHTS_B, labels)

    print_comparison(result_a, result_b)


if __name__ == "__main__":
    main()
