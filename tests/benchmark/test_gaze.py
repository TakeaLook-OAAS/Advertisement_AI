"""
Gaze 모델 벤치마크: gaze-estimation-adas (3D gaze vector)
두 가중치를 같은 테스트 데이터로 평가하여 Angular Error를 비교한다.

사용법:
    python -m tests.benchmark.test_gaze

데이터 구조:
    data/benchmark/gaze/
    ├── images/         # 테스트 이미지
    └── labels.json     # 정답 라벨
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from src.utils.types import BBoxXYXY, HeadPose, Track


# ── 설정 ──────────────────────────────────────────────────────
DATA_DIR = "data/benchmark/gaze"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

GAZE_WEIGHTS_A = "weights/gaze/gaze-estimation-adas-0002.xml"
GAZE_WEIGHTS_B = "weights/gaze/gaze-estimation-adas-0002.xml"


# ── 메트릭 함수 ──────────────────────────────────────────────

def angular_error(
    pred: Tuple[float, float, float],
    gt: Tuple[float, float, float],
) -> float:
    """
    두 3D 벡터 사이의 각도 오차 (degrees).
    arccos(cosine_similarity)로 계산한다.
    """
    pred_arr = np.array(pred, dtype=np.float64)
    gt_arr = np.array(gt, dtype=np.float64)

    # 정규화
    pred_norm = np.linalg.norm(pred_arr)
    gt_norm = np.linalg.norm(gt_arr)

    if pred_norm < 1e-8 or gt_norm < 1e-8:
        return 180.0  # 벡터가 거의 0이면 최대 오차

    cos_sim = np.dot(pred_arr, gt_arr) / (pred_norm * gt_norm)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_sim)))


def compute_angular_stats(errors: List[float]) -> Tuple[float, float]:
    """
    Angular error 리스트에서 mean, median을 계산한다.
    Returns: (mean, median)
    """
    if not errors:
        return 0.0, 0.0
    return float(np.mean(errors)), float(np.median(errors))


# ── 데이터 로드 ──────────────────────────────────────────────

def load_labels() -> List[Dict[str, Any]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── Gaze 벤치마크 ────────────────────────────────────────────

def bench_gaze(
    weights: str, labels: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    GazeDetector 모델 하나를 평가한다.
    Returns: (mean_error, median_error)
    """
    from src.models.gaze_openvino import GazeDetector

    cfg = {"weights": weights, "device": "CPU"}
    detector = GazeDetector(cfg)

    errors: List[float] = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        left_eye = parse_bbox(item["left_eye"])
        right_eye = parse_bbox(item["right_eye"])
        hp_data = item["headpose"]
        headpose = HeadPose(
            yaw=hp_data["yaw"],
            pitch=hp_data["pitch"],
            roll=hp_data["roll"],
        )

        track = Track(
            track_id=0,
            bbox=left_eye,  # fallback용
            left_eye=left_eye,
            right_eye=right_eye,
            headpose=headpose,
        )

        track = detector.detect(frame, track)

        if track.gaze is None:
            continue

        gt_gaze = item["gaze"]
        pred = (track.gaze.x, track.gaze.y, track.gaze.z)
        gt = (gt_gaze["x"], gt_gaze["y"], gt_gaze["z"])

        err = angular_error(pred, gt)
        errors.append(err)

    return compute_angular_stats(errors)


# ── 결과 출력 ────────────────────────────────────────────────

def print_comparison(
    result_a: Tuple[float, float],
    result_b: Tuple[float, float],
) -> None:
    print("\n=== Gaze Angular Error (degrees) ===")
    print(f"  Model A: mean={result_a[0]:.2f}, median={result_a[1]:.2f}")
    print(f"  Model B: mean={result_b[0]:.2f}, median={result_b[1]:.2f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일이 없습니다: {LABELS_PATH}")
        logger.info("data/benchmark/gaze/labels.json 을 먼저 준비하세요.")
        return

    labels = load_labels()
    logger.info(f"테스트 이미지 수: {len(labels)}")

    logger.info("Gaze Model A 평가 중...")
    result_a = bench_gaze(GAZE_WEIGHTS_A, labels)
    logger.info("Gaze Model B 평가 중...")
    result_b = bench_gaze(GAZE_WEIGHTS_B, labels)

    print_comparison(result_a, result_b)


if __name__ == "__main__":
    main()
