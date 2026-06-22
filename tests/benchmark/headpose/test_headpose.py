"""
Headpose 모델 벤치마크: 6DRepNet (yaw, pitch, roll)
정답 라벨과 현재 가중치를 비교하여 MAE(Mean Absolute Error)를 측정한다.

사용법:
    python -m tests.benchmark.headpose.test_headpose

데이터 구조:
    data/benchmark/headpose/
    ├── AFLW2000/       # 테스트 이미지 + .mat 파일 (원본)
    └── labels.json     # convert_aflw2000.py 로 생성
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.utils.types import BBoxXYXY, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["headpose"]


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


# ── 유틸 ─────────────────────────────────────────────────────

def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── Headpose 벤치마크 ────────────────────────────────────────

def bench_headpose(
    weights: str,
    images_dir: str,
    labels: List[Dict[str, Any]],
    device: str = "cpu",
) -> Tuple[float, float, float, float]:
    """
    HeadPoseEstimator 모델 하나를 평가한다.
    Returns: (mae_yaw, mae_pitch, mae_roll, mae_mean)
    """
    from src.models.headpose_6drepnet import HeadPoseEstimator

    cfg = {"weights": weights, "device": device}
    estimator = HeadPoseEstimator(cfg)

    preds: List[Tuple[float, float, float]] = []
    gts: List[Tuple[float, float, float]] = []

    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        face_bbox = parse_bbox(item["face"])
        track = Track(track_id=0, bbox=face_bbox, crop_bbox=face_bbox)

        track = estimator.infer(frame, track)

        # 모델 실패 시 HeadPose(0,0,0)을 반환하므로 None 체크 대신 zero 체크
        if track.headpose is None or (
            track.headpose.yaw == 0.0
            and track.headpose.pitch == 0.0
            and track.headpose.roll == 0.0
        ):
            continue

        preds.append((track.headpose.yaw, track.headpose.pitch, track.headpose.roll))
        gts.append((item["yaw"], item["pitch"], item["roll"]))

    logger.info(f"추론 성공: {len(preds)} / {len(labels)}")
    return compute_mae(preds, gts)


# ── 결과 출력 ────────────────────────────────────────────────

def print_result(result: Tuple[float, float, float, float]) -> None:
    print("\n=== Headpose MAE (degrees) ===")
    print(f"  {'Yaw':>8}  {'Pitch':>8}  {'Roll':>8}  {'Mean':>8}")
    print(f"  {result[0]:8.2f}  {result[1]:8.2f}  {result[2]:8.2f}  {result[3]:8.2f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    data_dir = cfg["data_dir"]
    images_dir = os.path.join(data_dir, cfg["images_subdir"])
    labels_path = os.path.join(data_dir, cfg["labels_file"])
    weights = cfg["weights"]
    device = cfg["device"]

    if not os.path.exists(labels_path):
        logger.error(f"라벨 파일이 없습니다: {labels_path}")
        logger.info(f"{labels_path} 을 먼저 준비하세요.")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.info(f"테스트 이미지 수: {len(labels)}")
    logger.info("Headpose 평가 중...")
    print_result(bench_headpose(weights, images_dir, labels, device))


if __name__ == "__main__":
    main()
