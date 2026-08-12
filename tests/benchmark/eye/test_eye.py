"""
Eye 모델 벤치마크: OpenVINO(facial-landmarks-35-adas-0002) vs PyTorch(MSE / MSE+GIoU 등)
여러 가중치의 눈 bbox 정확도를 비교한다.
WFLW 공식 test split(labels_test.json, 학습에 전혀 쓰이지 않은 2500장)을 기준으로
정답 bbox와 예측 bbox의 mean/median IoU를 측정한다.

사용법:
    python -m tests.benchmark.eye.test_eye

데이터 구조:
    data/benchmark/eye/
    └── labels_test.json     # eye_generate_labels.py --split test 가 생성
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.utils.types import BBoxXYXY, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["eye"]


# ── 유틸 ─────────────────────────────────────────────────────

def box_iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def denormalize_box(norm: Dict[str, float], face_bbox: BBoxXYXY) -> BBoxXYXY:
    """face_bbox 기준 정규화 좌표([0,1]) → 원본 프레임 픽셀 좌표"""
    fw, fh = face_bbox.w(), face_bbox.h()
    return BBoxXYXY(
        x1=int(face_bbox.x1 + norm["x1"] * fw),
        y1=int(face_bbox.y1 + norm["y1"] * fh),
        x2=int(face_bbox.x1 + norm["x2"] * fw),
        y2=int(face_bbox.y1 + norm["y2"] * fh),
    )


# ── 벤치마크 ──────────────────────────────────────────────────

def bench_eye(
    weights: str,
    backend: str,
    device: str,
    images_dir: str,
    labels: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """
    EyeDetector 하나를 평가한다.
    backend: "openvino" | "pytorch"
    Returns: (mean_iou, median_iou, avg_time_ms)
    """
    if backend == "openvino":
        from src.models.openvino.eye_openvino import EyeDetector
    else:
        from src.models.eye.eye_pytorch import EyeDetector

    detector = EyeDetector({"weights": weights, "device": device})

    ious: List[float] = []
    call_times: List[float] = []

    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        fb = item["face_bbox"]
        face_bbox = BBoxXYXY(x1=fb["x1"], y1=fb["y1"], x2=fb["x2"], y2=fb["y2"])

        track = Track(track_id=0, bbox=face_bbox, crop_bbox=face_bbox)
        t0 = time.perf_counter()
        track = detector.detect(frame, track)
        call_times.append(time.perf_counter() - t0)
        if track.left_eye is None or track.right_eye is None:
            continue

        gt_left = denormalize_box(item["eyes"]["left"], face_bbox)
        gt_right = denormalize_box(item["eyes"]["right"], face_bbox)

        ious.append(box_iou(track.left_eye, gt_left))
        ious.append(box_iou(track.right_eye, gt_right))

    avg_time_ms = float(np.mean(call_times)) * 1000 if call_times else 0.0
    if not ious:
        return 0.0, 0.0, avg_time_ms
    return float(np.mean(ious)), float(np.median(ious)), avg_time_ms


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    data_dir = cfg["data_dir"]
    images_dir = cfg["images_dir"]
    labels_path = os.path.join(data_dir, cfg["labels_file"])

    if not os.path.exists(labels_path):
        logger.error(f"라벨 파일이 없습니다: {labels_path}")
        logger.info(f"{labels_path} 을 먼저 준비하세요. (python -m scripts.eye.eye_generate_labels --split test)")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    logger.info(f"테스트 이미지 수: {len(labels)}")

    results: Dict[str, Tuple[float, float, float]] = {}
    for name, w in cfg["weights"].items():
        path = w["path"]
        if not os.path.exists(path):
            logger.warning(f"{name}: 가중치 없음 ({path}) — 건너뜀")
            continue
        backend = "openvino" if name == "openvino" else "pytorch"
        logger.info(f"{name} 평가 중...")
        results[name] = bench_eye(path, backend, w["device"], images_dir, labels)

    print("\n=== Eye bbox IoU 비교 (WFLW test set) ===")
    print(f"{'모델':<10}  {'mean IoU':>10}  {'median IoU':>10}  {'avg ms':>10}")
    print("-" * 48)
    for name, (mean, median, avg_ms) in results.items():
        print(f"{name:<10}  {mean:>9.3f}  {median:>9.3f}  {avg_ms:>9.2f}")


if __name__ == "__main__":
    main()
