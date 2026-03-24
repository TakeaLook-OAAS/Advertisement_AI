"""
속성 모델 벤치마크: MiVOLO (gender, age_group)
두 가중치를 같은 테스트 데이터로 평가하여 Accuracy, F1-score를 비교한다.

사용법:
    python -m tests.benchmark.test_attr

데이터 구조:
    data/benchmark/attr/
    ├── images/         # 테스트 이미지
    └── labels.json     # 정답 라벨
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from src.utils.types import BBoxXYXY, Track


# ── 설정 ──────────────────────────────────────────────────────
DATA_DIR = "data/benchmark/attr"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

MIVOLO_WEIGHTS_A = "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth"
MIVOLO_WEIGHTS_B = "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth"

MIVOLO_CFG_BASE = {
    "device": "cpu",
    "repo_root": "MiVOLO",
    "min_face_size": 20,
    "min_person_size": 40,
    "use_persons": True,
}


# ── 메트릭 함수 ──────────────────────────────────────────────

def compute_accuracy(preds: List[str], gts: List[str]) -> float:
    """정확도 = 맞은 수 / 전체 수"""
    if not gts:
        return 0.0
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    return correct / len(gts)


def compute_f1(preds: List[str], gts: List[str]) -> float:
    """매크로 F1-score (클래스별 F1의 평균)"""
    all_labels = set(preds) | set(gts)
    f1_scores = []

    for label in all_labels:
        tp = sum(1 for p, g in zip(preds, gts) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, gts) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, gts) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


# ── 데이터 로드 ──────────────────────────────────────────────

def load_labels() -> List[Dict[str, Any]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── MiVOLO 벤치마크 ─────────────────────────────────────────

def bench_mivolo(
    weights: str, labels: List[Dict[str, Any]]
) -> Tuple[float, float, float, float]:
    """
    MiVOLO 모델 하나를 평가한다.
    Returns: (gender_acc, gender_f1, age_acc, age_f1)
    """
    from src.models.mivolo_attr import MiVOLOAttr

    cfg = {**MIVOLO_CFG_BASE, "model": weights}
    model = MiVOLOAttr(cfg)

    pred_genders: List[str] = []
    gt_genders: List[str] = []
    pred_ages: List[str] = []
    gt_ages: List[str] = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        person_bbox = parse_bbox(item["bbox"])
        face_bbox = parse_bbox(item["face"])
        track = Track(track_id=0, bbox=person_bbox, crop_bbox=face_bbox)

        tracks = model.infer(frame, [track])
        track = tracks[0]

        if track.attr is None:
            continue

        pred_genders.append(track.attr.gender.value)
        gt_genders.append(item["gender"])
        pred_ages.append(track.attr.age_group.value)
        gt_ages.append(item["age_group"])

    gender_acc = compute_accuracy(pred_genders, gt_genders)
    gender_f1 = compute_f1(pred_genders, gt_genders)
    age_acc = compute_accuracy(pred_ages, gt_ages)
    age_f1 = compute_f1(pred_ages, gt_ages)

    return gender_acc, gender_f1, age_acc, age_f1


# ── 결과 출력 ────────────────────────────────────────────────

def print_comparison(
    result_a: Tuple[float, float, float, float],
    result_b: Tuple[float, float, float, float],
) -> None:
    print("\n=== Gender ===")
    print(f"  Model A: accuracy={result_a[0]:.4f}, f1={result_a[1]:.4f}")
    print(f"  Model B: accuracy={result_b[0]:.4f}, f1={result_b[1]:.4f}")

    print("\n=== Age Group ===")
    print(f"  Model A: accuracy={result_a[2]:.4f}, f1={result_a[3]:.4f}")
    print(f"  Model B: accuracy={result_b[2]:.4f}, f1={result_b[3]:.4f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일이 없습니다: {LABELS_PATH}")
        logger.info("data/benchmark/attr/labels.json 을 먼저 준비하세요.")
        return

    labels = load_labels()
    logger.info(f"테스트 이미지 수: {len(labels)}")

    logger.info("MiVOLO Model A 평가 중...")
    result_a = bench_mivolo(MIVOLO_WEIGHTS_A, labels)
    logger.info("MiVOLO Model B 평가 중...")
    result_b = bench_mivolo(MIVOLO_WEIGHTS_B, labels)

    print_comparison(result_a, result_b)


if __name__ == "__main__":
    main()
