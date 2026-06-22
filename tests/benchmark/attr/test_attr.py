"""
속성 모델 벤치마크: MiVOLO (gender, age_group)
정답 라벨과 현재 가중치를 비교하여 Accuracy, F1-score를 측정한다.

사용법:
    python -m tests.benchmark.attr.test_attr

데이터 구조:
    data/benchmark/attr/
    ├── images/         # 테스트 이미지
    └── labels.json     # 정답 라벨
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
        return yaml.safe_load(f)["attr"]


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


# ── 유틸 ─────────────────────────────────────────────────────

def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── MiVOLO 벤치마크 ─────────────────────────────────────────

def bench_mivolo(
    weights: str,
    images_dir: str,
    labels: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """
    MiVOLO 모델 하나를 평가한다.
    Returns: (gender_acc, gender_f1, age_acc, age_f1)
    """
    from src.models.mivolo_attr import MiVOLOAttr

    cfg = {**model_cfg, "model": weights}
    model = MiVOLOAttr(cfg)

    pred_genders: List[str] = []
    gt_genders: List[str] = []
    pred_ages: List[str] = []
    gt_ages: List[str] = []

    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
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

def print_result(result: Tuple[float, float, float, float]) -> None:
    print("\n=== Gender ===")
    print(f"  accuracy={result[0]:.4f}, f1={result[1]:.4f}")
    print("\n=== Age Group ===")
    print(f"  accuracy={result[2]:.4f}, f1={result[3]:.4f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    data_dir = cfg["data_dir"]
    images_dir = os.path.join(data_dir, cfg["images_subdir"])
    labels_path = os.path.join(data_dir, cfg["labels_file"])
    weights = cfg["weights"]
    model_cfg = {
        "device": cfg["device"],
        "repo_root": cfg["repo_root"],
        "min_face_size": cfg["min_face_size"],
        "min_person_size": cfg["min_person_size"],
        "use_persons": cfg["use_persons"],
    }

    if not os.path.exists(labels_path):
        logger.error(f"라벨 파일이 없습니다: {labels_path}")
        logger.info(f"{labels_path} 을 먼저 준비하세요.")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.info(f"테스트 이미지 수: {len(labels)}")
    logger.info("MiVOLO 평가 중...")
    print_result(bench_mivolo(weights, images_dir, labels, model_cfg))


if __name__ == "__main__":
    main()
