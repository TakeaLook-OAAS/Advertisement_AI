"""
Attention 판정 진단: gaze 각도(angle_deg)가 looking/not_looking을 구분할 신호를
실제로 갖고 있는지 확인한다.

look_judge.py의 angle_deg는 distance_adaptive 설정과 무관하게 항상 계산되는
"gaze 벡터 vs 카메라 정면" 전체 각도다 (judge()의 진단용 값). 이 스크립트는
attention.logic의 distance_adaptive/hysteresis 설정과 상관없이, GT looking=True/False
그룹별로 이 각도의 분포를 비교하고, 고정 threshold_deg를 스윕해서 어느 값이든
쓸만한 정확도가 나오는지 찾는다.

- 두 그룹 분포가 잘 안 겹치면: threshold만 잘 잡으면 됨 (스윕 결과의 best accuracy가 근거)
- 많이 겹치면: threshold 튜닝으로는 못 고치는, gaze/headpose 자체의 정확도 문제

사용법:
    python -m tests.benchmark.attention.analyze_gaze_angle
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.logic.look_judge import LookJudge
from src.models.eye.eye_pytorch import EyeDetector
from src.models.gaze.gaze_pytorch import GazeDetector
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.utils.types import BBoxXYXY, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["attention"]


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


def compute_angle_deg(
    frame,
    person_bbox: BBoxXYXY,
    face_bbox: BBoxXYXY,
    headpose_model: HeadPoseEstimator,
    eye_model: EyeDetector,
    gaze_model: GazeDetector,
    look_judge: LookJudge,
) -> float:
    track = Track(track_id=0, bbox=person_bbox, crop_bbox=face_bbox)

    track = headpose_model.infer(frame, track)
    track = eye_model.detect(frame, track)
    track = gaze_model.detect(frame, track)

    # angle_deg는 distance_adaptive 여부와 무관하게 항상 같은 방식(코사인 유사도)으로 계산된다.
    result = look_judge.judge(track.gaze)
    return result.angle_deg


def print_stats(name: str, angles: List[float]) -> None:
    arr = np.array(angles)
    print(f"\n  [{name}]  n={len(arr)}")
    if len(arr) == 0:
        return
    print(
        f"    mean={arr.mean():.1f}  median={np.median(arr):.1f}  std={arr.std():.1f}  "
        f"min={arr.min():.1f}  max={arr.max():.1f}"
    )
    print(
        f"    p10={np.percentile(arr, 10):.1f}  p25={np.percentile(arr, 25):.1f}  "
        f"p75={np.percentile(arr, 75):.1f}  p90={np.percentile(arr, 90):.1f}"
    )


def print_histogram(looking: List[float], not_looking: List[float], bin_width: int = 10, max_deg: int = 180) -> None:
    bins = list(range(0, max_deg + bin_width, bin_width))
    hist_l, _ = np.histogram(looking, bins=bins)
    hist_n, _ = np.histogram(not_looking, bins=bins)
    max_count = max(hist_l.max(initial=0), hist_n.max(initial=0), 1)
    scale = 40.0 / max_count

    print(f"\n  {'deg':>8}  {'looking':<42}  {'not_looking':<42}")
    for i in range(len(bins) - 1):
        label = f"{bins[i]:>3}-{bins[i+1]:<3}"
        bar_l = "#" * int(round(hist_l[i] * scale))
        bar_n = "#" * int(round(hist_n[i] * scale))
        print(f"  {label:>8}  {bar_l:<30} {hist_l[i]:>4}  {bar_n:<30} {hist_n[i]:>4}")


def sweep_threshold(looking: List[float], not_looking: List[float]) -> None:
    angles = np.array(looking + not_looking)
    gts = np.array([True] * len(looking) + [False] * len(not_looking))

    print(f"\n  {'threshold':>10}  {'accuracy':>10}  {'precision':>10}  {'recall':>10}  {'f1':>10}  {'pred_looking%':>14}")

    best = (-1.0, None)
    for t in range(0, 185, 5):
        preds = angles <= t
        tp = int(np.sum(preds & gts))
        fp = int(np.sum(preds & ~gts))
        fn = int(np.sum(~preds & gts))
        tn = int(np.sum(~preds & ~gts))

        acc = (tp + tn) / len(gts)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        pred_ratio = (tp + fp) / len(gts) * 100

        marker = ""
        if acc > best[0]:
            best = (acc, t)
            marker = "  <-- best"

        print(f"  {t:>10}  {acc:>10.3f}  {precision:>10.3f}  {recall:>10.3f}  {f1:>10.3f}  {pred_ratio:>13.1f}%{marker}")

    print(f"\n  최고 accuracy: threshold_deg={best[1]} → accuracy={best[0]:.3f}")

    baseline = max(np.mean(gts), 1 - np.mean(gts))
    print(f"  참고: 무조건 다수 클래스로 찍는 베이스라인 accuracy={baseline:.3f}")


def main() -> None:
    cfg = load_config()
    images_dir = cfg["images_dir"]
    labels_path = os.path.join(cfg["labels_dir"], cfg["labels_file"])

    if not os.path.exists(labels_path):
        logger.error(f"라벨 파일이 없습니다: {labels_path}")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    headpose_model = HeadPoseEstimator(cfg["headpose"])
    eye_model = EyeDetector(cfg["eye"])
    gaze_model = GazeDetector(cfg["gaze"])
    look_judge = LookJudge(cfg["logic"])

    angles_looking: List[float] = []
    angles_not_looking: List[float] = []

    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        faces_by_id = {f["id"]: f for f in item.get("faces", [])}

        for p in item.get("persons", []):
            if "looking" not in p:
                continue

            face = faces_by_id.get(p["id"])
            if face is None:
                continue

            angle = compute_angle_deg(
                frame, parse_bbox(p), parse_bbox(face),
                headpose_model, eye_model, gaze_model, look_judge,
            )

            if p["looking"]:
                angles_looking.append(angle)
            else:
                angles_not_looking.append(angle)

    print("\n=== Gaze Angle 분포 진단 (threshold_deg 설정과 무관한 raw angle_deg) ===")
    print_stats("gt_looking", angles_looking)
    print_stats("gt_not_looking", angles_not_looking)
    print_histogram(angles_looking, angles_not_looking)

    print("\n=== 고정 threshold_deg 스윕 (distance_adaptive 무시, angle_deg <= threshold 기준) ===")
    sweep_threshold(angles_looking, angles_not_looking)


if __name__ == "__main__":
    main()
