"""
Attention(시선) 최종 판정 벤치마크: 실제로 광고를 봤는지(GT) vs 우리 파이프라인이
봤다고 판정했는지(headpose→eye→gaze→look_judge)를 비교한다.

라벨은 독립된 정지 이미지 기준이라 히스테리시스(연속 프레임 조건)는 적용하지 않고,
프레임 단위 raw 판정만 비교한다.

사용법:
    python -m tests.benchmark.attention.test_attention

데이터 구조:
    data/benchmark/attention/
    ├── images/         # LabelImg로 라벨링한 이미지 + .txt + classes.txt
    │                    # classes.txt: person_looking / person_not_looking / face
    └── labels.json      # annotations_to_json.py 가 생성
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
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


def match_face(person: BBoxXYXY, faces: List[BBoxXYXY], used: set) -> Optional[int]:
    """person bbox 안에 중심이 들어오는 face 중 person 중심에 가장 가까운 것을 매칭한다."""
    px, py = person.center()
    best_idx, best_dist = None, float("inf")
    for i, f in enumerate(faces):
        if i in used:
            continue
        fx, fy = f.center()
        if not (person.x1 <= fx <= person.x2 and person.y1 <= fy <= person.y2):
            continue
        dist = (fx - px) ** 2 + (fy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def predict_is_looking(
    frame,
    person_bbox: BBoxXYXY,
    face_bbox: BBoxXYXY,
    headpose_model: HeadPoseEstimator,
    eye_model: EyeDetector,
    gaze_model: GazeDetector,
    look_judge: LookJudge,
) -> bool:
    track = Track(track_id=0, bbox=person_bbox, crop_bbox=face_bbox)

    track = headpose_model.infer(frame, track)
    track = eye_model.detect(frame, track)
    track = gaze_model.detect(frame, track)

    face_height_px = face_bbox.h()
    offset_deg_x, offset_deg_y = 0.0, 0.0
    if look_judge.distance_adaptive_enabled:
        cx, cy = face_bbox.center()
        offset_deg_x, offset_deg_y = look_judge._offset_deg_from_center(
            cx, cy, frame.shape[1], frame.shape[0]
        )

    result = look_judge.judge(track.gaze, face_height_px, offset_deg_x, offset_deg_y)
    return result.is_looking


def main() -> None:
    cfg = load_config()
    images_dir = os.path.join(cfg["data_dir"], cfg["images_subdir"])
    labels_path = os.path.join(cfg["data_dir"], cfg["labels_file"])

    if not os.path.exists(labels_path):
        logger.error(f"라벨 파일이 없습니다: {labels_path}")
        logger.info("먼저 python -m tests.benchmark.detection.annotations_to_json 을 실행하세요.")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    headpose_model = HeadPoseEstimator(cfg["headpose"])
    eye_model = EyeDetector(cfg["eye"])
    gaze_model = GazeDetector(cfg["gaze"])
    look_judge = LookJudge(cfg["logic"])

    tp = fp = tn = fn = 0
    no_face_match = 0
    no_gt_label = 0

    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        faces = [parse_bbox(f) for f in item.get("faces", [])]
        used_faces: set = set()

        for p in item.get("persons", []):
            if "looking" not in p:
                no_gt_label += 1
                continue

            person_bbox = parse_bbox(p)
            face_idx = match_face(person_bbox, faces, used_faces)
            if face_idx is None:
                no_face_match += 1
                continue
            used_faces.add(face_idx)

            pred = predict_is_looking(
                frame, person_bbox, faces[face_idx],
                headpose_model, eye_model, gaze_model, look_judge,
            )
            gt = bool(p["looking"])

            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n=== Attention 판정 벤치마크 (raw, hysteresis 없음) ===")
    print(f"  평가 대상: {total}명 (얼굴 매칭 실패 {no_face_match}명, GT looking 라벨 없음 {no_gt_label}명 제외)")
    print(f"\n  {'':>12}  pred_looking  pred_not_looking")
    print(f"  {'gt_looking':>12}  {tp:>12}  {fn:>16}")
    print(f"  {'gt_not_looking':>12}  {fp:>12}  {tn:>16}")
    print(f"\n  accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")


if __name__ == "__main__":
    main()
