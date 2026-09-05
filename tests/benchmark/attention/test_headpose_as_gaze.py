"""
Attention 판정 파이프라인에서 gaze vector 자리를 headpose(yaw/pitch)로 만든 벡터로
갈아끼워서 실제 LookJudge로 판정해본다.

test_only_headpose.py와 다른 점: 거긴 hypot(yaw, pitch) <= threshold_deg로 직접 채점하는
별도의 간이 로직이었다. 이 스크립트는 EyeDetector/GazeDetector만 빼고, 판정 자체는
운영과 완전히 동일한 LookJudge.judge()를 그대로 쓴다 — distance_adaptive/threshold_deg
설정(configs/test.yaml → attention.logic)도 그대로 적용된다.

headpose -> gaze 벡터 변환:
    gx = sin(yaw) * cos(pitch)
    gy = sin(pitch)
    gz = -cos(yaw) * cos(pitch)
정면(yaw=pitch=0)이면 look_judge.py의 카메라 정면 벡터 (0,0,-1)과 일치하고, 항상 단위벡터다.
distance_adaptive가 꺼져 있으면 전체 각도(angle_deg)만 threshold_deg와 비교하므로 좌우 부호는
결과에 영향이 없다. distance_adaptive를 켜서 쓸 경우에만 gx의 좌우 부호가 horizontal_deg
계산에 실제로 반영되니 주의(실제 gaze 모델의 좌우 부호 컨벤션과 일치하는지 별도 검증 필요).

설정: configs/test.yaml → attention (기존 섹션 그대로 재사용, 별도 설정 불필요)

사용법:
    python -m tests.benchmark.attention.test_headpose_as_gaze
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

import cv2
import yaml
from loguru import logger

from src.logic.look_judge import LookJudge
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.utils.types import BBoxXYXY, Gaze, HeadPose, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["attention"]


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


def headpose_to_gaze(hp: HeadPose) -> Gaze:
    yaw = math.radians(hp.yaw)
    pitch = math.radians(hp.pitch)
    gx = math.sin(yaw) * math.cos(pitch)
    gy = math.sin(pitch)
    gz = -math.cos(yaw) * math.cos(pitch)
    return Gaze(x=gx, y=gy, z=gz)


def predict_is_looking(
    frame,
    person_bbox: BBoxXYXY,
    face_bbox: BBoxXYXY,
    headpose_model: HeadPoseEstimator,
    look_judge: LookJudge,
) -> Optional[bool]:
    track = Track(track_id=0, bbox=person_bbox, crop_bbox=face_bbox)
    track = headpose_model.infer(frame, track)
    if track.headpose is None:
        return None

    gaze = headpose_to_gaze(track.headpose)

    face_height_px = face_bbox.h()
    offset_deg_x, offset_deg_y = 0.0, 0.0
    if look_judge.distance_adaptive_enabled:
        cx, cy = face_bbox.center()
        offset_deg_x, offset_deg_y = look_judge._offset_deg_from_center(
            cx, cy, frame.shape[1], frame.shape[0]
        )

    result = look_judge.judge(gaze, face_height_px, offset_deg_x, offset_deg_y)
    return result.is_looking


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
    look_judge = LookJudge(cfg["logic"])

    tp = fp = tn = fn = 0
    no_face_match = 0
    no_headpose = 0

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
                no_face_match += 1
                continue

            pred = predict_is_looking(
                frame, parse_bbox(p), parse_bbox(face), headpose_model, look_judge,
            )
            if pred is None:
                no_headpose += 1
                continue

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

    da = "ON" if look_judge.distance_adaptive_enabled else "OFF"
    print(f"\n=== Attention 판정 벤치마크 (gaze 자리에 headpose 벡터 대입, distance_adaptive={da}) ===")
    print(f"  평가 대상: {total}명 (얼굴 매칭 실패 {no_face_match}명, headpose 추정 실패 {no_headpose}명 제외)")
    print(f"\n  {'':>12}  pred_looking  pred_not_looking")
    print(f"  {'gt_looking':>12}  {tp:>12}  {fn:>16}")
    print(f"  {'gt_not_looking':>12}  {fp:>12}  {tn:>16}")
    print(f"\n  accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")


if __name__ == "__main__":
    main()
