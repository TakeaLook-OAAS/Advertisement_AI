"""
검출 모델 벤치마크: YOLO(person bbox), FaceDetector(crop_bbox), EyeDetector(eye bbox)
두 가중치를 같은 테스트 데이터로 평가하여 IoU, mAP@0.5를 비교한다.

사용법:
    python -m tests.benchmark.test_detection

데이터 구조:
    data/benchmark/detection/
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
DATA_DIR = "data/benchmark/detection"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

# 모델 A / B 가중치 경로 (사용 전 수정)
YOLO_WEIGHTS_A = "weights/yolo/yolov8n.pt"
YOLO_WEIGHTS_B = "weights/yolo/yolov8s.pt"

FACE_WEIGHTS_A = "weights/face_detection/face-detection-adas-0001.xml"
FACE_WEIGHTS_B = "weights/face_detection/face-detection-adas-0001.xml"

EYE_WEIGHTS_A = "weights/eye_detection/facial-landmarks-35-adas-0002.xml"
EYE_WEIGHTS_B = "weights/eye_detection/facial-landmarks-35-adas-0002.xml"


# ── 메트릭 함수 ──────────────────────────────────────────────

def compute_iou(pred: BBoxXYXY, gt: BBoxXYXY) -> float:
    """두 bbox의 IoU를 계산한다."""
    ix1 = max(pred.x1, gt.x1)
    iy1 = max(pred.y1, gt.y1)
    ix2 = min(pred.x2, gt.x2)
    iy2 = min(pred.y2, gt.y2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_pred = pred.area()
    area_gt = gt.area()
    union = area_pred + area_gt - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_and_score(
    preds: List[BBoxXYXY],
    gts: List[BBoxXYXY],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int, float]:
    """
    예측 bbox와 정답 bbox를 greedy IoU 매칭한다.

    Returns: (tp, fp, fn, avg_iou)
        tp: IoU >= thresh인 매칭 수
        fp: 매칭 안 된 예측 수
        fn: 매칭 안 된 정답 수
        avg_iou: 매칭된 쌍들의 평균 IoU
    """
    matched_gt = set()
    ious = []

    for pred in preds:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            ious.append(best_iou)

    tp = len(ious)
    fp = len(preds) - tp
    fn = len(gts) - tp
    avg_iou = float(np.mean(ious)) if ious else 0.0

    return tp, fp, fn, avg_iou


def compute_map(all_tp: int, all_fp: int, all_fn: int) -> float:
    """전체 이미지에 대한 mAP@0.5 (= precision 기준 간이 계산)."""
    if all_tp + all_fp == 0:
        return 0.0
    precision = all_tp / (all_tp + all_fp)
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    # 간이 AP: precision * recall (정밀한 AP 곡선은 아님)
    return precision


# ── 데이터 로드 ──────────────────────────────────────────────

def load_labels() -> List[Dict[str, Any]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── YOLO 벤치마크 ────────────────────────────────────────────

def bench_yolo(weights: str, labels: List[Dict[str, Any]]) -> Tuple[float, float]:
    """YOLO 모델 하나를 평가한다. Returns: (mAP@0.5, avg_iou)"""
    from src.models.yolo_detector import YoloDetector

    cfg = {"model": weights, "device": "cpu", "conf": 0.5, "classes": [0]}
    detector = YoloDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        dets = detector.detect(frame)
        pred_boxes = [d.bbox for d in dets]
        gt_boxes = [parse_bbox(b) for b in item.get("persons", [])]

        tp, fp, fn, avg_iou = match_and_score(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if avg_iou > 0:
            all_ious.append(avg_iou)

    map50 = compute_map(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return map50, mean_iou


# ── Face 벤치마크 ────────────────────────────────────────────

def bench_face(weights: str, labels: List[Dict[str, Any]]) -> Tuple[float, float]:
    """FaceDetector 모델 하나를 평가한다. Returns: (mAP@0.5, avg_iou)"""
    from src.models.face_openvino import FaceDetector

    cfg = {"weights": weights, "device": "CPU", "conf_thresh": 0.5}
    detector = FaceDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        gt_faces = [parse_bbox(b) for b in item.get("faces", [])]
        gt_persons = item.get("persons", [])

        # person bbox마다 face 검출
        pred_faces = []
        for pb in gt_persons:
            person_bbox = parse_bbox(pb)
            track = Track(track_id=0, bbox=person_bbox)
            track = detector.detect(frame, track)
            if track.crop_bbox is not None:
                pred_faces.append(track.crop_bbox)

        tp, fp, fn, avg_iou = match_and_score(pred_faces, gt_faces)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if avg_iou > 0:
            all_ious.append(avg_iou)

    map50 = compute_map(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return map50, mean_iou


# ── Eye 벤치마크 ─────────────────────────────────────────────

def bench_eye(weights: str, labels: List[Dict[str, Any]]) -> Tuple[float, float]:
    """EyeDetector 모델 하나를 평가한다. Returns: (mAP@0.5, avg_iou)"""
    from src.models.eye_openvino import EyeDetector

    cfg = {"weights": weights, "device": "CPU"}
    detector = EyeDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        for eye_gt in item.get("eyes", []):
            gt_left = parse_bbox(eye_gt["left"])
            gt_right = parse_bbox(eye_gt["right"])

            face_bbox = parse_bbox(eye_gt.get("face", item.get("faces", [{}])[0]))
            track = Track(track_id=0, bbox=face_bbox, crop_bbox=face_bbox)
            track = detector.detect(frame, track)

            # left eye
            if track.left_eye is not None:
                iou_l = compute_iou(track.left_eye, gt_left)
                if iou_l >= 0.5:
                    total_tp += 1
                    all_ious.append(iou_l)
                else:
                    total_fp += 1
            else:
                total_fn += 1

            # right eye
            if track.right_eye is not None:
                iou_r = compute_iou(track.right_eye, gt_right)
                if iou_r >= 0.5:
                    total_tp += 1
                    all_ious.append(iou_r)
                else:
                    total_fp += 1
            else:
                total_fn += 1

    map50 = compute_map(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return map50, mean_iou


# ── 결과 출력 ────────────────────────────────────────────────

def print_comparison(title: str, result_a: Tuple[float, float], result_b: Tuple[float, float]) -> None:
    print(f"\n=== {title} ===")
    print(f"  Model A: mAP@0.5 = {result_a[0]:.4f}, avg IoU = {result_a[1]:.4f}")
    print(f"  Model B: mAP@0.5 = {result_b[0]:.4f}, avg IoU = {result_b[1]:.4f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일이 없습니다: {LABELS_PATH}")
        logger.info("data/benchmark/detection/labels.json 을 먼저 준비하세요.")
        return

    labels = load_labels()
    logger.info(f"테스트 이미지 수: {len(labels)}")

    # YOLO
    logger.info("YOLO Model A 평가 중...")
    yolo_a = bench_yolo(YOLO_WEIGHTS_A, labels)
    logger.info("YOLO Model B 평가 중...")
    yolo_b = bench_yolo(YOLO_WEIGHTS_B, labels)
    print_comparison("YOLO Person Detection", yolo_a, yolo_b)

    # Face
    logger.info("Face Model A 평가 중...")
    face_a = bench_face(FACE_WEIGHTS_A, labels)
    logger.info("Face Model B 평가 중...")
    face_b = bench_face(FACE_WEIGHTS_B, labels)
    print_comparison("Face Detection", face_a, face_b)

    # Eye
    logger.info("Eye Model A 평가 중...")
    eye_a = bench_eye(EYE_WEIGHTS_A, labels)
    logger.info("Eye Model B 평가 중...")
    eye_b = bench_eye(EYE_WEIGHTS_B, labels)
    print_comparison("Eye Detection", eye_a, eye_b)


if __name__ == "__main__":
    main()
