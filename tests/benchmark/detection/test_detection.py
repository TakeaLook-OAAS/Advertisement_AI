"""
검출 모델 벤치마크: YOLO(person bbox), FaceDetector(crop_bbox), EyeDetector(eye bbox)
정답 라벨과 현재 가중치를 비교하여 Precision/Recall/F1, IoU를 측정한다.

사용법:
    python -m tests.benchmark.detection.test_detection

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

IOU_THRESH = 0.5

YOLO_WEIGHTS = "weights/yolo/yolov8n.pt"
FACE_WEIGHTS = "weights/face_detection/face-detection-adas-0001.xml"
EYE_WEIGHTS  = "weights/eye_detection/facial-landmarks-35-adas-0002.xml"


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
    iou_thresh: float = IOU_THRESH,
) -> Tuple[int, int, int, List[float]]:
    """
    예측 bbox와 정답 bbox를 greedy IoU 매칭한다.

    Returns: (tp, fp, fn, matched_ious)
        tp: IoU >= thresh인 매칭 수
        fp: 매칭 안 된 예측 수 (오검)
        fn: 매칭 안 된 정답 수 (미검출)
        matched_ious: 매칭된 쌍들의 IoU 리스트 (위치 정렬 품질 전용, 페널티 미포함)
    """
    matched_gt = set()
    matched_ious: List[float] = []

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
            matched_ious.append(best_iou)

    tp = len(matched_ious)
    fp = len(preds) - tp
    fn = len(gts) - tp

    return tp, fp, fn, matched_ious


def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """전체 누적 TP/FP/FN으로 Precision, Recall, F1을 계산한다.

    - Precision: 검출한 것 중 맞은 비율 → 과검출(FP)에 민감
    - Recall:    정답 중 잡은 비율     → 미검출(FN)에 민감
    - F1:        둘의 조화평균          → 과검출/미검출 둘 다 페널티
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ── 데이터 로드 ──────────────────────────────────────────────

def load_labels() -> List[Dict[str, Any]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# 평가 결과: (precision, recall, f1, avg_iou)
Result = Tuple[float, float, float, float]


# ── YOLO 벤치마크 ────────────────────────────────────────────

def bench_yolo(weights: str, labels: List[Dict[str, Any]]) -> Result:
    """YOLO 모델 하나를 평가한다. Returns: (precision, recall, f1, avg_iou)"""
    from src.models.yolo_detector import YoloDetector

    cfg = {"model": weights, "device": "cpu", "conf": 0.5, "classes": [0]}
    detector = YoloDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious: List[float] = []

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        dets = detector.detect(frame)
        pred_boxes = [d.bbox for d in dets]
        gt_boxes = [parse_bbox(b) for b in item.get("persons", [])]

        tp, fp, fn, matched_ious = match_and_score(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return precision, recall, f1, mean_iou


# ── Face 벤치마크 ────────────────────────────────────────────

def bench_face(weights: str, labels: List[Dict[str, Any]]) -> Result:
    """FaceDetector 모델 하나를 평가한다. Returns: (precision, recall, f1, avg_iou)

    주의: 정답 person bbox를 crop 입력으로 주는 '조건부' 평가다.
    (person 검출이 완벽하다는 전제하의 얼굴 검출 성능)
    """
    from src.models.face_openvino import FaceDetector

    cfg = {"weights": weights, "device": "CPU", "conf_thresh": 0.5}
    detector = FaceDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious: List[float] = []

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

        tp, fp, fn, matched_ious = match_and_score(pred_faces, gt_faces)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return precision, recall, f1, mean_iou


# ── Eye 벤치마크 ─────────────────────────────────────────────

def bench_eye(weights: str, labels: List[Dict[str, Any]]) -> Result:
    """EyeDetector 모델 하나를 평가한다. Returns: (precision, recall, f1, avg_iou)

    주의: 눈은 객체가 작아 IoU가 구조적으로 낮게 나온다. IOU_THRESH=0.5는
    눈 검출에는 다소 가혹할 수 있다.
    """
    from src.models.eye_openvino import EyeDetector

    cfg = {"weights": weights, "device": "CPU"}
    detector = EyeDetector(cfg)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious: List[float] = []

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
                if iou_l >= IOU_THRESH:
                    total_tp += 1
                    all_ious.append(iou_l)
                else:
                    total_fp += 1
            else:
                total_fn += 1

            # right eye
            if track.right_eye is not None:
                iou_r = compute_iou(track.right_eye, gt_right)
                if iou_r >= IOU_THRESH:
                    total_tp += 1
                    all_ious.append(iou_r)
                else:
                    total_fp += 1
            else:
                total_fn += 1

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return precision, recall, f1, mean_iou


# ── 결과 출력 ────────────────────────────────────────────────

def print_result(title: str, result: Result) -> None:
    p, r, f, i = result
    print(f"\n=== {title} ===")
    print(f"  {'Precision':>10}{'Recall':>10}{'F1':>10}{'avg IoU':>10}")
    print(f"  {p:>10.4f}{r:>10.4f}{f:>10.4f}{i:>10.4f}")


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일이 없습니다: {LABELS_PATH}")
        logger.info("data/benchmark/detection/labels.json 을 먼저 준비하세요.")
        return

    labels = load_labels()
    logger.info(f"테스트 이미지 수: {len(labels)}")

    logger.info("YOLO 평가 중...")
    print_result("YOLO Person Detection", bench_yolo(YOLO_WEIGHTS, labels))

    logger.info("Face 평가 중...")
    print_result("Face Detection (head bbox 있는것만)", bench_face(FACE_WEIGHTS, labels))

    logger.info("Eye 평가 중...")
    print_result("Eye Detection", bench_eye(EYE_WEIGHTS, labels))


if __name__ == "__main__":
    main()