"""
YOLO person 검출 결과 시각화: 정답/예측 bbox를 이미지에 그려 저장한다.
어떤 사람을 왜 놓쳤는지 눈으로 확인하는 용도.

사용법:
    python -m tests.benchmark.detection.visualize_bbox

색 범례:
    초록  = 정답 person (GT)
    빨강  = YOLO가 검출한 person (pred)
    노랑 글자 = 매칭된 쌍의 IoU 값 / 요약

저장 위치: data/benchmark/detection/bbox_check/
"""
from __future__ import annotations

import os
from typing import List

import cv2
from loguru import logger

from src.utils.types import BBoxXYXY, Track

from tests.benchmark.detection.test_detection import (
    IMAGES_DIR,
    IOU_THRESH,
    YOLO_WEIGHTS,
    compute_iou,
    load_labels,
    parse_bbox,
)


OUT_DIR = "data/benchmark/detection/bbox_check"

GREEN  = (0, 200, 0)
RED    = (0, 0, 255)
YELLOW = (0, 220, 220)


def draw_box(img, box: BBoxXYXY, color, label: str = "") -> None:
    cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), color, 2)
    if label:
        cv2.putText(
            img, label, (box.x1, max(box.y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def match_preds_to_gts(preds: List[BBoxXYXY], gts: List[BBoxXYXY]):
    """greedy IoU 매칭. Returns: (matched: list[(pi, gi, iou)], fp_idx, fn_idx)"""
    matched = []
    matched_gt = set()
    matched_pred = set()

    for pi, pred in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= IOU_THRESH and best_gi >= 0:
            matched.append((pi, best_gi, best_iou))
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    fp_idx = [pi for pi in range(len(preds)) if pi not in matched_pred]
    fn_idx = [gi for gi in range(len(gts)) if gi not in matched_gt]
    return matched, fp_idx, fn_idx


def main() -> None:
    from src.models.yolo_detector import YoloDetector

    os.makedirs(OUT_DIR, exist_ok=True)
    labels = load_labels()
    logger.info(f"시각화 대상 이미지 수: {len(labels)}")

    cfg = {"model": YOLO_WEIGHTS, "device": "cpu", "conf": 0.5, "classes": [0]}
    detector = YoloDetector(cfg)

    n_tp, n_fp, n_fn = 0, 0, 0

    for item in labels:
        img_path = os.path.join(IMAGES_DIR, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue

        gt_boxes = [parse_bbox(b) for b in item.get("persons", [])]
        pred_boxes = [d.bbox for d in detector.detect(frame)]

        matched, fp_idx, fn_idx = match_preds_to_gts(pred_boxes, gt_boxes)
        n_tp += len(matched)
        n_fp += len(fp_idx)
        n_fn += len(fn_idx)

        # 정답: 매칭되면 IoU 표기, FN이면 'MISS'
        fn_set = set(fn_idx)
        gi_to_iou = {gi: iou for (_, gi, iou) in matched}
        for gi, gt in enumerate(gt_boxes):
            pid = item.get("persons", [])[gi].get("id", gi)
            if gi in fn_set:
                draw_box(frame, gt, GREEN, label=f"P{pid}-MISS")
            else:
                draw_box(frame, gt, GREEN, label=f"P{pid} IoU {gi_to_iou[gi]:.2f}")

        # 예측: FP면 'FP'로 표시
        fp_set = set(fp_idx)
        for pi, pred in enumerate(pred_boxes):
            draw_box(frame, pred, RED, label="FP" if pi in fp_set else "pred")

        cap = f"TP={len(matched)} FP={len(fp_idx)} FN={len(fn_idx)}"
        cv2.putText(frame, cap, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, item["image"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, frame)

    logger.info(f"저장 완료: {OUT_DIR}")
    logger.info(f"전체 합계  TP={n_tp}  FP={n_fp}  FN={n_fn}")


if __name__ == "__main__":
    main()
