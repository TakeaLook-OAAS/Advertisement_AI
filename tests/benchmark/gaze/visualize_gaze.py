"""
Gaze 비교 시각화: GT(정답) vs 0002 예측

각 이미지에 두 화살표를 겹쳐 그린다:
  - 초록 화살표 : GT gaze (labels.json, 부호 정합 적용됨)
  - 빨강 화살표 : gaze-estimation-adas-0002 예측

두 화살표가 가까울수록 정확. 좌상단에 angular error(deg) 표시.
오차가 큰 순으로도 따로 저장해 실패 케이스를 보기 쉽게 한다.

헤드리스(Docker) 환경: 창을 띄우지 않고 파일로 저장.

설정: configs/test.yaml → gaze, visualize.gaze

사용법:
    python -m tests.benchmark.gaze.visualize_gaze
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.utils.types import BBoxXYXY, HeadPose, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["gaze"], cfg["visualize"]["gaze"]


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


def angular_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pn, gn = np.linalg.norm(pred), np.linalg.norm(gt)
    if pn < 1e-8 or gn < 1e-8:
        return 180.0
    cos = np.clip(np.dot(pred, gt) / (pn * gn), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def eye_center(bbox: Dict[str, int]) -> Tuple[int, int]:
    return ((bbox["x1"] + bbox["x2"]) // 2, (bbox["y1"] + bbox["y2"]) // 2)


def draw_gaze_arrow(
    frame: np.ndarray,
    origin: Tuple[int, int],
    vec: np.ndarray,
    color,
    arrow_scale: float,
    thickness=2,
) -> None:
    """3D gaze 단위벡터를 2D 화살표로 근사 투영해 그린다.
    길이는 근사이므로 방향만 의미 있음. (x→오른쪽, y는 0002 기준 위쪽(+)이라
    이미지 y축(아래+)과 반대 → 화면 그릴 때 dy 부호를 뒤집는다.)"""
    gx, gy, gz = vec
    dx = arrow_scale * gx / max(abs(gz), 1e-3)
    dy = -arrow_scale * gy / max(abs(gz), 1e-3)  # 0002 y(위+) → 이미지 y(아래+)
    end = (int(origin[0] + dx), int(origin[1] + dy))
    cv2.arrowedLine(frame, origin, end, color, thickness, tipLength=0.25)


def render(
    item: Dict[str, Any],
    pred: np.ndarray,
    err: float,
    images_dir: str,
    arrow_scale: float,
) -> np.ndarray | None:
    img_path = os.path.join(images_dir, item["image"])
    frame = cv2.imread(img_path)
    if frame is None:
        return None

    le, re = item["left_eye"], item["right_eye"]
    # gaze 원점: 양 눈 중심의 중간
    lc, rc = eye_center(le), eye_center(re)
    origin = ((lc[0] + rc[0]) // 2, (lc[1] + rc[1]) // 2)

    g = item["gaze"]
    gt_vec = np.array([g["x"], g["y"], g["z"]])

    # 화살표: 초록=GT, 빨강=예측
    draw_gaze_arrow(frame, origin, gt_vec, (0, 255, 0), arrow_scale, 2)
    draw_gaze_arrow(frame, origin, pred, (0, 0, 255), arrow_scale, 2)

    # 텍스트
    lines = [
        item["image"],
        f"err={err:.1f} deg",
        f"GT  =({gt_vec[0]:+.2f},{gt_vec[1]:+.2f},{gt_vec[2]:+.2f})",
        f"pred=({pred[0]:+.2f},{pred[1]:+.2f},{pred[2]:+.2f})",
    ]
    for k, txt in enumerate(lines):
        y = 20 + k * 20
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 범례
    cv2.putText(frame, "green=GT  red=pred", (8, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    gaze_cfg, vis_cfg = load_config()
    data_dir = gaze_cfg["data_dir"]
    images_dir = os.path.join(data_dir, gaze_cfg["images_subdir"])
    labels_path = os.path.join(data_dir, gaze_cfg["labels_file"])
    device = gaze_cfg["device"]
    gaze_weights = vis_cfg["weights"]   # 시각화는 OpenVINO 모델만 사용
    save_dir = vis_cfg["save_dir"]
    num_to_save = vis_cfg["num_to_save"]   # 순서대로 저장할 장 수
    num_worst = vis_cfg["num_worst"]       # 오차 큰 순으로 따로 저장할 장 수
    arrow_scale = vis_cfg["arrow_scale"]   # 화살표 픽셀 길이 스케일

    if not os.path.exists(labels_path):
        logger.error(f"라벨 없음: {labels_path}")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels: List[Dict[str, Any]] = json.load(f)
    logger.info(f"샘플 수: {len(labels)}")

    from src.models.gaze_openvino import GazeDetector
    detector = GazeDetector({"weights": gaze_weights, "device": device})

    os.makedirs(save_dir, exist_ok=True)

    # ── 전체 추론 + error 계산 ───────────────────────────────
    records: List[Tuple[float, Dict[str, Any], np.ndarray]] = []
    for item in labels:
        img_path = os.path.join(images_dir, item["image"])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        track = Track(
            track_id=0,
            bbox=parse_bbox(item["left_eye"]),
            left_eye=parse_bbox(item["left_eye"]),
            right_eye=parse_bbox(item["right_eye"]),
            headpose=HeadPose(**item["headpose"]),
        )
        track = detector.detect(frame, track)
        if track.gaze is None:
            continue
        pred = np.array([track.gaze.x, track.gaze.y, track.gaze.z])
        g = item["gaze"]
        gt = np.array([g["x"], g["y"], g["z"]])
        err = angular_error(pred, gt)
        records.append((err, item, pred))

    if not records:
        logger.error("유효 샘플 없음")
        return

    mean_e = float(np.mean([r[0] for r in records]))
    logger.info(f"mean angular error = {mean_e:.2f} deg  (n={len(records)})")

    # ── 순서대로 저장 ────────────────────────────────────────
    seq_dir = os.path.join(save_dir, "sequential")
    os.makedirs(seq_dir, exist_ok=True)
    for idx, (err, item, pred) in enumerate(records[:num_to_save]):
        out = render(item, pred, err, images_dir, arrow_scale)
        if out is not None:
            cv2.imwrite(os.path.join(seq_dir, f"cmp_{idx:04d}.jpg"), out)

    # ── 오차 큰 순(worst) 저장 ───────────────────────────────
    worst_dir = os.path.join(save_dir, "worst")
    os.makedirs(worst_dir, exist_ok=True)
    for rank, (err, item, pred) in enumerate(
        sorted(records, key=lambda r: r[0], reverse=True)[:num_worst]
    ):
        out = render(item, pred, err, images_dir, arrow_scale)
        if out is not None:
            cv2.imwrite(
                os.path.join(worst_dir, f"worst_{rank:02d}_{err:.0f}deg.jpg"), out
            )

    print(f"\n저장 완료:")
    print(f"  순서대로 {min(num_to_save, len(records))}장 → {seq_dir}")
    print(f"  오차 큰 순 {min(num_worst, len(records))}장 → {worst_dir}")
    print(f"  초록=GT, 빨강=예측. 두 화살표가 가까울수록 정확.")


if __name__ == "__main__":
    main()
