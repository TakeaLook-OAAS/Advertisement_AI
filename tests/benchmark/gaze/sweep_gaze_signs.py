"""
Gaze 좌표계 정합 스윕 진단

test_gaze 의 MAE 가 20° 처럼 어중간할 때, 원인이
  - GT ↔ 0002 출력의 좌표계 부호 불일치
  - left/right eye crop 좌우 바뀜
중 무엇인지 데이터로 찾는다.

방법:
  1) eye-swap 안 함 / 함  두 경우로 0002 를 추론 (추론 결과가 달라지므로 둘 다 필요)
  2) 각 경우에 대해 GT 에 (±x, ±y, ±z) 8가지 부호 조합 적용
  3) 16개 조합의 mean angular error 를 표로 출력 → 최소를 채택

설정: configs/test.yaml → gaze (data_dir, images_subdir, labels_file, device, weights.openvino)

사용법:
    python -m tests.benchmark.gaze.sweep_gaze_signs
"""
from __future__ import annotations

import json
import os
from itertools import product
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.utils.types import BBoxXYXY, HeadPose, Track

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["gaze"]


def angular_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pn, gn = np.linalg.norm(pred), np.linalg.norm(gt)
    if pn < 1e-8 or gn < 1e-8:
        return 180.0
    cos = np.clip(np.dot(pred, gt) / (pn * gn), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def parse_bbox(d: Dict[str, int]) -> BBoxXYXY:
    return BBoxXYXY(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


def main() -> None:
    cfg = load_config()
    data_dir = cfg["data_dir"]
    images_dir = os.path.join(data_dir, cfg["images_subdir"])
    labels_path = os.path.join(data_dir, cfg["labels_file"])
    device = cfg["device"]
    gaze_weights = cfg["weights"]["openvino"]  # sweep 은 OpenVINO 모델만 사용

    if not os.path.exists(labels_path):
        logger.error(f"라벨 없음: {labels_path}")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels: List[Dict[str, Any]] = json.load(f)
    logger.info(f"샘플 수: {len(labels)}")

    from src.models.gaze_openvino import GazeDetector
    detector = GazeDetector({"weights": gaze_weights, "device": device})

    # eye-swap 여부별로 0002 출력을 모아둔다. (pred, gt) 쌍 리스트.
    preds_by_swap: Dict[bool, List[np.ndarray]] = {False: [], True: []}
    gts: List[np.ndarray] = []

    for swap in (False, True):
        cache_gt = (swap is False)  # 첫 패스에서만 gt 수집
        for item in labels:
            img_path = os.path.join(images_dir, item["image"])
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            le = parse_bbox(item["left_eye"])
            re = parse_bbox(item["right_eye"])
            if swap:
                le, re = re, le

            hp = item["headpose"]
            track = Track(
                track_id=0, bbox=le,
                left_eye=le, right_eye=re,
                headpose=HeadPose(yaw=hp["yaw"], pitch=hp["pitch"], roll=hp["roll"]),
            )
            track = detector.detect(frame, track)
            if track.gaze is None:
                continue
            pred = np.array([track.gaze.x, track.gaze.y, track.gaze.z])
            preds_by_swap[swap].append(pred)

            if cache_gt:
                g = item["gaze"]
                gts.append(np.array([g["x"], g["y"], g["z"]]))

    gts_arr = np.array(gts)

    # ── 스윕 ──────────────────────────────────────────────────
    print("\n=== Sweep: eye_swap × GT sign flips ===")
    print(f"{'swap':<6}{'sx':>4}{'sy':>4}{'sz':>4}{'mean':>10}{'median':>10}")
    results: List[Tuple[float, str]] = []

    for swap in (False, True):
        preds = np.array(preds_by_swap[swap])
        n = min(len(preds), len(gts_arr))
        if n == 0:
            continue
        p = preds[:n]
        for sx, sy, sz in product((1, -1), repeat=3):
            flip = np.array([sx, sy, sz])
            errs = [angular_error(p[k], gts_arr[k] * flip) for k in range(n)]
            mean_e = float(np.mean(errs))
            med_e = float(np.median(errs))
            tag = f"swap={swap} flip=({sx:+d},{sy:+d},{sz:+d})"
            results.append((mean_e, tag))
            print(f"{str(swap):<6}{sx:>4}{sy:>4}{sz:>4}{mean_e:>10.2f}{med_e:>10.2f}")

    results.sort(key=lambda r: r[0])
    print("\n=== 최소 MAE 조합 ===")
    for mean_e, tag in results[:3]:
        print(f"  {mean_e:6.2f}°  {tag}")

    best = results[0]
    print(f"\n채택 → {best[1]}  (mean={best[0]:.2f}°)")
    print("해석:")
    print("  flip=(+,+,+) 가 최소면 → 부호 정합 OK, 20° 원인은 다른 곳(채널/headpose)")
    print("  특정 부호 조합에서 급감하면 → GT 좌표계를 그 부호로 변환해야 함")
    print("  swap=True 에서 최소면 → left/right eye 정의가 바뀌어 있었음")


if __name__ == "__main__":
    main()
