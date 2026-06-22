"""
AFLW2000 → labels.json 변환 스크립트

AFLW2000 .mat 파일에서 yaw/pitch/roll (degree) 과 face bbox를 추출하여
test_headpose.py 가 읽는 labels.json 포맷으로 저장한다.

필터: |yaw| > 99° 샘플 제외 (6DRepNet 논문 평가 기준과 동일)

설정: configs/test.yaml → convert.headpose

사용법:
    python -m tests.benchmark.headpose.convert_aflw2000
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import scipy.io
import yaml
from PIL import Image

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["convert"]["headpose"]


def pt2d_to_bbox(pt2d: np.ndarray, img_w: int, img_h: int, padding: float) -> dict:
    """
    2D 랜드마크에서 face bbox를 구한다. (6DRepNet 원래 평가 코드와 동일)
    pt2d: (2, N) - x<0인 점은 invisible 마커이므로 제외한다.
    """
    xs = pt2d[0]
    ys = pt2d[1]
    valid = xs >= 0  # x<0 은 invisible landmark
    xs = xs[valid]
    ys = ys[valid]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    w = x_max - x_min
    h = y_max - y_min

    x_min -= 2 * padding * w
    y_min -= 2 * padding * h
    x_max += 2 * padding * w
    y_max += 2 * padding * h

    return {
        "x1": int(max(0, x_min)),
        "y1": int(max(0, y_min)),
        "x2": int(min(img_w, x_max)),
        "y2": int(min(img_h, y_max)),
    }


def main() -> None:
    cfg = load_config()
    source_dir = cfg["source_dir"]
    output_path = cfg["output_path"]
    yaw_limit = cfg["yaw_limit"]
    padding = cfg["padding"]

    mat_files = sorted(
        f for f in os.listdir(source_dir) if f.endswith(".mat")
    )

    labels = []
    skipped = 0

    for mat_file in mat_files:
        img_file = mat_file.replace(".mat", ".jpg")
        img_path = os.path.join(source_dir, img_file)
        mat_path = os.path.join(source_dir, mat_file)

        if not os.path.exists(img_path):
            continue

        mat = scipy.io.loadmat(mat_path)
        pose = mat["Pose_Para"][0]  # [pitch, yaw, roll, tx, ty, tz, ...]

        pitch = float(np.degrees(pose[0]))
        yaw = float(np.degrees(pose[1]))
        roll = float(np.degrees(pose[2]))

        if abs(yaw) > yaw_limit:
            skipped += 1
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        pt2d = mat["pt2d"]  # (2, N) 2D landmarks
        face = pt2d_to_bbox(pt2d, img_w, img_h, padding)

        labels.append({
            "image": img_file,
            "face": face,
            "yaw": -yaw,   # headpose_6drepnet.py 가 yaw=-yaw 를 적용하므로 GT도 동일하게 부호 반전
            "pitch": pitch,
            "roll": roll,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"저장 완료: {output_path}")
    print(f"  총 샘플: {len(labels)}")
    print(f"  제외 (|yaw| > {yaw_limit}°): {skipped}")


if __name__ == "__main__":
    main()
