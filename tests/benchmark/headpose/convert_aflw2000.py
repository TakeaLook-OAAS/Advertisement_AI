"""
AFLW2000 → labels.json 변환 스크립트

AFLW2000 .mat 파일에서 yaw/pitch/roll (degree) 과 face bbox를 추출하여
test_headpose.py 가 읽는 labels.json 포맷으로 저장한다.

필터: |yaw| > 99° 샘플 제외 (6DRepNet 논문 평가 기준과 동일)

사용법:
    python -m tests.benchmark.headpose.convert_aflw2000
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.io
from PIL import Image

AFLW2000_DIR = "data/benchmark/headpose/AFLW2000"
OUTPUT_PATH = "data/benchmark/headpose/labels.json"

YAW_LIMIT = 99.0
PADDING = 0.20  # 6DRepNet 논문 평가 기준과 동일한 20% padding


def pt2d_to_bbox(pt2d: np.ndarray, img_w: int, img_h: int) -> dict:
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

    x_min -= 2 * PADDING * w
    y_min -= 2 * PADDING * h
    x_max += 2 * PADDING * w
    y_max += 2 * PADDING * h

    return {
        "x1": int(max(0, x_min)),
        "y1": int(max(0, y_min)),
        "x2": int(min(img_w, x_max)),
        "y2": int(min(img_h, y_max)),
    }


def main() -> None:
    mat_files = sorted(
        f for f in os.listdir(AFLW2000_DIR) if f.endswith(".mat")
    )

    labels = []
    skipped = 0

    for mat_file in mat_files:
        img_file = mat_file.replace(".mat", ".jpg")
        img_path = os.path.join(AFLW2000_DIR, img_file)
        mat_path = os.path.join(AFLW2000_DIR, mat_file)

        if not os.path.exists(img_path):
            continue

        mat = scipy.io.loadmat(mat_path)
        pose = mat["Pose_Para"][0]  # [pitch, yaw, roll, tx, ty, tz, ...]

        pitch = float(np.degrees(pose[0]))
        yaw = float(np.degrees(pose[1]))
        roll = float(np.degrees(pose[2]))

        if abs(yaw) > YAW_LIMIT:
            skipped += 1
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        pt2d = mat["pt2d"]  # (2, N) 2D landmarks
        face = pt2d_to_bbox(pt2d, img_w, img_h)

        labels.append({
            "image": img_file,
            "face": face,
            "yaw": -yaw,   # headpose_6drepnet.py 가 yaw=-yaw 를 적용하므로 GT도 동일하게 부호 반전
            "pitch": pitch,
            "roll": roll,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"  총 샘플: {len(labels)}")
    print(f"  제외 (|yaw| > {YAW_LIMIT}°): {skipped}")


if __name__ == "__main__":
    main()
