"""
WFLW 학습용 눈 bbox 라벨 생성.

EyeNet(facial-landmarks-35-adas-0002 대체) 학습 라벨:
  - face_bbox  : WFLW 제공 detection rect(x_min_rect,y_min_rect,x_max_rect,y_max_rect)를
                 이미지 경계로 clip한 값
  - eyes.left  : 눈 컨투어 8점(60-67, 이미지 기준 왼쪽 눈)을 감싸는 tight bbox에
                 패딩을 준 뒤 face_bbox 기준으로 정규화([0,1])
  - eyes.right : 눈 컨투어 8점(68-75, 이미지 기준 오른쪽 눈), 동일 방식

WFLW annotation line 포맷 (207 컬럼, list_98pt_rect_attr_{train,test}.txt):
  x0 y0 ... x97 y97 (98 landmarks)
  x_min_rect y_min_rect x_max_rect y_max_rect (detection rect)
  pose expression illumination makeup occlusion blur (속성, 미사용)
  image_name

사용법 (프로젝트 루트에서):
    python -m scripts.eye.eye_generate_labels                # train split
    python -m scripts.eye.eye_generate_labels --split test    # test split

출력: data/benchmark/eye/labels_train.json (또는 labels_test.json)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2
import yaml

CONFIG_PATH = "configs/train.yaml"

NUM_LANDMARKS = 98
LEFT_EYE_IDX = range(60, 68)   # 이미지 기준 왼쪽 눈 컨투어 8점
RIGHT_EYE_IDX = range(68, 76)  # 이미지 기준 오른쪽 눈 컨투어 8점

EYE_PAD_RATIO = 0.25
MIN_FACE_SIZE = 30


# ── 파싱 / 유틸 함수 ──────────────────────────────────────────

def _parse_line(line: str) -> Optional[dict]:
    cols = line.split()
    if len(cols) < 2 * NUM_LANDMARKS + 4 + 6 + 1:
        return None

    pts = [(float(cols[2 * i]), float(cols[2 * i + 1])) for i in range(NUM_LANDMARKS)]
    rect = tuple(float(v) for v in cols[196:200])
    image_name = cols[206]

    return {"points": pts, "rect": rect, "image_name": image_name}


def _tight_bbox(pts: List[Tuple[float, float]], pad_ratio: float) -> Tuple[float, float, float, float]:
    """랜드마크 점들을 감싸는 tight bbox에 비율 패딩을 준다."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def _normalize_box(box: Tuple[float, float, float, float], face_bbox: Tuple[float, float, float, float]) -> dict:
    fx1, fy1, fx2, fy2 = face_bbox
    fw, fh = fx2 - fx1, fy2 - fy1
    x1, y1, x2, y2 = box
    return {
        "x1": min(1.0, max(0.0, (x1 - fx1) / fw)),
        "y1": min(1.0, max(0.0, (y1 - fy1) / fh)),
        "x2": min(1.0, max(0.0, (x2 - fx1) / fw)),
        "y2": min(1.0, max(0.0, (y2 - fy1) / fh)),
    }


def _process_file(
    ann_path: str,
    images_dir: str,
    pad_ratio: float,
    min_face_size: float,
) -> Tuple[list, int]:
    with open(ann_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    results, skipped = [], 0

    for line in lines:
        parsed = _parse_line(line)
        if parsed is None:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, parsed["image_name"])
        frame = cv2.imread(img_path)
        if frame is None:
            skipped += 1
            continue
        h, w = frame.shape[:2]

        fx1, fy1, fx2, fy2 = parsed["rect"]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(w), fx2), min(float(h), fy2)
        if fx2 - fx1 < min_face_size or fy2 - fy1 < min_face_size:
            skipped += 1
            continue
        face_bbox = (fx1, fy1, fx2, fy2)

        pts = parsed["points"]
        left_box = _tight_bbox([pts[i] for i in LEFT_EYE_IDX], pad_ratio)
        right_box = _tight_bbox([pts[i] for i in RIGHT_EYE_IDX], pad_ratio)

        results.append({
            "image": parsed["image_name"],
            "face_bbox": {"x1": int(fx1), "y1": int(fy1), "x2": int(fx2), "y2": int(fy2)},
            "eyes": {
                "left": _normalize_box(left_box, face_bbox),
                "right": _normalize_box(right_box, face_bbox),
            },
        })

    return results, skipped


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["generate"]["eye"]

    parser = argparse.ArgumentParser(description="WFLW 기반 눈 bbox 라벨 생성")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--images-dir", default=cfg.get("wflw_images_dir", "data/train/WFLW_images"))
    parser.add_argument(
        "--annotations-dir",
        default=cfg.get("wflw_annotations_dir", "data/train/WFLW_annotations/list_98pt_rect_attr_train_test"),
    )
    parser.add_argument("--output-dir", default=cfg.get("output_dir", "data/benchmark/eye"))
    parser.add_argument("--eye-pad-ratio", type=float, default=cfg.get("eye_pad_ratio", 0.25))
    parser.add_argument("--min-face-size", type=float, default=cfg.get("min_face_size", 30))
    args = parser.parse_args()

    global EYE_PAD_RATIO, MIN_FACE_SIZE
    EYE_PAD_RATIO = args.eye_pad_ratio
    MIN_FACE_SIZE = args.min_face_size

    ann_path = os.path.join(args.annotations_dir, f"list_98pt_rect_attr_{args.split}.txt")
    out = os.path.join(args.output_dir, f"labels_{args.split}.json")

    print(f"[{args.split}] 처리 중... ({ann_path})")
    labels, skipped = _process_file(ann_path, args.images_dir, EYE_PAD_RATIO, MIN_FACE_SIZE)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out}")
    print(f"총 샘플: {len(labels)} | 제외: {skipped}")


if __name__ == "__main__":
    main()
