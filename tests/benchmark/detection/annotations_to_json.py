"""
LabelImg에서 저장한 YOLO .txt 파일들을 labels.json으로 변환한다.
이미지와 .txt 파일이 같은 폴더에 있다고 가정한다.

사용법:
    python -m tests.benchmark.detection.annotations_to_json
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import cv2

IMAGES_DIR = "data/benchmark/detection/images"
OUTPUT_PATH = "data/benchmark/detection/labels.json"


def load_classes(images_dir: str) -> Dict[int, str]:
    classes_path = os.path.join(images_dir, "classes.txt")
    if not os.path.exists(classes_path):
        return {}
    with open(classes_path, encoding="utf-8") as f:
        return {i: line.strip() for i, line in enumerate(f) if line.strip()}


def yolo_to_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Dict[str, int]:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def parse_txt(txt_path: str, img_w: int, img_h: int, classes: Dict[int, str]) -> Dict[str, List]:
    persons = []
    faces = []

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bbox = yolo_to_bbox(cx, cy, w, h, img_w, img_h)
            class_name = classes.get(class_id, "")

            if class_name == "person":
                bbox["id"] = len(persons)
                persons.append(bbox)
            elif class_name == "face":
                bbox["id"] = len(faces)
                faces.append(bbox)

    return {"persons": persons, "faces": faces}


def main() -> None:
    classes = load_classes(IMAGES_DIR)
    if not classes:
        print(f"[ERROR] classes.txt 없음: {IMAGES_DIR}")
        return

    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in image_exts
    )

    labels: List[Dict[str, Any]] = []
    skipped = 0

    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(IMAGES_DIR, txt_file)

        if not os.path.exists(txt_path):
            skipped += 1
            continue

        img = cv2.imread(os.path.join(IMAGES_DIR, img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        parsed = parse_txt(txt_path, img_w, img_h, classes)
        entry = {"image": img_file, **parsed}
        labels.append(entry)
        print(f"  {img_file} → persons: {len(parsed['persons'])}, faces: {len(parsed['faces'])}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(labels)}장 변환 → {OUTPUT_PATH}")
    if skipped:
        print(f"(라벨 없는 이미지 {skipped}장 제외)")
    print("이제 실행 가능: python -m tests.benchmark.test_detection")


if __name__ == "__main__":
    main()
