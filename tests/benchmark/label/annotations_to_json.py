"""
LabelImg에서 저장한 YOLO .txt 파일들을 labels.json으로 변환한다.
이미지와 .txt 파일이 같은 폴더에 있다고 가정한다.

설정: configs/test.yaml → label.annotations_to_json (images_dir, output_path)

사용법:
    python -m tests.benchmark.label.annotations_to_json
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import cv2
import yaml

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["label"]["annotations_to_json"]


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


def match_person_id(face: Dict[str, int], persons: List[Dict[str, int]]) -> int:
    """face bbox 중심점을 포함하는 person bbox를 찾아 그 id를 반환한다.
    person들이 겹쳐 있으면 넓이가 가장 작은(가장 타이트한) person을 우선한다.
    포함하는 person이 없으면 IoU가 가장 큰 person의 id를, 그마저도 없으면 -1을 반환한다.
    """
    fcx = (face["x1"] + face["x2"]) / 2
    fcy = (face["y1"] + face["y2"]) / 2

    containing = [
        p for p in persons
        if p["x1"] <= fcx <= p["x2"] and p["y1"] <= fcy <= p["y2"]
    ]
    if containing:
        smallest = min(containing, key=lambda p: (p["x2"] - p["x1"]) * (p["y2"] - p["y1"]))
        return smallest["id"]

    best_iou, best_id = 0.0, -1
    for p in persons:
        ix1, iy1 = max(face["x1"], p["x1"]), max(face["y1"], p["y1"])
        ix2, iy2 = min(face["x2"], p["x2"]), min(face["y2"], p["y2"])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_f = (face["x2"] - face["x1"]) * (face["y2"] - face["y1"])
        area_p = (p["x2"] - p["x1"]) * (p["y2"] - p["y1"])
        union = area_f + area_p - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_id = iou, p["id"]
    return best_id


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

            if class_name == "person_looking":
                bbox["id"] = len(persons)
                bbox["looking"] = True
                bbox["gender"] = ""
                bbox["age_group"] = ""
                persons.append(bbox)
            elif class_name == "person_not_looking":
                bbox["id"] = len(persons)
                bbox["looking"] = False
                bbox["gender"] = ""
                bbox["age_group"] = ""
                persons.append(bbox)
            elif class_name == "face":
                faces.append(bbox)

    for face in faces:
        face["id"] = match_person_id(face, persons)

    return {"persons": persons, "faces": faces}


def main() -> None:
    cfg = load_config()
    images_dir = cfg["images_dir"]
    output_path = cfg["output_path"]

    classes = load_classes(images_dir)
    if not classes:
        print(f"[ERROR] classes.txt 없음: {images_dir}")
        return

    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    )

    labels: List[Dict[str, Any]] = []
    skipped = 0

    for img_file in image_files:
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(images_dir, txt_file)

        if not os.path.exists(txt_path):
            skipped += 1
            continue

        img = cv2.imread(os.path.join(images_dir, img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        parsed = parse_txt(txt_path, img_w, img_h, classes)
        entry = {"image": img_file, **parsed}
        labels.append(entry)
        print(f"  {img_file} → persons: {len(parsed['persons'])}, faces: {len(parsed['faces'])}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(labels)}장 변환 → {output_path}")
    if skipped:
        print(f"(라벨 없는 이미지 {skipped}장 제외)")
    print("이제 실행 가능:")
    print("  python -m tests.benchmark.detection.test_detection")
    print("  python -m tests.benchmark.attention.test_attention")
    print("  python -m tests.benchmark.attr.test_attr  (gender/age_group 채운 후)")


if __name__ == "__main__":
    main()
