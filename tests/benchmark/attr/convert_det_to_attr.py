"""
detection labels.json → attr labels.json 골격 변환기

detection 라벨은 이미지당 persons[], faces[] 두 리스트가 분리돼 있다.
attr 라벨은 '사람 1명 = 1개 항목' 구조이고 person bbox + face bbox 쌍이 필요하다.

이 스크립트는:
  1. 각 face 의 중심점이 어느 person 박스 안에 들어가는지로 person↔face 를 매칭
  2. 매칭된(=정면 얼굴 있는) 사람만 attr 항목으로 펼침(flatten)
  3. gender / age_group 은 빈 칸("")으로 남겨 사람이 채우게 함

사용법:
    python -m tests.benchmark.attr.convert_det_to_attr

입력:  data/benchmark/detection/labels.json   (+ images/)
출력:  data/benchmark/attr/labels.json        (+ images/ 복사)

이후 사람이 할 일:
    attr/labels.json 의 각 항목에서 "gender", "age_group" 을 채운다.
    값은 반드시 모델 출력 문자열과 정확히 일치해야 한다.
      gender    : Gender enum 의 .value
      age_group : AgeGroup enum 의 .value
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


DET_DIR = "data/benchmark/detection"
DET_LABELS = os.path.join(DET_DIR, "labels.json")
DET_IMAGES = os.path.join(DET_DIR, "images")

ATTR_DIR = "data/benchmark/attr"
ATTR_LABELS = os.path.join(ATTR_DIR, "labels.json")
ATTR_IMAGES = os.path.join(ATTR_DIR, "images")


def center_of(box: Dict[str, int]) -> Tuple[float, float]:
    return (box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0


def point_in_box(px: float, py: float, box: Dict[str, int]) -> bool:
    return box["x1"] <= px <= box["x2"] and box["y1"] <= py <= box["y2"]


def box_area(box: Dict[str, int]) -> float:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def match_face_to_person(
    face: Dict[str, int], persons: List[Dict[str, int]]
) -> Optional[int]:
    """
    face 중심이 들어가는 person 박스의 인덱스를 반환.
    여러 person 에 걸치면 면적이 가장 작은(=가장 꼭 맞는) person 을 택한다.
    어디에도 안 들어가면 None.
    """
    fx, fy = center_of(face)
    candidates = [
        (box_area(p), i) for i, p in enumerate(persons) if point_in_box(fx, fy, p)
    ]
    if not candidates:
        return None
    candidates.sort()  # 면적 작은 순
    return candidates[0][1]


def main() -> None:
    if not os.path.exists(DET_LABELS):
        logger.error(f"detection 라벨이 없습니다: {DET_LABELS}")
        return

    with open(DET_LABELS, "r", encoding="utf-8") as f:
        det_labels = json.load(f)

    os.makedirs(ATTR_IMAGES, exist_ok=True)

    attr_items: List[Dict[str, Any]] = []
    n_persons = 0
    n_faces = 0
    n_matched = 0
    n_face_no_person = 0
    used_images = set()

    for item in det_labels:
        image = item["image"]
        persons = item.get("persons", [])
        faces = item.get("faces", [])
        n_persons += len(persons)
        n_faces += len(faces)

        # 한 person 에 여러 face 가 매칭되는 중복을 막기 위해 사용된 person 추적
        person_taken = set()

        for face in faces:
            pi = match_face_to_person(face, persons)
            if pi is None:
                n_face_no_person += 1
                continue
            if pi in person_taken:
                # 이미 다른 face 가 차지한 person → 더 작은 박스 우선이지만
                # 여기선 단순히 첫 매칭 유지하고 건너뜀
                logger.warning(
                    f"{image}: person #{pi} 에 face 가 2개 이상 매칭됨, 뒤 face 건너뜀"
                )
                continue
            person_taken.add(pi)
            n_matched += 1

            attr_items.append({
                "image": image,
                "bbox": persons[pi],   # person 전신
                "face": face,          # 짝지어진 얼굴
                "gender": "",          # ← 사람이 채울 칸
                "age_group": "",       # ← 사람이 채울 칸
            })
            used_images.add(image)

    # 사용된 이미지만 복사
    n_copied = 0
    for image in used_images:
        src = os.path.join(DET_IMAGES, image)
        dst = os.path.join(ATTR_IMAGES, image)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
        else:
            logger.warning(f"원본 이미지 없음: {src}")

    with open(ATTR_LABELS, "w", encoding="utf-8") as f:
        json.dump(attr_items, f, ensure_ascii=False, indent=2)

    logger.info("── 변환 요약 ──────────────────")
    logger.info(f"  detection person 총합 : {n_persons}")
    logger.info(f"  detection face 총합   : {n_faces}")
    logger.info(f"  person 매칭 성공      : {n_matched}  ← attr 항목 수")
    logger.info(f"  person 못 찾은 face   : {n_face_no_person}")
    logger.info(f"  복사된 이미지         : {n_copied}")
    logger.info(f"저장: {ATTR_LABELS}")
    logger.info("")
    logger.info('이제 attr/labels.json 의 각 항목에서 "gender", "age_group" 을 채우세요.')
    logger.info("값은 Gender / AgeGroup enum 의 .value 와 정확히 일치해야 합니다.")


if __name__ == "__main__":
    main()