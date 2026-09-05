"""
labels.json의 persons[].gender/age_group을 일괄로 더미 값으로 채운다.
실제 성별/나이 라벨링 전에 MiVOLO attr 벤치마크(test_attr.py)를 돌려보기 위한 용도.

주의: age_group 값은 AgeGroup enum의 멤버 이름(age_20_29)이 아니라 실제 value
문자열("20-29")이어야 한다 — test_attr.py가 track.attr.age_group.value와 GT 문자열을
그대로 비교하기 때문에, 값이 안 맞으면 정확도가 항상 0이 된다.

사용법:
    python -m tests.benchmark.attr.fill_dummy_attr
    python -m tests.benchmark.attr.fill_dummy_attr --gender female --age-group 30-39
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import yaml

from src.utils.types import AgeGroup, Gender

CONFIG_PATH = "configs/test.yaml"

VALID_GENDERS = {g.value for g in Gender}
VALID_AGE_GROUPS = {a.value for a in AgeGroup}


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["attr"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", default="male", choices=sorted(VALID_GENDERS))
    parser.add_argument("--age-group", default="20-29", choices=sorted(VALID_AGE_GROUPS))
    args = parser.parse_args()

    cfg = load_config()
    labels_path = os.path.join(cfg["labels_dir"], cfg["labels_file"])

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    count = 0
    for item in labels:
        for p in item.get("persons", []):
            p["gender"] = args.gender
            p["age_group"] = args.age_group
            count += 1

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(
        f"{count}명의 gender/age_group을 gender={args.gender!r}, age_group={args.age_group!r}로 채움 "
        f"({len(labels)}개 이미지, {labels_path})"
    )


if __name__ == "__main__":
    main()
