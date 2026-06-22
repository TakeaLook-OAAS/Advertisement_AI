"""
영상에서 프레임을 추출하여 data/benchmark/detection/images/ 에 저장한다.

설정: configs/test.yaml → extract_frames (output_dir, samples_dir)
      --video / --all / --fps 는 CLI 인자로도 덮어쓸 수 있다.

사용법:
    python -m tests.benchmark.detection.extract_frames
    python -m tests.benchmark.detection.extract_frames --video data/samples/test1.mp4 --fps 10
    python -m tests.benchmark.detection.extract_frames --all --fps 10
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import cv2
import yaml

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["extract_frames"]


def extract(video_path: str, fps: int, output_dir: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return 0

    interval = max(1, fps)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            filename = f"{video_name}_f{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"  {video_path} → {saved}장 저장")
    return saved


def main() -> None:
    cfg = load_config()
    output_dir = cfg["output_dir"]
    samples_dir = cfg["samples_dir"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="단일 영상 경로")
    parser.add_argument("--all", action="store_true", help=f"{samples_dir}/ 전체 영상 처리")
    parser.add_argument("--fps", type=int, default=1, help="N프레임마다 1장 추출 (기본 1)")
    args = parser.parse_args()

    if args.video:
        videos = [args.video]
    else:
        # --all 이거나 기본값이면 samples_dir 전체 처리
        videos = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

    total = 0
    for v in videos:
        total += extract(v, args.fps, output_dir)

    print(f"\n총 {total}장 → {output_dir}")
    print(f"(매 {args.fps}프레임마다 1장 추출)")
    print("\n다음 단계:")
    print("  1. LabelImg 실행")
    print(f"  2. Open Dir → {output_dir}")
    print("  3. Change Save Dir → data/benchmark/detection/annotations")
    print("  4. 포맷을 PascalVOC(XML)로 설정")
    print("  5. 클래스: person, face 로 bbox 그리기")
    print("  6. 완료 후: python -m tests.benchmark.annotations_to_json")


if __name__ == "__main__":
    main()
