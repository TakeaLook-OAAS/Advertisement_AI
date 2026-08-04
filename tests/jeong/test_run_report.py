"""
main.py 파이프라인을 실행하고, 끝나면 세그먼트 JSON들을 읽어
설정값 / 트랙 수 / 시선 발생 트랙 / 시선 비율 / 총 시선시간을 표로 출력한다.
(총 처리 시간은 파이프라인 실행 중 runner.py가 "총 처리 시간: N초" 로그로 출력한다)

사용법:
  python tests/jeong/test_run_report.py
"""

from __future__ import annotations

import glob
import json
from typing import Any, Dict, List

from src.main import main as run_pipeline
from src.utils.config import load_config


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_segments(json_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(f"{json_dir.rstrip('/')}/segment_*.json"))
    return [_load_json(p) for p in paths]


def _print_config_summary(cfg: Dict[str, Any]) -> None:
    yolo_cfg = cfg.get("models", {}).get("yolo", {})
    presence_cfg = cfg.get("logic", {}).get("presence", {})
    hysteresis_cfg = cfg.get("logic", {}).get("attention", {}).get("hysteresis", {})

    print(f"weights    : {yolo_cfg.get('weights')}")
    print(f"frame_skip : {cfg.get('pipeline', {}).get('frame_skip')}")
    print(f"min_hits   : {presence_cfg.get('min_hits')}")
    print(
        f"hysteresis : enabled={hysteresis_cfg.get('enabled')} "
        f"enter={hysteresis_cfg.get('enter_frames')} exit={hysteresis_cfg.get('exit_frames')}"
    )


def _print_segment_table(segments: List[Dict[str, Any]]) -> None:
    header = f"{'segment':>8} {'tracks':>8} {'gaze_tracks':>12} {'ratio':>8} {'look_ms':>10}"
    print(header)
    print("-" * len(header))

    tot_n = tot_gaze = tot_dur = 0
    for seg in segments:
        tracks = seg["tracks"]
        n = len(tracks)
        gaze_n = sum(1 for t in tracks if t["look_times"])
        dur = sum(t["total_look_duration_ms"] for t in tracks)
        ratio = f"{gaze_n / n * 100:.1f}%" if n else "0.0%"
        print(f"{seg['segment']['index']:>8} {n:>8} {gaze_n:>12} {ratio:>8} {dur:>10}")
        tot_n += n
        tot_gaze += gaze_n
        tot_dur += dur

    total_ratio = f"{tot_gaze / tot_n * 100:.1f}%" if tot_n else "0.0%"
    print("-" * len(header))
    print(f"{'total':>8} {tot_n:>8} {tot_gaze:>12} {total_ratio:>8} {tot_dur:>10}")


def main() -> None:
    run_pipeline()  # main.py 실행 → 세그먼트 JSON 저장 + "총 처리 시간" 로그 출력

    cfg = load_config("configs/dev.yaml")
    json_dir = cfg.get("output", {}).get("json_dir", "data/output/segments/")
    segments = _load_segments(json_dir)

    print()
    _print_config_summary(cfg)
    print()
    _print_segment_table(segments)


if __name__ == "__main__":
    main()
