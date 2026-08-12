"""
트래커 벤치마크: ByteTrack(공식) vs OC-SORT
같은 샘플 영상에서 같은 YOLO 검출 시퀀스를 두 트래커에 동일하게 흘려서 비교한다.

주의: 프로젝트에 트랙 ID 정답(GT)이 없어서 MOTA/IDF1 같은 정확도 지표는 낼 수 없다.
      대신 GT 없이도 잴 수 있는 프록시 지표로 비교한다:
      - update() 처리 속도 (FPS)
      - 트랙당 평균 지속 프레임 수 (길수록 끊김/재검출이 적다는 신호)
      - 프레임당 평균/최대 활성 트랙 수
      - 총 발급된 고유 ID 수 (같은 영상에서 적을수록 ID가 안정적이라는 신호)

사용법:
    python -m tests.benchmark.tracker.test_tracker
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml
from loguru import logger

from src.utils.types import Det

CONFIG_PATH = "configs/test.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tracker"]


# ── 검출 시퀀스 수집 ─────────────────────────────────────────

def collect_detections(video_path: str, yolo_cfg: Dict[str, Any], max_frames: int) -> List[List[Det]]:
    """영상을 읽어 프레임별 YOLO 검출 결과를 미리 뽑아둔다.
    두 트래커가 완전히 동일한 입력을 받도록 하기 위함."""
    from src.models.yolo_detector import YoloDetector

    detector = YoloDetector(yolo_cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    dets_per_frame: List[List[Det]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets_per_frame.append(detector.detect(frame))
        if max_frames > 0 and len(dets_per_frame) >= max_frames:
            break
    cap.release()
    return dets_per_frame


def get_video_size(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# ── 벤치마크 ──────────────────────────────────────────────────

@dataclass
class TrackerStats:
    total_update_s: float = 0.0
    n_frames: int = 0
    active_counts: List[int] = field(default_factory=list)
    track_frame_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def fps(self) -> float:
        return self.n_frames / self.total_update_s if self.total_update_s > 0 else 0.0

    @property
    def unique_ids(self) -> int:
        return len(self.track_frame_counts)

    @property
    def avg_track_len(self) -> float:
        if not self.track_frame_counts:
            return 0.0
        return float(np.mean(list(self.track_frame_counts.values())))

    @property
    def avg_active(self) -> float:
        return float(np.mean(self.active_counts)) if self.active_counts else 0.0

    @property
    def max_active(self) -> int:
        return max(self.active_counts) if self.active_counts else 0


def bench_tracker(tracker, dets_per_frame: List[List[Det]]) -> TrackerStats:
    stats = TrackerStats()
    for dets in dets_per_frame:
        t0 = perf_counter()
        tracks = tracker.update(dets)
        stats.total_update_s += perf_counter() - t0

        stats.n_frames += 1
        stats.active_counts.append(len(tracks))
        for t in tracks:
            stats.track_frame_counts[t.track_id] += 1
    return stats


# ── 결과 출력 ────────────────────────────────────────────────

def print_result(name: str, stats: TrackerStats) -> None:
    print(f"\n=== {name} ===")
    print(f"  {'FPS':>10}{'고유ID수':>10}{'평균지속(f)':>12}{'평균활성':>10}{'최대활성':>10}")
    print(
        f"  {stats.fps:>10.1f}{stats.unique_ids:>10d}"
        f"{stats.avg_track_len:>12.1f}{stats.avg_active:>10.2f}{stats.max_active:>10d}"
    )


# ── 메인 ─────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    video_path = cfg["video"]
    max_frames = int(cfg.get("max_frames", 0))

    logger.info(f"검출 시퀀스 수집 중: {video_path} (max_frames={max_frames})")
    dets_per_frame = collect_detections(video_path, cfg["yolo"], max_frames)
    img_w, img_h = get_video_size(video_path)
    logger.info(f"프레임 수: {len(dets_per_frame)}  해상도: {img_w}x{img_h}")

    trackers_cfg = cfg["trackers"]

    results: Dict[str, TrackerStats] = {}

    if "bytetrack" in trackers_cfg:
        from src.models.bytetrack_tracker import OfficialByteTrackAdapter

        bt_cfg = dict(trackers_cfg["bytetrack"])
        bt_cfg["img_w"], bt_cfg["img_h"] = img_w, img_h
        logger.info("ByteTrack 평가 중...")
        results["bytetrack"] = bench_tracker(OfficialByteTrackAdapter(bt_cfg), dets_per_frame)

    if "ocsort" in trackers_cfg:
        from src.models.ocsort_tracker import OCSortAdapter

        oc_cfg = dict(trackers_cfg["ocsort"])
        oc_cfg["img_w"], oc_cfg["img_h"] = img_w, img_h
        logger.info("OC-SORT 평가 중...")
        results["ocsort"] = bench_tracker(OCSortAdapter(oc_cfg), dets_per_frame)

    print("\n=== 트래커 비교 (GT 없는 프록시 지표, 동일 검출 시퀀스 기준) ===")
    for name, stats in results.items():
        print_result(name, stats)


if __name__ == "__main__":
    main()
