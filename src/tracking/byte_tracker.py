"""
ByteTrack: 다중 객체 트래커.

알고리즘 (프레임별):
  1. 검출을 D_high (≥ track_thresh)와 D_low (≥ low_thresh)로 분리
  2. 모든 Kalman 상태 예측
  3. 1차 매칭: 확정 트랙 + lost 트랙  ↔  D_high   (IoU 비용)
  4. 2차 매칭: 매칭 안 된 tracked 트랙 ↔  D_low    (IoU 비용)
     → 매칭 실패 트랙 → Lost 전환
  5. 3차 매칭: 미확정 트랙             ↔  남은 D_high
     → 매칭 실패 미확정 트랙 → 제거
     → 남은 검출              → 새 STrack 생성
  6. max_lost_frames 초과 Lost 트랙 제거
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False

from .strack import STrack, TrackState


# ---------------------------------------------------------------------------
# IoU 헬퍼
# ---------------------------------------------------------------------------

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    벡터화 IoU 행렬.
    a: (N, 4) tlbr   b: (M, 4) tlbr   →  (N, M)
    """
    a = a[:, None]   # (N, 1, 4)
    b = b[None]      # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter_area

    return np.where(union > 0, inter_area / union, 0.0)


def _cost_matrix(tracks: List[STrack], dets: List[STrack]) -> np.ndarray:
    """비용 = 1 - IoU  (shape: N_tracks × N_dets)."""
    if not tracks or not dets:
        return np.empty((len(tracks), len(dets)), dtype=float)
    t_bboxes = np.stack([t.tlbr for t in tracks])
    d_bboxes = np.stack([d.tlbr for d in dets])
    return 1.0 - _iou_batch(t_bboxes, d_bboxes)


def _hungarian(
    cost: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    비용 행렬에 헝가리안 알고리즘 적용.
    cost ≤ thresh인 쌍만 유효 매칭으로 수락.
    반환: (매칭 쌍 목록, 매칭 안 된 행, 매칭 안 된 열).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    if _SCIPY:
        rows, cols = linear_sum_assignment(cost)
    else:
        # 그리디 폴백 (실시간 사용에 충분)
        rows, cols = _greedy_assignment(cost)

    matched = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    matched_r = {r for r, _ in matched}
    matched_c = {c for _, c in matched}
    unmatched_r = [i for i in range(cost.shape[0]) if i not in matched_r]
    unmatched_c = [j for j in range(cost.shape[1]) if j not in matched_c]
    return matched, unmatched_r, unmatched_c


def _greedy_assignment(cost: np.ndarray):
    """scipy 미설치 시 그리디 방식으로 매칭."""
    rows, cols = [], []
    used_r, used_c = set(), set()
    order = np.argsort(cost.ravel())
    for idx in order:
        r, c = divmod(int(idx), cost.shape[1])
        if r not in used_r and c not in used_c:
            rows.append(r)
            cols.append(c)
            used_r.add(r)
            used_c.add(c)
    return rows, cols


# ---------------------------------------------------------------------------
# BYTETracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """
    ByteTrack 다중 객체 트래커.

    설정 키 (YAML의 `tracking:` 하위):
      track_thresh      float  0.5   고신뢰도 검출 임계값
      low_thresh        float  0.1   최소 검출 점수
      match_thresh      float  0.8   1차/2차 매칭 IoU 비용 임계값
      max_lost_frames   int    30    Lost 트랙 유지 프레임 수
      min_hits          int    1     트랙 반환 전 최소 매칭 횟수
    """

    def __init__(self, cfg: dict):
        self.track_thresh = float(cfg.get("track_thresh", 0.5))
        self.low_thresh   = float(cfg.get("low_thresh",   0.1))
        self.match_thresh = float(cfg.get("match_thresh", 0.8))
        self.max_lost     = int(cfg.get("max_lost_frames", 30))
        self.min_hits     = int(cfg.get("min_hits", 1))

        self.frame_id: int = 0
        self.tracked_stracks: List[STrack] = []   # 활성 추적 트랙
        self.lost_stracks:    List[STrack] = []   # 일시적 소실 트랙
        self.removed_stracks: List[STrack] = []   # 완전 제거된 트랙

    # ------------------------------------------------------------------
    def update(self, detections: np.ndarray) -> List[STrack]:
        """
        한 프레임의 추적 단계 실행.

        detections: np.ndarray  shape (N, 5)  [x1, y1, x2, y2, score]
                                또는 빈 프레임의 경우 (0, 5).

        반환: 확정된 활성 STrack 객체 목록.
        """
        self.frame_id += 1

        # ── 1. 검출 분리 ────────────────────────────────────────────────
        det_high: List[STrack] = []
        det_low:  List[STrack] = []

        if len(detections) > 0:
            scores = detections[:, 4]
            for i, s in enumerate(scores):
                box = detections[i, :4]
                if s >= self.track_thresh:
                    det_high.append(STrack(box, float(s)))
                elif s >= self.low_thresh:
                    det_low.append(STrack(box, float(s)))

        # ── 2. Kalman 상태 예측 ─────────────────────────────────────────
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict()

        # tracked_stracks 내에서 확정/미확정 분리
        confirmed   = [t for t in self.tracked_stracks if t.is_activated]
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]

        # ── 3. 1차 매칭: (확정 + lost) ↔ det_high ──────────────────────
        pool_1 = confirmed + self.lost_stracks
        cost_1 = _cost_matrix(pool_1, det_high)
        matched_1, unmatched_t1, unmatched_d1 = _hungarian(cost_1, self.match_thresh)

        activated_tracks: List[STrack] = []
        refound_tracks:   List[STrack] = []

        for ti, di in matched_1:
            t, d = pool_1[ti], det_high[di]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated_tracks.append(t)
            else:                            # Lost → 재활성화
                t.re_activate(d, self.frame_id)
                refound_tracks.append(t)

        # ── 4. 2차 매칭: 매칭 안 된 tracked 트랙 ↔ det_low ────────────
        unmatched_tracked = [
            pool_1[i] for i in unmatched_t1
            if pool_1[i].state == TrackState.Tracked
        ]
        cost_2 = _cost_matrix(unmatched_tracked, det_low)
        matched_2, unmatched_t2, _ = _hungarian(cost_2, self.match_thresh)

        for ti, di in matched_2:
            unmatched_tracked[ti].update(det_low[di], self.frame_id)
            activated_tracks.append(unmatched_tracked[ti])

        # 두 매칭 모두 실패한 트랙 → Lost 전환
        # (새로 Lost된 Tracked 트랙만 추가.
        #  이미 Lost 상태인 트랙은 step 8의 dedup dict에서 처리.)
        lost_tracks: List[STrack] = []
        for i in unmatched_t2:
            t = unmatched_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
            lost_tracks.append(t)

        # ── 5. 3차 매칭: 미확정 트랙 ↔ 남은 det_high ──────────────────
        leftover_high = [det_high[i] for i in unmatched_d1]
        cost_3 = _cost_matrix(unconfirmed, leftover_high)
        matched_3, unmatched_uc, unmatched_d3 = _hungarian(cost_3, 0.7)

        for ti, di in matched_3:
            unconfirmed[ti].update(leftover_high[di], self.frame_id)
            activated_tracks.append(unconfirmed[ti])

        removed_tracks: List[STrack] = []
        for i in unmatched_uc:
            unconfirmed[i].mark_removed()
            removed_tracks.append(unconfirmed[i])

        # ── 6. 매칭 안 된 고신뢰도 검출 → 새 트랙 생성 ─────────────────
        for i in unmatched_d3:
            d = leftover_high[i]
            if d.score >= self.track_thresh:
                d.activate(self.frame_id)
                activated_tracks.append(d)

        # ── 7. max_lost_frames 초과 Lost 트랙 제거 ─────────────────────
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.max_lost:
                t.mark_removed()
                removed_tracks.append(t)

        # ── 8. 상태 목록 재구성 ─────────────────────────────────────────
        self.tracked_stracks = [
            t for t in (activated_tracks + refound_tracks)
            if t.state == TrackState.Tracked
        ]
        # track_id를 키로 dict 사용해 중복 방지.
        # 새로 lost된 트랙이 기존 lost 트랙보다 우선.
        lost_dict = {t.track_id: t for t in self.lost_stracks + lost_tracks
                     if t.state == TrackState.Lost}
        self.lost_stracks = list(lost_dict.values())
        self.removed_stracks.extend(removed_tracks)

        # ── 9. 확정된 활성 트랙 반환 ───────────────────────────────────
        return [
            t for t in self.tracked_stracks
            if t.is_activated and t.tracklet_len >= self.min_hits - 1
        ]

    def reset(self) -> None:
        """트래커 상태 초기화 (씬 전환 등에 사용)."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
