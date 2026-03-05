"""
Bounding Box 추적을 위한 표준 Kalman 필터.

상태 벡터 (8차원): [cx, cy, w, h, vcx, vcy, vw, vh]
  - 등속도 모델
측정 벡터 (4차원): [cx, cy, w, h]
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    ndim = 4

    def __init__(self):
        dt = 1.0

        # 상태 전이 행렬 F (8×8)
        self.F = np.eye(2 * self.ndim, dtype=float)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = dt

        # 측정 행렬 H (4×8)
        self.H = np.eye(self.ndim, 2 * self.ndim, dtype=float)

        # 위치/속도 성분의 프로세스 노이즈 가중치
        self._std_pos = 1.0 / 20.0
        self._std_vel = 1.0 / 160.0

    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray):
        """[cx, cy, w, h] 검출로부터 초기 평균과 공분산 생성."""
        h = measurement[3]
        mean = np.r_[measurement, np.zeros(self.ndim, dtype=float)]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    # ------------------------------------------------------------------
    def predict(self, mean: np.ndarray, cov: np.ndarray):
        """Kalman 예측 단계."""
        h = mean[3]
        std_pos = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h]
        std_vel = [self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    # ------------------------------------------------------------------
    def project(self, mean: np.ndarray, cov: np.ndarray):
        """상태를 측정 공간으로 투영."""
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        R = np.diag(np.square(std))
        projected_mean = self.H @ mean
        projected_cov = self.H @ cov @ self.H.T + R
        return projected_mean, projected_cov

    # ------------------------------------------------------------------
    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        """Kalman 보정 단계."""
        proj_mean, proj_cov = self.project(mean, cov)

        # 정규화: 거의 특이 행렬 역산 방지
        proj_cov += np.eye(self.ndim) * 1e-3

        # Kalman 이득  K = P H^T (H P H^T + R)^{-1}
        K = cov @ self.H.T @ np.linalg.inv(proj_cov)
        innovation = measurement - proj_mean
        new_mean = mean + K @ innovation
        new_cov = (np.eye(2 * self.ndim) - K @ self.H) @ cov
        return new_mean, new_cov

    # ------------------------------------------------------------------
    def gating_distance(
        self, mean: np.ndarray, cov: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """
        투영된 상태에서 각 측정까지의 마할라노비스 거리.
        measurements: (N, 4)  →  반환 (N,)
        """
        proj_mean, proj_cov = self.project(mean, cov)
        diff = measurements - proj_mean          # (N, 4)
        cov_inv = np.linalg.inv(proj_cov)        # (4, 4)
        return np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
