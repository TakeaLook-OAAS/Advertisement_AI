"""GazeNet 모델 아키텍처 - 훈련·추론에서 공통으로 사용."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeNet(nn.Module):
    """
    입력:
        left, right : (B, 3, 60, 60) eye crop, float32 in [0, 1], BGR
        hp          : (B, 3) head pose [yaw, pitch, roll] / 90  (정규화된 값)
    출력:
        (B, 3) 카메라 좌표계 단위 gaze 방향 벡터
    """

    EYE_SIZE = 60

    def __init__(self) -> None:
        super().__init__()
        s = self.EYE_SIZE // 8  # 3번 MaxPool2d(2) → 60→30→15→7

        # 좌우 눈 공유 인코더
        self._eye_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * s * s, 256),
            nn.ReLU(inplace=True),
        )

        self._hp_enc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
        )

        self._regressor = nn.Sequential(
            nn.Linear(256 * 2 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        hp: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat(
            [self._eye_enc(left), self._eye_enc(right), self._hp_enc(hp)],
            dim=1,
        )
        return F.normalize(self._regressor(feat), dim=1)
