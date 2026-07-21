"""EyeNet 모델 아키텍처 - 훈련·추론에서 공통으로 사용."""
from __future__ import annotations

import torch
import torch.nn as nn


class EyeNet(nn.Module):
    """
    입력:
        face : (B, 3, 60, 60) face crop, float32 in [0, 1], BGR
    출력:
        (B, 8) face crop 기준 정규화 눈 bbox 좌표
               [left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2]
               좌/우는 이미지 기준(왼쪽 눈 / 오른쪽 눈)
    """

    FACE_SIZE = 60

    def __init__(self) -> None:
        super().__init__()
        s = self.FACE_SIZE // 8  # 3번 MaxPool2d(2) → 60→30→15→7

        self._enc = nn.Sequential(
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

        self._regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 8),
        )

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        return self._regressor(self._enc(face))
