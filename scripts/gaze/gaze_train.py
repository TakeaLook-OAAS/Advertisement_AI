"""
GazeNet 학습 스크립트.

사전 조건:
    python -m scripts.gaze.gaze_generate_labels   # 학습 라벨 생성

사용법 (프로젝트 루트에서):
    python -m scripts.gaze.gaze_train
    python -m scripts.gaze.gaze_train --epochs 100 --batch 32

출력: weights/gaze/gaze_pytorch.pth  (val angular error 최소 시점)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.gaze.gaze_net import GazeNet

# ── 기본 설정 ──────────────────────────────────────────────────
LABELS_PATH = "data/benchmark/gaze/labels_train.json"
IMAGES_DIR = "data/benchmark/gaze/MPIIFaceGaze"
OUTPUT_PATH = "weights/gaze/gaze_pytorch.pth"
EYE_SIZE = GazeNet.EYE_SIZE  # 60


# ── Dataset ───────────────────────────────────────────────────

class GazeDataset(Dataset):
    def __init__(self, samples: list, augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        frame = cv2.imread(os.path.join(IMAGES_DIR, item["image"]))
        if frame is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        le, re = item["left_eye"], item["right_eye"]
        left_crop = frame[le["y1"]:le["y2"], le["x1"]:le["x2"]]
        right_crop = frame[re["y1"]:re["y2"], re["x1"]:re["x2"]]

        if left_crop.size == 0 or right_crop.size == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        left_crop = cv2.resize(left_crop, (EYE_SIZE, EYE_SIZE))
        right_crop = cv2.resize(right_crop, (EYE_SIZE, EYE_SIZE))

        gaze = np.array([item["gaze"]["x"], item["gaze"]["y"], item["gaze"]["z"]], dtype=np.float32)
        hp = item["headpose"]
        hp_vec = np.array([hp["yaw"], hp["pitch"], hp["roll"]], dtype=np.float32) / 90.0

        if self.augment:
            # 좌우 반전: 눈 교환 + gaze x / yaw 부호 반전
            if random.random() < 0.5:
                left_crop, right_crop = cv2.flip(right_crop, 1), cv2.flip(left_crop, 1)
                gaze = gaze.copy()
                gaze[0] = -gaze[0]
                hp_vec = hp_vec.copy()
                hp_vec[0] = -hp_vec[0]

            # Brightness jitter
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                left_crop = np.clip(left_crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                right_crop = np.clip(right_crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        left_t = torch.tensor(left_crop.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        right_t = torch.tensor(right_crop.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        hp_t = torch.tensor(hp_vec, dtype=torch.float32)
        gaze_t = torch.tensor(gaze, dtype=torch.float32)

        return left_t, right_t, hp_t, gaze_t


# ── 메트릭 ────────────────────────────────────────────────────

def angular_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> float:
    cos = F.cosine_similarity(pred, gt).clamp(-1.0, 1.0)
    return torch.acos(cos).mean().item() * (180.0 / math.pi)


# ── 훈련 / 평가 루프 ──────────────────────────────────────────

def train_one_epoch(model: GazeNet, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for left, right, hp, gaze in loader:
        left, right, hp, gaze = (
            left.to(device), right.to(device), hp.to(device), gaze.to(device)
        )
        optimizer.zero_grad()
        pred = model(left, right, hp)
        loss = (1.0 - F.cosine_similarity(pred, gaze)).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(left)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: GazeNet, loader: DataLoader, device) -> float:
    model.eval()
    preds, gts = [], []
    for left, right, hp, gaze in loader:
        pred = model(left.to(device), right.to(device), hp.to(device))
        preds.append(pred.cpu())
        gts.append(gaze)
    return angular_error_deg(torch.cat(preds), torch.cat(gts))


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GazeNet 학습")
    parser.add_argument("--labels", default=LABELS_PATH, help="학습 라벨 JSON 경로")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="검증 비율")
    parser.add_argument("--out", default=OUTPUT_PATH, help="가중치 저장 경로")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"라벨 파일 없음: {args.labels}")
        print("먼저 실행하세요: python -m scripts.gaze.gaze_generate_labels")
        return

    with open(args.labels, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"총 샘플: {len(samples)}")

    # Train / Val 분할
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    train_samples = samples[n_val:]
    val_samples = samples[:n_val]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = GazeDataset(train_samples, augment=True)
    val_ds = GazeDataset(val_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GazeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_err = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_err = evaluate(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_err < best_val_err:
            best_val_err = val_err
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": model.state_dict()}, args.out)
            marker = "  ✓ saved"

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val={val_err:.2f}°{marker}")

    print(f"\n학습 완료. Best val angular error: {best_val_err:.2f}°")
    print(f"저장 위치: {args.out}")


if __name__ == "__main__":
    main()
