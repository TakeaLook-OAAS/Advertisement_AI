"""
EyeNet 학습 스크립트 (눈 bbox 회귀, facial-landmarks-35-adas-0002 대체).
WFLW 눈 컨투어 8점 기반 tight bbox(+패딩)를 GT로 사용한다.

사전 조건:
    python -m scripts.eye.eye_generate_labels   # 학습 라벨 생성

사용법 (프로젝트 루트에서):
    python -m scripts.eye.eye_train_mse
    python -m scripts.eye.eye_train_mse --epochs 100 --batch 64

출력: weights/eye_detection/eye_pytorch.pth  (val mean IoU 최대 시점)
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.eye.eye_net import EyeNet

CONFIG_PATH = "configs/train.yaml"
FACE_SIZE = EyeNet.FACE_SIZE  # 60


# ── Dataset ───────────────────────────────────────────────────

class EyeDataset(Dataset):
    def __init__(self, samples: list, images_dir: str, augment: bool = False) -> None:
        self.samples = samples
        self.images_dir = images_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        frame = cv2.imread(os.path.join(self.images_dir, item["image"]))
        if frame is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        fb = item["face_bbox"]
        face_crop = frame[fb["y1"]:fb["y2"], fb["x1"]:fb["x2"]]
        if face_crop.size == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

        left = item["eyes"]["left"]
        right = item["eyes"]["right"]
        boxes = np.array([
            [left["x1"], left["y1"], left["x2"], left["y2"]],
            [right["x1"], right["y1"], right["x2"], right["y2"]],
        ], dtype=np.float32)  # (2, 4) = [left, right] x [x1,y1,x2,y2]

        if self.augment:
            # 좌우 반전: 이미지 반전 + x좌표 반전(1-x) + 좌/우 눈 교환
            if random.random() < 0.5:
                face_crop = cv2.flip(face_crop, 1)
                boxes = boxes.copy()
                old_x1, old_x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
                boxes[:, 0] = 1.0 - old_x2
                boxes[:, 2] = 1.0 - old_x1
                boxes = boxes[[1, 0]]  # left<->right 교환

            # Brightness jitter
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                face_crop = np.clip(face_crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img_t = torch.tensor(face_crop.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        target_t = torch.tensor(boxes.reshape(-1), dtype=torch.float32)  # (8,)

        return img_t, target_t


# ── 메트릭 ────────────────────────────────────────────────────

def box_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """축 정렬 박스 IoU. pred, gt: (..., 4) = (x1,y1,x2,y2)"""
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)

    ix1, iy1 = torch.maximum(px1, gx1), torch.maximum(py1, gy1)
    ix2, iy2 = torch.minimum(px2, gx2), torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    p_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    g_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = (p_area + g_area - inter).clamp(min=1e-8)

    return inter / union


def mean_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """pred, gt: (B, 8) = [left(x1,y1,x2,y2), right(x1,y1,x2,y2)]"""
    pred = pred.view(-1, 2, 4)
    gt = gt.view(-1, 2, 4)
    return box_iou(pred, gt).mean().item()


# ── 훈련 / 평가 루프 ──────────────────────────────────────────

def train_one_epoch(model: EyeNet, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for img, target in loader:
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(img)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: EyeNet, loader: DataLoader, device) -> float:
    model.eval()
    preds, gts = [], []
    for img, target in loader:
        pred = model(img.to(device))
        preds.append(pred.cpu())
        gts.append(target)
    return mean_iou(torch.cat(preds), torch.cat(gts))


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]["eye"]

    parser = argparse.ArgumentParser(description="EyeNet 학습 (WFLW)")
    parser.add_argument("--labels", default=cfg.get("labels_path", "data/benchmark/eye/labels_train.json"))
    parser.add_argument("--images-dir", default=cfg.get("images_dir", "data/train/WFLW_images"))
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 50))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 64))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    parser.add_argument("--val-ratio", type=float, default=cfg.get("val_ratio", 0.1))
    parser.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 2))
    parser.add_argument("--device", default=cfg.get("device", "cuda"))
    parser.add_argument("--out", default=cfg.get("output_path", "weights/eye_detection/eye_pytorch.pth"))
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"라벨 파일 없음: {args.labels}")
        print("먼저 실행하세요: python -m scripts.eye.eye_generate_labels")
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

    train_ds = EyeDataset(train_samples, args.images_dir, augment=True)
    val_ds = EyeDataset(val_samples, args.images_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}\n")

    model = EyeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_iou = evaluate(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": model.state_dict()}, args.out)
            marker = "  ✓ saved"

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_iou={val_iou:.3f}{marker}")

    print(f"\n학습 완료. Best val IoU: {best_val_iou:.3f}")
    print(f"저장 위치: {args.out}")


if __name__ == "__main__":
    main()
