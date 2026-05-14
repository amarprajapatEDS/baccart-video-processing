"""Train the [Rank][Suit] card classifier on a folder of card crops.

Expected dataset layout:
    data/
      cards/
        AH/  *.png
        2H/  *.png
        ...
        KS/  *.png   (52 folders total)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import default_config
from src.classification import build_classifier
from src.classification.card_classes import CARD_LABELS
from utils import resolve_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Folder with 52 per-class subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="weights/mobilenetv3_cards_fp16.pt")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    set_seed(42)
    cfg = default_config()
    device = resolve_device(cfg.classification.device)

    h, w = cfg.classification.input_size
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.classification.pixel_mean, std=cfg.classification.pixel_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.classification.pixel_mean, std=cfg.classification.pixel_std),
    ])

    full = ImageFolder(args.data, transform=train_tf)
    if set(full.classes) != set(CARD_LABELS):
        missing = set(CARD_LABELS) - set(full.classes)
        extra = set(full.classes) - set(CARD_LABELS)
        raise RuntimeError(f"class mismatch — missing: {sorted(missing)} extra: {sorted(extra)}")

    n_val = max(1, int(len(full) * 0.15))
    val_set, train_set = torch.utils.data.random_split(
        full, [n_val, len(full) - n_val], generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = val_tf  # type: ignore[attr-defined]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_classifier(num_classes=cfg.classification.num_classes,
                             backbone=cfg.classification.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item(); total += y.size(0)
        acc = correct / max(1, total)
        print(f"[epoch {epoch}] train_loss={running/len(train_set):.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "classes": CARD_LABELS, "val_acc": acc}, out_path)
            print(f"  saved best -> {out_path}")
    print(f"done. best val_acc={best_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
