from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets

from app.data import CLASS_NAMES, HistologyDataset, seed_everything


def split_dataset(data_dir: str | Path, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    dataset = datasets.ImageFolder(str(data_dir))
    target_names = sorted(dataset.classes)
    expected = sorted(CLASS_NAMES)
    if target_names != expected:
        raise ValueError(f"Expected class folders: {expected}. Found: {target_names}")
    indices = np.arange(len(dataset))
    targets = np.asarray(dataset.targets)
    train_val, test = train_test_split(indices, test_size=0.15, random_state=seed, stratify=targets)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=seed, stratify=targets[train_val])
    return train.tolist(), val.tolist(), test.tolist()


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = split_dataset(args.data_dir, seed=args.seed)
    (output_dir / "splits.json").write_text(
        json.dumps({"train": train_idx, "validation": val_idx, "test": test_idx, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )

    train_dataset = HistologyDataset(args.data_dir, training=True)
    val_dataset = HistologyDataset(args.data_dir, training=False)

    from torch.utils.data import DataLoader, Subset

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    from app.model import HybridClassifier

    model = HybridClassifier(num_classes=len(CLASS_NAMES), handcrafted_dim=train_dataset.feature_dim, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0

    from sklearn.metrics import f1_score

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        true_labels = []
        predicted_labels = []
        for images, handcrafted, labels in train_loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, handcrafted)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predicted_labels.extend(logits.argmax(dim=1).detach().cpu().tolist())
            true_labels.extend(labels.detach().cpu().tolist())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(true_labels, predicted_labels, average="macro")

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for images, handcrafted, labels in val_loader:
                logits = model(images.to(device), handcrafted.to(device))
                val_pred.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(labels.tolist())
        val_f1 = f1_score(val_true, val_pred, average="macro")

        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
                "feature_dim": train_dataset.feature_dim,
                "validation_f1": best_f1,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "test_indices": test_idx,
                "seed": args.seed,
            }
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Best validation macro-F1: {best_f1:.4f} ; checkpoint saved to {output_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train the hybrid LC25000 classification model")
    parser.add_argument("--data-dir", required=True, help="Directory containing class folders for LC25000")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where checkpoints and splits are stored")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
