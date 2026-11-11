from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset

from app.data import CLASS_NAMES, HistologyDataset
from app.model import HybridClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LC25000 model using its held-out test split")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset = HistologyDataset(args.data_dir, training=False)
    test_ids = checkpoint["test_indices"]
    loader = DataLoader(Subset(dataset, test_ids), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = HybridClassifier(num_classes=len(CHECKPOINT_CLASSES := checkpoint["classes"]), handcrafted_dim=checkpoint.get("feature_dim", 1821), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    labels, probs = [], []
    with torch.no_grad():
        for images, handcrafted, target in loader:
            logits = model(images.to(device), handcrafted.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.numpy())
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    predictions = probs.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=range(len(CHECKPOINT_CLASSES)), zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=range(len(CHECKPOINT_CLASSES)))
    specificity = []
    for idx in range(len(CHECKPOINT_CLASSES)):
        true_negative = cm.sum() - cm[idx, :].sum() - cm[:, idx].sum() + cm[idx, idx]
        false_positive = cm[:, idx].sum() - cm[idx, idx]
        specificity.append(float(true_negative / (true_negative + false_positive)) if (true_negative + false_positive) else 0.0)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_specificity": float(np.mean(specificity)),
        "specificity_by_class": dict(zip(CHECKPOINT_CLASSES, specificity)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(labels, predictions, labels=range(len(CHECKPOINT_CLASSES)), target_names=CHECKPOINT_CLASSES, output_dict=True, zero_division=0),
    }

    try:
        metrics["roc_auc_macro_ovr"] = float(roc_auc_score(label_binarize(labels, classes=range(len(CHECKPOINT_CLASSES))), probs, average="macro", multi_class="ovr"))
    except ValueError:
        metrics["roc_auc_macro_ovr"] = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CHECKPOINT_CLASSES)))
    ax.set_yticks(range(len(CHECKPOINT_CLASSES)))
    ax.set_xticklabels(CHECKPOINT_CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CHECKPOINT_CLASSES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    for row in range(len(CHECKPOINT_CLASSES)):
        for col in range(len(CHECKPOINT_CLASSES)):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 6))
    truth_bin = label_binarize(labels, classes=range(len(CHECKPOINT_CLASSES)))
    for idx, name in enumerate(CHECKPOINT_CLASSES):
        try:
            fpr, tpr, _ = roc_curve(truth_bin[:, idx], probs[:, idx])
            auc = roc_auc_score(truth_bin[:, idx], probs[:, idx])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
