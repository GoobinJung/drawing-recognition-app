import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import DrawingCNN


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "drawing_cnn.pt"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test.npz"
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    class_names = checkpoint["class_names"]

    with np.load(TEST_PATH, allow_pickle=False) as data:
        images = torch.from_numpy(data["images"]).float()
        labels = torch.from_numpy(data["labels"]).long()
        test_classes = data["class_names"].tolist()

    if class_names != test_classes:
        raise ValueError("The class names do not match.")

    images = images.unsqueeze(1) / 255.0

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = DrawingCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probability_batches = []

    with torch.no_grad():
        for batch_images, _ in loader:
            outputs = model(batch_images)
            probabilities = torch.softmax(outputs, dim=1)
            probability_batches.append(probabilities)

    probabilities = torch.cat(probability_batches)
    top_scores, top_classes = probabilities.topk(3, dim=1)
    predictions = top_classes[:, 0]

    total = len(labels)
    correct = (predictions == labels).sum().item()
    top3_correct = (
        (top_classes == labels.unsqueeze(1))
        .any(dim=1)
        .sum()
        .item()
    )

    confusion = np.zeros(
        (len(class_names), len(class_names)),
        dtype=np.int64,
    )
    np.add.at(
        confusion,
        (labels.numpy(), predictions.numpy()),
        1,
    )

    per_class_recall = {}
    for index, name in enumerate(class_names):
        class_total = int(confusion[index].sum())
        per_class_recall[name] = (
            int(confusion[index, index]) / class_total
            if class_total else 0.0
        )

    wrong_indices = (
        (predictions != labels)
        .nonzero(as_tuple=True)[0]
        .tolist()
    )

    mistakes = []
    for index in wrong_indices:
        mistakes.append({
            "sample_index": index,
            "actual": class_names[labels[index].item()],
            "predicted": class_names[predictions[index].item()],
            "confidence": top_scores[index, 0].item(),
        })

    metrics = {
        "split": "test",
        "model_epoch": checkpoint["epoch"],
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
        "mistakes": mistakes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "test_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model epoch: {checkpoint['epoch']}")
    print(f"Test images: {total:,}")
    print(f"Correct predictions: {correct:,}")
    print(f"Test accuracy: {correct / total:.2%}")
    print(f"Top-3 accuracy: {top3_correct / total:.2%}")

    print("\nRecall for each category:")
    for name, recall in per_class_recall.items():
        print(f"{name}: {recall:.2%}")

    print("\nConfusion matrix:")
    print("Rows = actual category; columns = predicted category")
    print("Category order:", ", ".join(class_names))
    print(confusion)

    print(f"\nIncorrect predictions: {len(mistakes)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()