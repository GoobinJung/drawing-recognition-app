import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DrawingCNN


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def load_dataset(split_name):
    with np.load(
        DATA_DIR / f"{split_name}.npz",
        allow_pickle=False,
    ) as data:
        images = torch.from_numpy(data["images"]).float()
        labels = torch.from_numpy(data["labels"]).long()
        class_names = data["class_names"].tolist()

    # Convert (N, 28, 28) into (N, 1, 28, 28).
    # Change pixel values from 0–255 to 0–1.
    images = images.unsqueeze(1) / 255.0

    return TensorDataset(images, labels), class_names


def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_function(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images,
    )


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, class_names = load_dataset("train")
    validation_dataset, validation_classes = load_dataset("validation")

    if class_names != validation_classes:
        raise ValueError("The class names do not match.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = DrawingCNN(num_classes=len(class_names))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model_path = MODEL_DIR / "drawing_cnn.pt"
    history_path = RESULTS_DIR / "training_history.json"
    best_accuracy = -1.0

    history = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "class_names": class_names,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "torch_version": str(torch.__version__),
        "epochs": [],
    }

    print("Training on CPU.", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, optimizer
        )

        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function
        )

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": 28,
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "seed": SEED,
            }, model_path)

            history["best_epoch"] = epoch
            history["best_validation_accuracy"] = best_accuracy

        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.2%}",
            flush=True,
        )

    print("\nTraining finished.")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()