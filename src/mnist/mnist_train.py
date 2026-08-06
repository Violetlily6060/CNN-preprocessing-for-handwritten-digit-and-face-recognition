import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config
import src.pipeline as pipe
from src.mnist.mnist_dataset import load_mnist_train, load_mnist_val
from src.model import CNN


class PreloadedDataset(Dataset):
    def __init__(self, dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img)
            labels.append(
                label if isinstance(label, torch.Tensor) else torch.tensor(label)
            )

        self.images = torch.stack(images)
        self.labels = torch.stack(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def generate_pipeline_permutations():
    components = {
        "p2": pipe.p2,
        "p3": pipe.p3,
        "p4": pipe.p4,
    }

    pipelines = {"p1": (pipe.p1_mnist, "p1")}
    keys = list(components.keys())

    for r in range(1, len(keys) + 1):
        for perm in itertools.permutations(keys, r):
            name = "p1" + "".join(perm)
            transforms = [pipe.p1_mnist] + [components[k] for k in perm]
            pipelines[name] = (v2.Compose(transforms), name)

    return pipelines


def run_epoch(model, dataloader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("experiment") / "mnist" / timestamp
    models_dir = base_dir / "trained_model"
    stats_dir = base_dir / "training_statistic"

    models_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    pipelines = generate_pipeline_permutations()

    print(f"Starting MNIST Pipeline Experiments on device: {config.DEVICE}")
    print(f"Total permutations: {len(pipelines)}")
    print(f"Max Epochs: {config.MNIST.EPOCHS} | Patience: {config.MNIST.PATIENCE}")
    print(f"Artifacts folder: {base_dir.resolve()}\n")

    for p_name, (transform_fn, file_stem) in pipelines.items():
        print("=" * 50)
        print(f" Running Pipeline: {p_name}")
        print("=" * 50)

        raw_train_dataset = load_mnist_train(transform_fn=transform_fn)
        raw_val_dataset = load_mnist_val(transform_fn=transform_fn)

        print(" Preprocessing & loading dataset into memory...")
        preload_start = time.time()
        train_dataset = PreloadedDataset(raw_train_dataset)
        val_dataset = PreloadedDataset(raw_val_dataset)
        preload_duration = round(time.time() - preload_start, 2)
        print(f" Data preloaded in {preload_duration}s")

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        model = CNN(num_class=config.MNIST.NUM_CLASS).to(config.DEVICE)
        dummy_input = torch.zeros(2, *config.MNIST.INPUT_SHAPE).to(config.DEVICE)
        model(dummy_input)

        optimizer = config.OPTIMIZER_CLASS(
            model.parameters(), lr=config.LEARNING_RATE
        )
        criterion = config.CRITERION

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config.DEVICE)

        stats = {
            "pipeline_name": p_name,
            "timestamp": timestamp,
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_name": (
                    torch.cuda.get_device_name(config.DEVICE)
                    if torch.cuda.is_available()
                    else platform.processor()
                ),
                "os": platform.system(),
            },
            "hyperparameters": {
                "batch_size": config.BATCH_SIZE,
                "max_epochs": config.MNIST.EPOCHS,
                "patience": config.MNIST.PATIENCE,
                "learning_rate": config.LEARNING_RATE,
                "dropout_rate": config.DROPOUT_RATE,
                "optimizer": optimizer.__class__.__name__,
                "criterion": criterion.__class__.__name__,
            },
            "model_metrics": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": round(
                    sum(p.numel() * p.element_size() for p in model.parameters())
                    / (1024**2),
                    4,
                ),
            },
            "history": [],
            "best_epoch": 0,
            "stopped_epoch": 0,
            "early_stopped": False,
            "best_val_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "final_train_accuracy": 0.0,
            "final_val_accuracy": 0.0,
            "overfitting_gap": 0.0,
            "performance_metrics": {
                "preload_time_sec": preload_duration,
                "total_training_time_sec": 0.0,
                "avg_epoch_time_sec": 0.0,
                "throughput_samples_per_sec": 0.0,
                "peak_gpu_memory_mb": 0.0,
            },
        }

        best_val_loss = float("inf")
        patience_counter = 0
        experiment_start_time = time.time()
        actual_epochs_run = 0

        for epoch in range(1, config.MNIST.EPOCHS + 1):
            epoch_start_time = time.time()
            actual_epochs_run = epoch

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, is_train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, config.DEVICE, is_train=False
            )

            epoch_duration = round(time.time() - epoch_start_time, 2)

            stats["history"].append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "epoch_time_sec": epoch_duration,
                }
            )

            print(
                f"Epoch [{epoch:02d}/{config.MNIST.EPOCHS:02d}] | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"Time: {epoch_duration}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                stats["best_epoch"] = epoch
                stats["best_val_loss"] = round(best_val_loss, 4)
                stats["best_val_accuracy"] = round(val_acc, 4)

                model_save_path = models_dir / f"{file_stem}.pt"
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                print(
                    f" --> Validation loss did not improve. Patience: {patience_counter}/{config.MNIST.PATIENCE}"
                )

            if patience_counter >= config.MNIST.PATIENCE:
                print(f" Early stopping triggered at epoch {epoch}!")
                stats["early_stopped"] = True
                break

        total_time = round(time.time() - experiment_start_time, 2)
        total_samples = len(train_dataset) * actual_epochs_run

        stats["stopped_epoch"] = actual_epochs_run
        stats["final_train_accuracy"] = stats["history"][-1]["train_acc"]
        stats["final_val_accuracy"] = stats["history"][-1]["val_acc"]
        stats["overfitting_gap"] = round(
            stats["final_train_accuracy"] - stats["final_val_accuracy"], 4
        )

        stats["performance_metrics"]["total_training_time_sec"] = total_time
        stats["performance_metrics"]["avg_epoch_time_sec"] = (
            round(total_time / actual_epochs_run, 2) if actual_epochs_run > 0 else 0.0
        )
        stats["performance_metrics"]["throughput_samples_per_sec"] = (
            round(total_samples / total_time, 2) if total_time > 0 else 0.0
        )

        if torch.cuda.is_available():
            stats["performance_metrics"]["peak_gpu_memory_mb"] = round(
                torch.cuda.max_memory_allocated(config.DEVICE) / (1024**2), 2
            )

        json_save_path = stats_dir / f"{file_stem}.json"
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        print(
            f"Finished {p_name}. Model saved to {model_save_path} | Stats saved to {json_save_path}\n"
        )


if __name__ == "__main__":
    run_experiment()