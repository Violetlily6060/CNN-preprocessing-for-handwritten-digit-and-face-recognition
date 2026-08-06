import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import itertools
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config
import src.pipeline as pipe
from src.vggface2.vggface2_dataset import load_vggface2_test
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

    pipelines = {"p1": (pipe.p1_vggface2, "p1")}
    keys = list(components.keys())

    for r in range(1, len(keys) + 1):
        for perm in itertools.permutations(keys, r):
            name = "p1" + "".join(perm)
            transforms = [pipe.p1_vggface2] + [components[k] for k in perm]
            pipelines[name] = (v2.Compose(transforms), name)

    return pipelines


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 100) -> Tuple[float, float, List[float]]:
    f1_per_class = []
    weights = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(float(f1))
        weights.append(float(support))

    macro_f1 = float(np.mean(f1_per_class))
    weighted_f1 = float(np.average(f1_per_class, weights=weights)) if sum(weights) > 0 else 0.0

    return macro_f1, weighted_f1, f1_per_class


def compute_cohens_kappa(conf_matrix: np.ndarray) -> float:
    total = np.sum(conf_matrix)
    if total == 0:
        return 0.0
    po = np.trace(conf_matrix) / total
    pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total ** 2)
    return float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 0 else 0.0


@torch.no_grad()
def evaluate_test_set(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_top2_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, top2_indices = torch.topk(outputs, k=2, dim=1)

        all_preds.extend(top2_indices[:, 0].cpu().numpy())
        all_top2_preds.extend(top2_indices.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    avg_loss = running_loss / total_samples

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_top2 = np.array(all_top2_preds)

    accuracy = float(np.mean(y_true == y_pred))
    top2_accuracy = float(np.mean([y_true[i] in y_top2[i] for i in range(total_samples)]))
    
    macro_f1, weighted_f1, f1_per_class = compute_f1_score(y_true, y_pred, config.VGGFACE2.NUM_CLASS)

    conf_matrix = np.zeros((config.VGGFACE2.NUM_CLASS, config.VGGFACE2.NUM_CLASS), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    kappa = compute_cohens_kappa(conf_matrix)

    misclassified_indices = np.where(y_true != y_pred)[0].tolist()
    failure_cases = [
        {"sample_idx": int(idx), "true_label": int(y_true[idx]), "pred_label": int(y_pred[idx])}
        for idx in misclassified_indices[:50]
    ]

    return {
        "test_loss": float(avg_loss),
        "test_accuracy": accuracy,
        "test_top2_accuracy": top2_accuracy,
        "cohens_kappa": kappa,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "per_class_f1": f1_per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "total_misclassifications": len(misclassified_indices),
        "error_rate": float(len(misclassified_indices) / total_samples),
        "failure_cases": failure_cases
    }


def find_timestamp_runs(exp_root: Path) -> List[Path]:
    if not exp_root.exists():
        return []
    return sorted([d for d in exp_root.iterdir() if d.is_dir() and d.name != "statistics"])


def calculate_one_way_anova(pipeline_metrics: Dict[str, List[float]]) -> Dict:
    groups = list(pipeline_metrics.values())
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"f_statistic": None, "p_value": None, "statistically_significant_95": False}

    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "statistically_significant_95": bool(p_val < 0.05) if not np.isnan(p_val) else False
    }


def run_testing_suite():
    exp_root = Path("experiment") / "vggface2"
    run_dirs = find_timestamp_runs(exp_root)

    if not run_dirs:
        print(f"Error: No trained run folders found in {exp_root.resolve()}")
        return

    test_stats_dir = exp_root / "model_evaluation"
    test_stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(run_dirs)} run directories for evaluation:")
    for r in run_dirs:
        print(f" - {r.name}")

    pipelines = generate_pipeline_permutations()
    criterion = config.CRITERION

    aggregated_pipeline_results = defaultdict(lambda: {
        "runs_data": [],
        "accuracies": [],
        "top2_accuracies": [],
        "kappas": [],
        "losses": [],
        "macro_f1s": [],
        "weighted_f1s": [],
        "training_times": [],
        "throughputs": [],
        "error_rates": []
    })

    for run_dir in run_dirs:
        print("\n" + "=" * 60)
        print(f" Evaluating Models in Run: {run_dir.name}")
        print("=" * 60)

        models_dir = run_dir / "trained_model"
        train_stats_dir = run_dir / "training_statistic"

        for p_name, (transform_fn, file_stem) in pipelines.items():
            model_path = models_dir / f"{file_stem}.pt"
            json_train_path = train_stats_dir / f"{file_stem}.json"

            if not model_path.exists():
                print(f"[Skip] Checkpointing file missing for {p_name} in {run_dir.name}")
                continue

            train_time_sec = 0.0
            throughput = 0.0
            if json_train_path.exists():
                with open(json_train_path, "r", encoding="utf-8") as f:
                    train_meta = json.load(f)
                    perf = train_meta.get("performance_metrics", {})
                    train_time_sec = perf.get("total_training_time_sec", 0.0)
                    throughput = perf.get("throughput_samples_per_sec", 0.0)

            raw_test_dataset = load_vggface2_test(transform_fn=transform_fn)
            test_dataset = PreloadedDataset(raw_test_dataset)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            model = CNN(num_class=config.VGGFACE2.NUM_CLASS).to(config.DEVICE)
            dummy_input = torch.zeros(2, *config.VGGFACE2.INPUT_SHAPE).to(config.DEVICE)
            model(dummy_input)

            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

            eval_results = evaluate_test_set(model, test_loader, criterion, config.DEVICE)
            eval_results["training_time_sec"] = train_time_sec
            eval_results["throughput_samples_per_sec"] = throughput

            agg = aggregated_pipeline_results[file_stem]
            agg["runs_data"].append(eval_results)
            agg["accuracies"].append(eval_results["test_accuracy"])
            agg["top2_accuracies"].append(eval_results["test_top2_accuracy"])
            agg["kappas"].append(eval_results["cohens_kappa"])
            agg["losses"].append(eval_results["test_loss"])
            agg["macro_f1s"].append(eval_results["test_macro_f1"])
            agg["weighted_f1s"].append(eval_results["test_weighted_f1"])
            agg["training_times"].append(train_time_sec)
            agg["throughputs"].append(throughput)
            agg["error_rates"].append(eval_results["error_rate"])

            print(f"[{p_name:10s}] Test Acc: {eval_results['test_accuracy']:.4f} | "
                  f"Top-2 Acc: {eval_results['test_top2_accuracy']:.4f} | "
                  f"Kappa: {eval_results['cohens_kappa']:.4f} | "
                  f"Test Loss: {eval_results['test_loss']:.4f} | "
                  f"Macro F1: {eval_results['test_macro_f1']:.4f}")

    summary_report = {
        "total_runs_evaluated": len(run_dirs),
        "run_timestamps": [r.name for r in run_dirs],
        "pipelines": {}
    }

    anova_acc_groups = {}
    anova_loss_groups = {}
    anova_f1_groups = {}
    anova_kappa_groups = {}

    print("\n" + "=" * 60)
    print(" Exporting Pipeline Results & Generating Summary")
    print("=" * 60)

    for p_name, agg in aggregated_pipeline_results.items():
        n_runs = len(agg["runs_data"])
        if n_runs == 0:
            continue

        mean_acc = float(np.mean(agg["accuracies"]))
        std_acc = float(np.std(agg["accuracies"], ddof=1 if n_runs > 1 else 0))

        mean_top2_acc = float(np.mean(agg["top2_accuracies"]))
        std_top2_acc = float(np.std(agg["top2_accuracies"], ddof=1 if n_runs > 1 else 0))

        mean_kappa = float(np.mean(agg["kappas"]))
        std_kappa = float(np.std(agg["kappas"], ddof=1 if n_runs > 1 else 0))

        mean_loss = float(np.mean(agg["losses"]))
        std_loss = float(np.std(agg["losses"], ddof=1 if n_runs > 1 else 0))

        mean_macro_f1 = float(np.mean(agg["macro_f1s"]))
        std_macro_f1 = float(np.std(agg["macro_f1s"], ddof=1 if n_runs > 1 else 0))

        mean_weighted_f1 = float(np.mean(agg["weighted_f1s"]))
        std_weighted_f1 = float(np.std(agg["weighted_f1s"], ddof=1 if n_runs > 1 else 0))

        mean_train_time = float(np.mean(agg["training_times"]))
        std_train_time = float(np.std(agg["training_times"], ddof=1 if n_runs > 1 else 0))

        mean_throughput = float(np.mean(agg["throughputs"]))
        
        cv_accuracy = float((std_acc / mean_acc) * 100) if mean_acc > 0 else 0.0
        efficiency_ratio = float(mean_acc / mean_train_time) if mean_train_time > 0 else 0.0

        anova_acc_groups[p_name] = agg["accuracies"]
        anova_loss_groups[p_name] = agg["losses"]
        anova_f1_groups[p_name] = agg["macro_f1s"]
        anova_kappa_groups[p_name] = agg["kappas"]

        pipeline_json_data = {
            "pipeline_name": p_name,
            "num_independent_runs": n_runs,
            "summary_statistics": {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_top2_accuracy": mean_top2_acc,
                "std_top2_accuracy": std_top2_acc,
                "mean_cohens_kappa": mean_kappa,
                "std_cohens_kappa": std_kappa,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "mean_macro_f1": mean_macro_f1,
                "std_macro_f1": std_macro_f1,
                "mean_weighted_f1": mean_weighted_f1,
                "std_weighted_f1": std_weighted_f1,
                "mean_training_time_sec": mean_train_time,
                "std_training_time_sec": std_train_time,
                "mean_throughput_samples_per_sec": mean_throughput,
                "stability_cv_accuracy_pct": cv_accuracy,
                "efficiency_accuracy_per_sec": efficiency_ratio
            },
            "per_run_details": agg["runs_data"]
        }

        test_file_path = test_stats_dir / f"{p_name}_test.json"
        with open(test_file_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_json_data, f, indent=4)

        summary_report["pipelines"][p_name] = pipeline_json_data["summary_statistics"]

    summary_report["statistical_tests"] = {
        "one_way_anova_accuracy": calculate_one_way_anova(anova_acc_groups),
        "one_way_anova_loss": calculate_one_way_anova(anova_loss_groups),
        "one_way_anova_macro_f1": calculate_one_way_anova(anova_f1_groups),
        "one_way_anova_cohens_kappa": calculate_one_way_anova(anova_kappa_groups)
    }

    rankings = []
    for p_name, stats_meta in summary_report["pipelines"].items():
        acc = stats_meta["mean_accuracy"]
        f1 = stats_meta["mean_macro_f1"]
        kappa = stats_meta["mean_cohens_kappa"]
        eff = stats_meta["efficiency_accuracy_per_sec"]

        rankings.append({
            "pipeline_name": p_name,
            "mean_accuracy": acc,
            "mean_macro_f1": f1,
            "mean_cohens_kappa": kappa,
            "stability_cv_pct": stats_meta["stability_cv_accuracy_pct"],
            "efficiency_ratio": eff
        })

    if rankings:
        max_eff = max(r["efficiency_ratio"] for r in rankings) or 1.0
        for r in rankings:
            norm_acc = r["mean_accuracy"]
            norm_f1 = r["mean_macro_f1"]
            norm_stab = r["stability_cv_pct"]
            norm_eff = r["efficiency_ratio"] / max_eff

            composite_score = (0.40 * norm_acc) + (0.30 * norm_f1) + (0.15 * (1.0 - min(norm_stab, 1.0))) + (0.15 * norm_eff)
            r["composite_score"] = float(round(composite_score, 4))

        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        summary_report["pipeline_rankings"] = rankings

    summary_path = test_stats_dir / "vggface2_summary_test.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print(f"\nSuccessfully saved individual pipeline test files (e.g. p1_test.json) and summary to:")
    print(f" -> {summary_path.resolve()}\n")


if __name__ == "__main__":
    run_testing_suite()