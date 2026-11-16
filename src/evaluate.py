"""
IntelliSentiment AI — Model Evaluation Module
Generates confusion matrices, ROC curves and a comprehensive report.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

STATIC_IMAGES = Path(__file__).parent.parent / "static" / "images"
STATIC_IMAGES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


# ------------------------------------------------------------------ #
#  Core Metric Computation
# ------------------------------------------------------------------ #

def compute_metrics(y_true, y_pred, y_proba=None, label_classes=None) -> dict:
    """
    Compute a full suite of classification metrics.

    Returns:
        dict with accuracy, precision, recall, f1 (macro & weighted),
        roc_auc (if proba available), and per-class report dict.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=label_classes,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "classification_report": report,
        "roc_auc": None,
    }

    if y_proba is not None:
        try:
            dummies = pd.get_dummies(pd.Series(y_true)).values
            if dummies.shape[1] == y_proba.shape[1]:
                metrics["roc_auc"] = round(
                    roc_auc_score(dummies, y_proba, multi_class="ovr", average="macro"), 4
                )
        except Exception as e:
            logger.warning("ROC-AUC computation failed: %s", e)

    return metrics


# ------------------------------------------------------------------ #
#  Confusion Matrix Plot
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    y_true,
    y_pred,
    label_classes: Optional[List[str]] = None,
    model_name: str = "Model",
    normalize: bool = True,
) -> Path:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        normalize: If True, show proportions (0–1) instead of raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=label_classes)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
        title_suffix = " (Normalised)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = ""

    classes = label_classes or list(range(cm.shape[0]))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}{title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    out = STATIC_IMAGES / f"confusion_matrix_{safe_name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out)
    return out


# ------------------------------------------------------------------ #
#  Model Comparison Chart
# ------------------------------------------------------------------ #

def plot_model_comparison(all_metrics: Dict[str, dict]) -> Path:
    """
    Grouped bar chart comparing Accuracy, Macro F1, Precision and Recall
    across all trained models.
    """
    metric_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Accuracy", "Macro F1", "Precision", "Recall"]
    model_names = list(all_metrics.keys())

    x = np.arange(len(metric_keys))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model, color) in enumerate(zip(model_names, colors)):
        vals = [all_metrics[model].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out = STATIC_IMAGES / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Model comparison chart saved: %s", out)
    return out


# ------------------------------------------------------------------ #
#  Full Evaluation Runner
# ------------------------------------------------------------------ #

def evaluate_all_models(training_results: dict) -> dict:
    """
    Generate evaluation charts and store metrics for all trained models.

    Args:
        training_results: Output dict from ``src.train.train_models``.

    Returns:
        Augmented metrics dict with chart paths added.
    """
    models       = training_results["models"]
    metrics      = training_results["metrics"]
    X_test       = training_results["X_test"]
    y_test       = training_results["y_test"]
    label_classes = training_results.get("label_classes")
    best_name    = training_results["best_model_name"]

    eval_results = {}

    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass

        m = compute_metrics(y_test, y_pred, y_proba, label_classes)
        m["confusion_matrix_path"] = str(
            plot_confusion_matrix(y_test, y_pred, label_classes, model_name=name)
        )
        # Merge with training-time metrics
        m.update({k: v for k, v in metrics[name].items() if k not in m})
        eval_results[name] = m

    # Comparison chart
    comparison_path = plot_model_comparison(eval_results)
    for name in eval_results:
        eval_results[name]["comparison_chart_path"] = str(comparison_path)

    logger.info("Evaluation complete for %d models.", len(eval_results))
    return eval_results


def print_evaluation_report(eval_results: dict, best_name: str) -> None:
    """Print a formatted evaluation report to stdout."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  IntelliSentiment AI — Model Evaluation Report")
    print(sep)
    for name, m in eval_results.items():
        marker = " ★ BEST" if name == best_name else ""
        print(f"\n  {name}{marker}")
        print(f"    Accuracy        : {m['accuracy']:.4f}")
        print(f"    Macro Precision : {m['macro_precision']:.4f}")
        print(f"    Macro Recall    : {m['macro_recall']:.4f}")
        print(f"    Macro F1-Score  : {m['macro_f1']:.4f}")
        print(f"    Weighted F1     : {m['weighted_f1']:.4f}")
        if m.get("roc_auc"):
            print(f"    ROC-AUC (macro) : {m['roc_auc']:.4f}")
        if m.get("cv_f1"):
            print(f"    CV F1 (val)     : {m['cv_f1']:.4f}")
    print(f"\n{sep}")
    print(f"  Best Model Selected: {best_name}")
    print(f"  Selection Criterion: Highest Macro F1-Score")
    print(sep)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import generate_synthetic_dataset
    from src.preprocessing import preprocess_dataframe
    from src.train import train_models

    df = generate_synthetic_dataset(800)
    df = preprocess_dataframe(df)
    training_results = train_models(df)
    eval_results = evaluate_all_models(training_results)
    print_evaluation_report(eval_results, training_results["best_model_name"])

