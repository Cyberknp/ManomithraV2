"""
Quality Review — Accuracy Metrics Calculator
=============================================
Evaluates the trained emotion-detection Keras models against the test set and
produces a full quality report including:

  • Overall accuracy, loss
  • Per-class precision, recall, F1-score
  • Confusion matrix (printed + saved as image)
  • Training-history curves (if a history file is available)
  • Side-by-side comparison of `emotion_model.keras` vs `best_model.keras`

Usage
-----
    python quality_review.py                 # evaluate both models
    python quality_review.py --model best    # evaluate only best_model.keras
    python quality_review.py --model emotion # evaluate only emotion_model.keras
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────────

IMG_SIZE = 48
BATCH_SIZE = 64

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_DIR = PROJECT_DIR / "facial" / "train" / "train"
TEST_DIR = PROJECT_DIR / "facial" / "test" / "test"

MODEL_PATHS = {
    "best": BASE_DIR / "best_model.keras",
    "emotion": BASE_DIR / "emotion_model.keras",
}

REPORT_DIR = BASE_DIR / "reports"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ensure_report_dir():
    """Create the reports directory if it doesn't exist."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_class_names():
    """
    Derive the canonical class-name list from the *training* directory so that
    the evaluation labels are aligned with what the model was trained on.
    Falls back to the test directory if the training one is unavailable.
    """
    for candidate in (TRAIN_DIR, TEST_DIR):
        if candidate.exists():
            names = sorted(
                d.name
                for d in candidate.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            if names:
                return names

    # Fallback: standard FER-2013 classes
    return ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _load_test_dataset(class_names):
    """
    Build a normalised tf.data.Dataset from the test directory.

    If the test directory contains fewer classes than the training set the
    one-hot labels are remapped/padded to match the full class space (same
    logic used during training in facedetection.py).
    """
    if not TEST_DIR.exists():
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        sys.exit(1)

    raw = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
    )

    test_class_names = raw.class_names
    num_classes = len(class_names)

    # Remap labels when class sets differ
    if test_class_names != class_names:
        idx_map = []
        for c in test_class_names:
            if c in class_names:
                idx_map.append(class_names.index(c))
            else:
                idx_map.append(-1)  # class not in training set — will be dropped

        def _remap(images, labels):
            new = tf.zeros((tf.shape(labels)[0], num_classes), dtype=tf.float32)
            for test_i, train_i in enumerate(idx_map):
                if train_i < 0:
                    continue
                col = labels[:, test_i : test_i + 1]
                padding = [[0, 0], [train_i, num_classes - train_i - 1]]
                new = new + tf.pad(col, padding)
            return images, new

        raw = raw.map(_remap, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalise pixel values to [0, 1]
    raw = raw.map(
        lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return raw


# ── Core Metrics ───────────────────────────────────────────────────────────────


class ModelMetricsCalculator:
    """Calculates and stores accuracy metrics for a single Keras model."""

    def __init__(self, model_name: str, model_path: Path, class_names: list[str]):
        self.model_name = model_name
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Populated after evaluate()
        self.loss: float | None = None
        self.accuracy: float | None = None
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.confusion_matrix: np.ndarray | None = None
        self.per_class: dict | None = None
        self.weighted_avg: dict | None = None
        self.macro_avg: dict | None = None

    # ── public API ─────────────────────────────────────────────────────────

    def evaluate(self, dataset: tf.data.Dataset) -> "ModelMetricsCalculator":
        """Run the full evaluation pipeline."""
        print(f"\n{'=' * 70}")
        print(f"  Evaluating: {self.model_name}  ({self.model_path.name})")
        print(f"{'=' * 70}")

        if not self.model_path.exists():
            print(f"  [SKIP] Model file not found: {self.model_path}")
            return self

        model = tf.keras.models.load_model(str(self.model_path))

        # 1. Built-in evaluate for loss & accuracy
        self.loss, self.accuracy = model.evaluate(dataset, verbose=0)

        # 2. Gather predictions & ground-truth
        all_y_true, all_y_pred = [], []
        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            all_y_true.append(np.argmax(labels.numpy(), axis=1))
            all_y_pred.append(np.argmax(preds, axis=1))

        self.y_true = np.concatenate(all_y_true)
        self.y_pred = np.concatenate(all_y_pred)

        # 3. Confusion matrix
        self.confusion_matrix = self._compute_confusion_matrix()

        # 4. Per-class precision / recall / F1
        self.per_class, self.macro_avg, self.weighted_avg = (
            self._compute_classification_report()
        )

        return self

    def print_report(self):
        """Pretty-print the full metrics report to stdout."""
        if self.accuracy is None:
            return

        print(f"\n── Summary ({'─' * 50})")
        print(f"  Model        : {self.model_name}")
        print(f"  Test Loss    : {self.loss:.4f}")
        print(f"  Test Accuracy: {self.accuracy:.4f}  ({self.accuracy * 100:.2f}%)")
        total = len(self.y_true)
        correct = int(np.sum(self.y_true == self.y_pred))
        print(f"  Correct / Total: {correct} / {total}")

        # Per-class table
        print(f"\n── Per-Class Metrics ({'─' * 40})")
        header = f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
        print(header)
        print(f"  {'─' * 54}")
        for cls_name in self.class_names:
            m = self.per_class.get(cls_name)
            if m is None:
                continue
            print(
                f"  {cls_name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['support']:>10d}"
            )
        print(f"  {'─' * 54}")
        ma = self.macro_avg
        wa = self.weighted_avg
        print(
            f"  {'Macro Avg':<12} {ma['precision']:>10.4f} {ma['recall']:>10.4f} "
            f"{ma['f1']:>10.4f} {total:>10d}"
        )
        print(
            f"  {'Weighted Avg':<12} {wa['precision']:>10.4f} {wa['recall']:>10.4f} "
            f"{wa['f1']:>10.4f} {total:>10d}"
        )

        # Confusion matrix
        print(f"\n── Confusion Matrix ({'─' * 40})")
        col_w = max(7, max(len(c) for c in self.class_names) + 1)
        row_header = " " * (col_w + 2)
        for c in self.class_names:
            row_header += f"{c:>{col_w}}"
        print(f"  Predicted →")
        print(f"  {row_header}")
        for i, cls_name in enumerate(self.class_names):
            row = f"  {cls_name:<{col_w + 2}}"
            for j in range(self.num_classes):
                row += f"{self.confusion_matrix[i, j]:>{col_w}d}"
            print(row)

    def save_report(self):
        """Persist the metrics as a JSON file under reports/."""
        if self.accuracy is None:
            return
        _ensure_report_dir()
        report = {
            "model": self.model_name,
            "model_file": self.model_path.name,
            "test_loss": round(self.loss, 6),
            "test_accuracy": round(self.accuracy, 6),
            "total_samples": int(len(self.y_true)),
            "correct_predictions": int(np.sum(self.y_true == self.y_pred)),
            "class_names": self.class_names,
            "per_class": self.per_class,
            "macro_avg": self.macro_avg,
            "weighted_avg": self.weighted_avg,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
        out = REPORT_DIR / f"{self.model_name}_metrics.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✓ Report saved to {out}")

    def save_confusion_matrix_plot(self):
        """Save a confusion-matrix heatmap as a PNG image."""
        if self.confusion_matrix is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [INFO] matplotlib not installed — skipping confusion matrix plot.")
            return

        _ensure_report_dir()

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            self.confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues
        )
        ax.set_title(f"Confusion Matrix — {self.model_name}", fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.8)

        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.class_names, fontsize=9)

        # Annotate cells
        thresh = self.confusion_matrix.max() / 2.0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                val = self.confusion_matrix[i, j]
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8,
                )

        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        fig.tight_layout()
        out = REPORT_DIR / f"{self.model_name}_confusion_matrix.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ Confusion matrix plot saved to {out}")

    def save_per_class_bar_chart(self):
        """Save a grouped bar chart of precision, recall, F1 per class."""
        if self.per_class is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        _ensure_report_dir()

        labels = [c for c in self.class_names if c in self.per_class]
        precision = [self.per_class[c]["precision"] for c in labels]
        recall = [self.per_class[c]["recall"] for c in labels]
        f1 = [self.per_class[c]["f1"] for c in labels]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, precision, width, label="Precision", color="#4c72b0")
        ax.bar(x, recall, width, label="Recall", color="#55a868")
        ax.bar(x + width, f1, width, label="F1-Score", color="#c44e52")

        ax.set_ylabel("Score")
        ax.set_title(f"Per-Class Metrics — {self.model_name}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = REPORT_DIR / f"{self.model_name}_per_class_metrics.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ Per-class bar chart saved to {out}")

    # ── private helpers ────────────────────────────────────────────────────

    def _compute_confusion_matrix(self) -> np.ndarray:
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            cm[t, p] += 1
        return cm

    def _compute_classification_report(self):
        per_class = {}
        macro_p, macro_r, macro_f = 0.0, 0.0, 0.0
        weighted_p, weighted_r, weighted_f = 0.0, 0.0, 0.0
        total_support = len(self.y_true)
        valid_classes = 0

        for idx, cls_name in enumerate(self.class_names):
            tp = int(self.confusion_matrix[idx, idx])
            fp = int(self.confusion_matrix[:, idx].sum() - tp)
            fn = int(self.confusion_matrix[idx, :].sum() - tp)
            support = int(self.confusion_matrix[idx, :].sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[cls_name] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

            if support > 0:
                valid_classes += 1
                macro_p += precision
                macro_r += recall
                macro_f += f1
                weighted_p += precision * support
                weighted_r += recall * support
                weighted_f += f1 * support

        macro_avg = {
            "precision": round(macro_p / valid_classes, 6) if valid_classes else 0.0,
            "recall": round(macro_r / valid_classes, 6) if valid_classes else 0.0,
            "f1": round(macro_f / valid_classes, 6) if valid_classes else 0.0,
        }
        weighted_avg = {
            "precision": round(weighted_p / total_support, 6) if total_support else 0.0,
            "recall": round(weighted_r / total_support, 6) if total_support else 0.0,
            "f1": round(weighted_f / total_support, 6) if total_support else 0.0,
        }
        return per_class, macro_avg, weighted_avg


# ── Comparison ─────────────────────────────────────────────────────────────────


def compare_models(calculators: list[ModelMetricsCalculator]):
    """Print a side-by-side comparison table for evaluated models."""
    evaluated = [c for c in calculators if c.accuracy is not None]
    if len(evaluated) < 2:
        return

    print(f"\n{'=' * 70}")
    print("  Model Comparison")
    print(f"{'=' * 70}")

    header = f"  {'Metric':<20}"
    for c in evaluated:
        header += f" {c.model_name:>20}"
    print(header)
    print(f"  {'─' * 20 + ' ' + ' '.join(['─' * 20] * len(evaluated))}")

    rows = [
        ("Test Loss", [f"{c.loss:.4f}" for c in evaluated]),
        ("Test Accuracy", [f"{c.accuracy:.4f}" for c in evaluated]),
        ("Accuracy (%)", [f"{c.accuracy * 100:.2f}%" for c in evaluated]),
        ("Macro Precision", [f"{c.macro_avg['precision']:.4f}" for c in evaluated]),
        ("Macro Recall", [f"{c.macro_avg['recall']:.4f}" for c in evaluated]),
        ("Macro F1", [f"{c.macro_avg['f1']:.4f}" for c in evaluated]),
        ("Weighted F1", [f"{c.weighted_avg['f1']:.4f}" for c in evaluated]),
    ]

    for label, values in rows:
        row = f"  {label:<20}"
        for v in values:
            row += f" {v:>20}"
        print(row)

    # Determine winner
    best = max(evaluated, key=lambda c: c.accuracy)
    print(
        f"\n  ★ Best model by accuracy: {best.model_name} ({best.accuracy * 100:.2f}%)"
    )


def save_comparison_report(calculators: list[ModelMetricsCalculator]):
    """Save a JSON comparison summary."""
    evaluated = [c for c in calculators if c.accuracy is not None]
    if not evaluated:
        return

    _ensure_report_dir()
    comparison = {
        "models": [],
        "best_by_accuracy": None,
    }

    best_acc = -1.0
    for c in evaluated:
        entry = {
            "name": c.model_name,
            "file": c.model_path.name,
            "test_loss": round(c.loss, 6),
            "test_accuracy": round(c.accuracy, 6),
            "macro_f1": c.macro_avg["f1"],
            "weighted_f1": c.weighted_avg["f1"],
        }
        comparison["models"].append(entry)
        if c.accuracy > best_acc:
            best_acc = c.accuracy
            comparison["best_by_accuracy"] = c.model_name

    out = REPORT_DIR / "model_comparison.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  ✓ Comparison report saved to {out}")


# ── Training History Plots ─────────────────────────────────────────────────────


def plot_training_history(history_path: Path | None = None):
    """
    Plot accuracy & loss curves from a training history JSON.

    The history file is expected to be a JSON dict with keys like
    'accuracy', 'val_accuracy', 'loss', 'val_loss' — each mapping to a
    list of per-epoch values.  If no file is supplied, this step is skipped.
    """
    if history_path is None:
        # Try to auto-detect
        candidates = list(BASE_DIR.glob("*history*.json"))
        if not candidates:
            return
        history_path = candidates[0]

    if not history_path.exists():
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with open(history_path) as f:
        history = json.load(f)

    _ensure_report_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    if "accuracy" in history:
        ax1.plot(history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history:
        ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    if "loss" in history:
        ax2.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = REPORT_DIR / "training_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  ✓ Training history plot saved to {out}")


# ── CLI Entrypoint ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained emotion-detection models and generate quality reports."
    )
    parser.add_argument(
        "--model",
        choices=["best", "emotion", "all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to a training history JSON file (optional).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating matplotlib plots.",
    )
    args = parser.parse_args()

    # Resolve which models to evaluate
    if args.model == "all":
        model_keys = list(MODEL_PATHS.keys())
    else:
        model_keys = [args.model]

    class_names = _load_class_names()
    print(f"Classes ({len(class_names)}): {class_names}")

    dataset = _load_test_dataset(class_names)

    calculators: list[ModelMetricsCalculator] = []

    for key in model_keys:
        calc = ModelMetricsCalculator(key, MODEL_PATHS[key], class_names)
        calc.evaluate(dataset)
        calc.print_report()
        calc.save_report()
        if not args.no_plots:
            calc.save_confusion_matrix_plot()
            calc.save_per_class_bar_chart()
        calculators.append(calc)

    # Comparison (only meaningful when both models are evaluated)
    if len(calculators) > 1:
        compare_models(calculators)
        save_comparison_report(calculators)

    # Training history plots
    if not args.no_plots:
        history_path = Path(args.history) if args.history else None
        plot_training_history(history_path)

    print(f"\n{'=' * 70}")
    print("  Quality review complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
