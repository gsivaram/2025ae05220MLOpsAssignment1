"""Metrics and diagnostic plots for trained classifiers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heartrisk import config
from heartrisk.eda import INK, INK_2, SURFACE, apply_style

# Categorical slots 1-3 of the validated palette: one per candidate model.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

# Scorers used by cross-validation; keys become MLflow metric names.
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
    }


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "ROC curve — test set") -> plt.Figure:
    """curves: {label: (y_true, proba)}"""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1)
    for color, (label, (y, p)) in zip(MODEL_COLORS, curves.items(), strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  AUC {roc_auc_score(y, p):.3f}")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "Precision–recall — test set") -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for color, (label, (y, p)) in zip(MODEL_COLORS, curves.items(), strict=False):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label}  AP {average_precision_score(y, p):.3f}")
    base = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base, color=INK_2, linestyle=":", linewidth=1)
    ax.text(0.02, base + 0.02, f"prevalence {base:.0%}", color=INK_2, fontsize=8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, proba, threshold: float = 0.5, title: str = "Confusion matrix — test set") -> plt.Figure:
    apply_style()
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap=SEQUENTIAL_BLUE, vmin=0)
    labels = [["true negative", "false positive"], ["false negative", "true positive"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i - 0.08, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(j, i + 0.22, labels[i][j], ha="center", va="center", fontsize=8,
                    color="white" if dark else INK_2)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def permutation_importances(pipeline, X: pd.DataFrame, y, n_repeats: int = 30) -> pd.DataFrame:
    """Model-agnostic importance on the *raw* clinical inputs (drop in ROC-AUC when shuffled)."""
    result = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "mean": result.importances_mean, "std": result.importances_std})
            .sort_values("mean", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, title: str = "Permutation importance (drop in test ROC-AUC)") -> plt.Figure:
    apply_style()
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    y = np.arange(len(imp))
    ax.barh(y, imp["mean"], xerr=imp["std"], color=MODEL_COLORS[0], height=0.6, edgecolor=SURFACE,
            linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, imp["feature"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean decrease in ROC-AUC")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary: pd.DataFrame, metric: str = "roc_auc") -> plt.Figure:
    """summary: index = model name, columns include cv_<metric>_mean, cv_<metric>_std, test_<metric>."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.75 * len(summary)))
    y = np.arange(len(summary))
    means, stds = summary[f"cv_{metric}_mean"], summary[f"cv_{metric}_std"]
    for i, color in enumerate(MODEL_COLORS[: len(summary)]):
        ax.errorbar(means.iloc[i], i, xerr=stds.iloc[i], fmt="o", color=color, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=2, capsize=0)
        ax.scatter(summary[f"test_{metric}"].iloc[i], i, marker="D", s=40, facecolor=SURFACE,
                   edgecolor=color, linewidth=2, zorder=3)
        ax.text(1.02, i, f"CV {means.iloc[i]:.3f} ± {stds.iloc[i]:.3f}\ntest {summary[f'test_{metric}'].iloc[i]:.3f}",
                va="center", ha="left", fontsize=8.5, color=INK, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(y, summary.index)
    ax.set_ylim(-0.6, len(summary) - 0.4)
    lo = min(means.min() - stds.max(), summary[f"test_{metric}"].min()) - 0.02
    ax.set_xlim(lo, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{metric}   (● 5-fold CV mean ± 1 sd   ◇ held-out test)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    return fig
