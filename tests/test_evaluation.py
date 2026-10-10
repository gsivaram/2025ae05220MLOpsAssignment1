import matplotlib.pyplot as plt
import numpy as np
import pytest

from heartrisk.evaluation import compute_metrics, plot_confusion, plot_roc


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert all(v == pytest.approx(1.0) for v in m.values())


def test_threshold_changes_hard_metrics_not_auc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.9])
    low, high = compute_metrics(y, p, threshold=0.3), compute_metrics(y, p, threshold=0.5)
    assert low["recall"] == 1.0 and high["recall"] == 0.5
    assert low["roc_auc"] == high["roc_auc"]


def test_no_positive_predictions_does_not_crash():
    m = compute_metrics(np.array([0, 1]), np.array([0.1, 0.2]))
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_plots_return_figures():
    y, p = np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.9])
    for fig in (plot_roc({"m": (y, p)}), plot_confusion(y, p)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
