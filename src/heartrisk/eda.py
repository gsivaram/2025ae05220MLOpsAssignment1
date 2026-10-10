"""EDA plotting helpers. Every function returns a Figure and optionally saves it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from heartrisk import config  # noqa: E402

# Validated categorical slots 1 and 2 (blue / orange) for the two classes.
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_NAMES = {0: "No disease", 1: "Disease"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a52a2a"]
)

UNITS = {
    "age": "years",
    "trestbps": "resting BP, mm Hg",
    "chol": "serum cholesterol, mg/dl",
    "thalach": "max heart rate, bpm",
    "oldpeak": "ST depression, mm",
    "ca": "major vessels coloured",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, name: str | None, out_dir: Path) -> plt.Figure:
    if name:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png")
    return fig


def plot_class_balance(df: pd.DataFrame, name: str | None = "01_class_balance",
                       out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    for i, (cls, n) in enumerate(counts.items()):
        ax.barh(i, n, color=CLASS_COLORS[cls], height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.text(n + total * 0.01, i, f"{n}  ({n / total:.0%})", va="center", color=INK)
    ax.set_yticks(range(len(counts)), [CLASS_NAMES[c] for c in counts.index])
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("patients")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Target is close to balanced ({total} patients)")
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_numeric_histograms(df: pd.DataFrame, name: str | None = "02_numeric_histograms",
                            out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), cols, strict=True):
        if col == "ca":
            # Discrete 0-3 count: grouped bars read better than overlapping bins.
            counts = df.groupby([col, config.TARGET]).size().unstack(fill_value=0)
            x = counts.index.to_numpy()
            for offset, cls in ((-0.2, 0), (0.2, 1)):
                ax.bar(x + offset, counts[cls], width=0.4, color=CLASS_COLORS[cls],
                       edgecolor=SURFACE, linewidth=2, label=CLASS_NAMES[cls])
            ax.set_xticks(x, [str(int(v)) for v in x])
        else:
            bins = np.histogram_bin_edges(df[col].dropna(), bins=18)
            for cls in (0, 1):
                values = df.loc[df[config.TARGET] == cls, col].dropna()
                ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.18, color=CLASS_COLORS[cls])
                ax.hist(values, bins=bins, histtype="step", linewidth=2, color=CLASS_COLORS[cls],
                        label=CLASS_NAMES[cls])
        ax.set_title(col)
        ax.set_xlabel(UNITS[col])
        ax.set_ylabel("patients")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Numeric feature distributions by outcome", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_numeric_boxplots(df: pd.DataFrame, name: str | None = "03_numeric_boxplots",
                          out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    # ca is a 0-3 count, already shown as grouped bars in the histogram figure.
    cols = [c for c in config.NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 3.4))
    for ax, col in zip(axes, cols, strict=True):
        groups = [df.loc[df[config.TARGET] == c, col].dropna() for c in (0, 1)]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                        medianprops={"color": INK, "linewidth": 1.4},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_2,
                                    "markeredgecolor": "none", "alpha": 0.6})
        for patch, c in zip(bp["boxes"], (0, 1), strict=True):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK_2)
        ax.set_xticks([1, 2], ["No", "Yes"])
        ax.set_title(col)
        ax.grid(axis="x", visible=False)
    fig.supxlabel("heart disease", color=INK_2, fontsize=9.5)
    fig.suptitle("Spread and outliers by outcome", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_categorical_rates(df: pd.DataFrame, name: str | None = "04_categorical_disease_rate",
                           out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Disease rate per category, against the overall prevalence line."""
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES
    labels = {**config.CODE_LABELS,
              "fbs": {0: "≤120 mg/dl", 1: ">120 mg/dl"},
              "exang": {0: "no", 1: "yes"}}
    prevalence = df[config.TARGET].mean()

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    axes = axes.ravel()
    for ax, col in zip(axes, cols, strict=False):
        stats = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        names = [labels.get(col, {}).get(int(k), str(int(k))) for k in stats.index]
        y = np.arange(len(stats))
        ax.barh(y, stats["mean"], color=CLASS_COLORS[1], height=0.6, edgecolor=SURFACE, linewidth=2)
        for yi, (rate, n) in zip(y, stats.itertuples(index=False), strict=True):
            ax.text(1.04, yi, f"{rate:.0%}", va="center", fontsize=8.5, color=INK, fontweight="bold")
            ax.text(1.04, yi + 0.22, f"n={n}", va="top", fontsize=7.5, color=INK_2)
        ax.axvline(prevalence, color=INK_2, linestyle="--", linewidth=1)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(f"Disease rate per category (dashed line = overall {prevalence:.0%})",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_correlation_heatmap(df: pd.DataFrame, name: str | None = "05_correlation_heatmap",
                             out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Spearman correlation (robust to the ordinal codes) as a lower triangle."""
    corr = df[config.FEATURES + [config.TARGET]].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    shown = corr.mask(mask)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1)
    n = len(corr)
    for i in range(n):
        for j in range(i + 1):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Spearman ρ", color=INK_2)
    ax.set_title("Feature correlations (bottom row = correlation with target)")
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_age_vs_maxhr(df: pd.DataFrame, name: str | None = "06_age_vs_thalach",
                      out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cls in (0, 1):
        sub = df[df[config.TARGET] == cls]
        ax.scatter(sub["age"], sub["thalach"], s=34, color=CLASS_COLORS[cls], alpha=0.8,
                   edgecolor=SURFACE, linewidth=1.2, label=CLASS_NAMES[cls])
    ax.plot([28, 78], [220 - 28, 220 - 78], color=INK_2, linestyle=":", linewidth=1.2)
    ax.text(70, 220 - 70 + 6, "220 − age", color=INK_2, fontsize=8.5)
    ax.set_xlabel(UNITS["age"])
    ax.set_ylabel(UNITS["thalach"])
    ax.legend(loc="lower left")
    ax.set_title("Patients with disease reach lower peak heart rates at every age")
    fig.tight_layout()
    return _save(fig, name, out_dir)


ALL_PLOTS = [
    plot_class_balance,
    plot_numeric_histograms,
    plot_numeric_boxplots,
    plot_categorical_rates,
    plot_correlation_heatmap,
    plot_age_vs_maxhr,
]


def generate_all(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    apply_style()
    paths = []
    for fn in ALL_PLOTS:
        fig = fn(df, out_dir=out_dir)
        paths.append(out_dir / f"{fn.__defaults__[0]}.png")
        plt.close(fig)
    return paths
