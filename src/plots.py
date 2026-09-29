"""All figures. Each function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rfm_distributions(rfm: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, col in enumerate(["Recency", "Frequency", "Monetary"]):
        sns.histplot(rfm[col], bins=40, ax=axes[0, j], color="#4C72B0")
        axes[0, j].set_title(f"{col} (raw)")
        sns.histplot(np.log1p(rfm[col]), bins=40, ax=axes[1, j], color="#55A868")
        axes[1, j].set_title(f"{col} (log1p)")
    fig.suptitle("RFM distributions: the log transform fixes the heavy right skew", y=1.02)
    return _save(fig, out_dir, "01_rfm_distributions.png")


def plot_k_selection(metrics: pd.DataFrame, chosen_k: int, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o")
    axes[0].set_title("Elbow method (inertia)")
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#C44E52")
    axes[1].set_title("Silhouette score")
    for ax in axes:
        ax.axvline(chosen_k, ls="--", color="grey")
        ax.set_xlabel("Number of clusters (k)")
    return _save(fig, out_dir, "02_choosing_k.png")


def plot_clusters_pca(X: np.ndarray, labels: pd.Series, out_dir: Path) -> Path:
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels.values, palette="tab10", s=25, alpha=0.7, ax=ax)
    ax.set_title("Customer segments (PCA projection of scaled RFM)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, out_dir, "03_segments_pca.png")


def plot_segment_profile(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of each segment's average R, F, M relative to the overall mean."""
    cols = ["Avg_Recency", "Avg_Frequency", "Avg_Monetary"]
    relative = summary[cols] / summary[cols].mean()
    relative.columns = ["Recency", "Frequency", "Monetary"]
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(summary) + 2))
    sns.heatmap(relative, annot=True, fmt=".2f", cmap="RdYlGn", center=1, ax=ax)
    ax.set_title("Segment profile (1.0 = average; low Recency is good)")
    ax.set_ylabel("")
    return _save(fig, out_dir, "04_segment_profile.png")


def plot_size_vs_revenue(summary: pd.DataFrame, out_dir: Path) -> Path:
    data = summary[["Pct_Customers", "Pct_Revenue"]].reset_index().melt(
        id_vars="Segment", var_name="Metric", value_name="Percent"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="Segment", y="Percent", hue="Metric", ax=ax)
    ax.set_title("Share of customers vs share of revenue")
    ax.set_ylabel("%")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_dir, "05_size_vs_revenue.png")
