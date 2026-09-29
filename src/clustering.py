"""Scaling, K-means model selection, and business-friendly segment naming."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]

SEGMENT_ACTIONS = {
    "Champions": "Reward them: loyalty perks, early access to new products, ask for reviews and referrals.",
    "Loyal Customers": "Upsell and cross-sell; invite them into a loyalty programme.",
    "New / Promising": "Nurture with a welcome series and a second-purchase incentive.",
    "At Risk": "Win-back campaign: personalised offer and a 'we miss you' message.",
    "Lost / Hibernating": "Low-cost reactivation only (e.g. one discount email); do not overspend.",
    "Needs Attention": "Test targeted promotions to see what re-engages them.",
}


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p to tame heavy right skew, then standardise so K-means treats R, F, M equally."""
    logged = np.log1p(rfm[FEATURES])
    scaler = StandardScaler()
    return scaler.fit_transform(logged), scaler


def evaluate_k(X: np.ndarray, k_range=range(2, 11), seed: int = 42) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each candidate k."""
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sample = 10000 if len(X) > 10000 else None
        sil = silhouette_score(X, model.labels_, sample_size=sample, random_state=seed)
        records.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pd.DataFrame(records)


def choose_k(metrics: pd.DataFrame, min_k: int = 3, max_k: int = 6) -> int:
    """Pick the k with the best silhouette inside a business-usable range.

    k=2 often wins on silhouette but only splits 'good' from 'bad' customers,
    which is too coarse to act on, so the search starts at min_k.
    """
    window = metrics[(metrics["k"] >= min_k) & (metrics["k"] <= max_k)]
    return int(window.loc[window["silhouette"].idxmax(), "k"])


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def _name_segment(r: float, f: float, m: float) -> str:
    """Rule-based name from a cluster centre in standardised space.

    r is flipped so that higher = more recent = better.
    """
    if r > 0 and f > 0.5 and m > 0.5:
        return "Champions"
    if r <= 0 and f > 0.25 and m > 0.25:
        return "At Risk"
    if r > 0.5 and f < 0:
        return "New / Promising"
    if r < -0.5 and f < 0 and m < 0:
        return "Lost / Hibernating"
    if r > 0 and f > 0:
        return "Loyal Customers"
    return "Needs Attention"


def name_segments(model: KMeans) -> dict[int, str]:
    """Map every cluster id to a unique, human-readable segment name."""
    names: dict[int, str] = {}
    for cid, (rec, freq, mon) in enumerate(model.cluster_centers_):
        names[cid] = _name_segment(-rec, freq, mon)

    # If two clusters received the same name, number them (best monetary first).
    for name in set(names.values()):
        ids = [c for c, n in names.items() if n == name]
        if len(ids) > 1:
            ids.sort(key=lambda c: -model.cluster_centers_[c][2])
            for i, c in enumerate(ids, start=1):
                names[c] = f"{name} {i}"
    return names


def summarise_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: size, average RFM, revenue share, recommended action."""
    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("Recency", "size"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(1)
    )
    summary["Pct_Customers"] = (100 * summary["Customers"] / summary["Customers"].sum()).round(1)
    summary["Pct_Revenue"] = (100 * summary["Total_Revenue"] / summary["Total_Revenue"].sum()).round(1)
    summary["Action"] = [
        SEGMENT_ACTIONS.get(name.rsplit(" ", 1)[0] if name[-1].isdigit() else name, SEGMENT_ACTIONS["Needs Attention"])
        for name in summary.index
    ]
    return summary.sort_values("Total_Revenue", ascending=False)
