"""RFM feature engineering."""
from __future__ import annotations

import pandas as pd


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate cleaned transactions to one row per customer.

    Recency   - days between the customer's last purchase and the snapshot date
    Frequency - number of distinct invoices
    Monetary  - total money spent
    """
    if snapshot_date is None:
        # one day after the last transaction, so the most recent buyer has Recency = 1
        snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    grouped = df.groupby("CustomerID")
    rfm = pd.DataFrame(
        {
            # normalise to midnight so the time of day cannot shave a day off Recency
            "Recency": (snapshot_date - grouped["InvoiceDate"].max().dt.normalize()).dt.days,
            "Frequency": grouped["InvoiceNo"].nunique(),
            "Monetary": grouped["TotalPrice"].sum().round(2),
        }
    )
    return rfm


def add_rfm_scores(rfm: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Add classic 1-5 quintile scores (5 = best) as an interpretable baseline.

    Frequency is ranked first because many customers share the same order count,
    which would otherwise make the quantile edges collide.
    """
    out = rfm.copy()
    out["R_score"] = pd.qcut(out["Recency"].rank(method="first"), bins, labels=range(bins, 0, -1)).astype(int)
    out["F_score"] = pd.qcut(out["Frequency"].rank(method="first"), bins, labels=range(1, bins + 1)).astype(int)
    out["M_score"] = pd.qcut(out["Monetary"].rank(method="first"), bins, labels=range(1, bins + 1)).astype(int)
    out["RFM_Score"] = out["R_score"].astype(str) + out["F_score"].astype(str) + out["M_score"].astype(str)
    return out
