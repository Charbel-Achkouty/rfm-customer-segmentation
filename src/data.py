"""Data loading, cleaning and demo-data generation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# "Online Retail II" and some Kaggle versions use different column names.
COLUMN_ALIASES = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}
REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw transactions from .xlsx/.xls or .csv and normalise column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for download instructions, "
            "or run with --demo to use synthetic data."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        # Online Retail II ships with several sheets, so read them all.
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only valid purchases and add a TotalPrice column.

    Rules:
      * drop rows without a CustomerID (cannot be attributed to a customer)
      * drop cancellations (invoice numbers starting with 'C')
      * drop non-positive quantities or prices (returns, adjustments)
      * drop exact duplicate rows
    """
    start = len(df)
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df = df.reset_index(drop=True)

    logger.info("Cleaning kept %d of %d rows (%.1f%%)", len(df), start, 100 * len(df) / start)
    return df


def make_demo_data(n_customers: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic retail transactions so the pipeline runs without the real dataset.

    Customers are drawn from five behavioural archetypes. The output deliberately
    includes cancellations and missing IDs so the cleaning step has work to do.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2011-12-09")

    # name, share, mean orders, typical days since last order, mean basket value scale
    archetypes = [
        ("champions", 0.12, 25, 15, 1.6),
        ("loyal", 0.25, 8, 50, 1.0),
        ("new", 0.20, 1.5, 25, 0.8),
        ("at_risk", 0.18, 6, 200, 1.1),
        ("lost", 0.25, 1.5, 300, 0.6),
    ]
    shares = np.array([a[1] for a in archetypes])
    counts = rng.multinomial(n_customers, shares / shares.sum())

    rows = []
    invoice = 500000
    customer_id = 12000
    for (_, _, mean_orders, rec_scale, value), n in zip(archetypes, counts):
        for _ in range(n):
            customer_id += 1
            n_orders = 1 + rng.poisson(max(mean_orders - 1, 0.1))
            recency = min(int(rng.exponential(rec_scale)) + 1, 360)
            span = max(0, min(300, 370 - recency)) if n_orders > 1 else 0
            for _ in range(n_orders):
                invoice += 1
                date = end - pd.Timedelta(days=recency + int(rng.uniform(0, span)))
                date += pd.Timedelta(hours=int(rng.integers(8, 18)))
                for _ in range(int(rng.integers(1, 6))):
                    rows.append(
                        (
                            f"{invoice}",
                            f"P{int(rng.integers(1000, 1300))}",
                            int(rng.integers(1, 12)),
                            date,
                            round(float(rng.lognormal(1.2, 0.6)) * value, 2) + 0.05,
                            customer_id,
                            "United Kingdom",
                        )
                    )

    df = pd.DataFrame(
        rows,
        columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
    )

    # add realistic dirt
    dirty = df.sample(frac=0.03, random_state=seed).copy()
    dirty["InvoiceNo"] = "C" + dirty["InvoiceNo"]
    dirty["Quantity"] = -dirty["Quantity"]
    guest = df.sample(frac=0.02, random_state=seed + 1).copy()
    guest["CustomerID"] = np.nan
    df = pd.concat([df, dirty, guest], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
