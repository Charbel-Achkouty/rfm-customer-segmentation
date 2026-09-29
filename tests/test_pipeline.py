import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.clustering import choose_k, evaluate_k, fit_kmeans, name_segments, prepare_features
from src.data import clean, make_demo_data
from src.rfm import add_rfm_scores, compute_rfm


def test_cleaning_removes_bad_rows():
    raw = make_demo_data(n_customers=200)
    df = clean(raw)
    assert df["CustomerID"].notna().all()
    assert not df["InvoiceNo"].str.startswith("C").any()
    assert (df["Quantity"] > 0).all() and (df["UnitPrice"] > 0).all()
    assert len(df) < len(raw)


def test_rfm_values_are_sane():
    df = clean(make_demo_data(n_customers=200))
    rfm = compute_rfm(df)
    assert rfm.index.is_unique
    assert (rfm["Recency"] >= 1).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()
    assert rfm["Monetary"].sum() == pytest.approx(df["TotalPrice"].sum(), rel=1e-6)


def test_rfm_known_example():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "InvoiceNo": ["A", "B", "C"],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-10", "2011-01-05"]),
            "TotalPrice": [10.0, 20.0, 5.0],
        }
    )
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2011-01-11"))
    assert rfm.loc[1].to_dict() == {"Recency": 1, "Frequency": 2, "Monetary": 30.0}
    assert rfm.loc[2].to_dict() == {"Recency": 6, "Frequency": 1, "Monetary": 5.0}


def test_scores_in_range():
    rfm = add_rfm_scores(compute_rfm(clean(make_demo_data(n_customers=300))))
    for col in ["R_score", "F_score", "M_score"]:
        assert rfm[col].between(1, 5).all()


def test_clustering_and_naming():
    rfm = compute_rfm(clean(make_demo_data(n_customers=400)))
    X, _ = prepare_features(rfm)
    metrics = evaluate_k(X, k_range=range(2, 7))
    k = choose_k(metrics)
    assert 3 <= k <= 6
    model = fit_kmeans(X, k)
    names = name_segments(model)
    assert len(set(names.values())) == k  # every segment gets a unique name
