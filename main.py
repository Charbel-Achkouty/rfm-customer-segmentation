"""RFM customer segmentation with K-means.

Examples:
    python main.py --data "data/Online Retail.xlsx"
    python main.py --demo              # synthetic data, no download needed
    python main.py --data data.csv --k 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.clustering import (
    choose_k,
    evaluate_k,
    fit_kmeans,
    name_segments,
    prepare_features,
    summarise_segments,
)
from src.data import clean, load_raw, make_demo_data
from src.plots import (
    plot_clusters_pca,
    plot_k_selection,
    plot_rfm_distributions,
    plot_segment_profile,
    plot_size_vs_revenue,
)
from src.rfm import add_rfm_scores, compute_rfm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="path to the Online Retail .xlsx or .csv file")
    src.add_argument("--demo", action="store_true", help="use synthetic demo data")
    p.add_argument("--k", type=int, default=None, help="force the number of clusters (default: auto)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="output folder")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("main")

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    raw = make_demo_data(seed=args.seed) if args.demo else load_raw(args.data)
    df = clean(raw)
    log.info("Customers: %d | Invoices: %d | Period: %s to %s",
             df["CustomerID"].nunique(), df["InvoiceNo"].nunique(),
             df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date())

    # 2. features
    rfm = add_rfm_scores(compute_rfm(df))
    X, _ = prepare_features(rfm)

    # 3. model selection + fit
    metrics = evaluate_k(X, seed=args.seed)
    k = args.k or choose_k(metrics)
    log.info("Chosen k = %d (silhouette %.3f)", k, metrics.loc[metrics["k"] == k, "silhouette"].iloc[0])
    model = fit_kmeans(X, k, seed=args.seed)
    rfm["Cluster"] = model.labels_
    rfm["Segment"] = rfm["Cluster"].map(name_segments(model))

    # 4. business summary
    summary = summarise_segments(rfm)
    rfm.to_csv(args.out / "customer_segments.csv")
    summary.to_csv(args.out / "segment_summary.csv")
    metrics.to_csv(args.out / "k_selection_metrics.csv", index=False)

    # 5. figures
    plot_rfm_distributions(rfm, fig_dir)
    plot_k_selection(metrics, k, fig_dir)
    plot_clusters_pca(X, rfm["Segment"], fig_dir)
    plot_segment_profile(summary, fig_dir)
    plot_size_vs_revenue(summary, fig_dir)

    print("\n=== Segment summary ===")
    print(summary.drop(columns="Action").to_string())
    print(f"\nSaved tables and figures to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
