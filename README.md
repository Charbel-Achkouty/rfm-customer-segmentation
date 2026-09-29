# Customer Segmentation with RFM Analysis and K-means

Segmenting e-commerce customers by **Recency, Frequency and Monetary value (RFM)** with K-means clustering, then turning each segment into a concrete marketing action.

> Dataset: [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (UK online store, Dec 2010 - Dec 2011).

## Business question

Which customers generate most of the revenue, which are drifting away, and what should the marketing team do about each group?

## Approach

1. **Clean** the transactions: drop rows with no CustomerID, cancellations (invoice starts with `C`), non-positive quantity/price and duplicates.
2. **Engineer RFM features** per customer
   - Recency: days since last purchase (relative to one day after the last transaction)
   - Frequency: number of distinct invoices
   - Monetary: total spend
3. **Preprocess**: `log1p` transform (RFM is heavily right-skewed) followed by standardisation.
4. **Choose k** using the elbow method and silhouette score, restricted to k = 3-6 so the result stays actionable.
5. **Fit K-means**, then name each cluster with transparent rules based on its centre (e.g. recent + frequent + high spend = *Champions*).
6. **Report**: segment sizes, revenue share, and a recommended action per segment.

## Project structure

```
.
├── main.py                 # run the full pipeline
├── src/
│   ├── data.py             # loading, cleaning, synthetic demo data
│   ├── rfm.py              # RFM features and quintile scores
│   ├── clustering.py       # scaling, k selection, K-means, segment naming
│   └── plots.py            # figures
├── tests/test_pipeline.py  # unit tests
├── data/                   # put the dataset here (see data/README.md)
└── outputs/                # generated tables and figures
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --data "data/Online Retail.xlsx"     # real data
python main.py --demo                               # synthetic data, no download
python main.py --data data/data.csv --k 4           # force the number of clusters
python -m pytest                                    # run tests
```

Outputs are written to `outputs/`: `customer_segments.csv`, `segment_summary.csv`, `k_selection_metrics.csv` and five figures in `outputs/figures/`.

## Results

_Fill this in after running on the real dataset._

| Segment | Customers (%) | Revenue (%) | Avg recency (days) | Avg frequency | Avg spend |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

Key findings (examples of what to write):
- Champions are X% of customers but generate Y% of revenue.
- Z% of customers are At Risk and still hold W% of revenue, so they are the best win-back target.

Add 2-3 screenshots from `outputs/figures/` here, for example `04_segment_profile.png` and `05_size_vs_revenue.png`.

## Design decisions

- **Log transform before scaling.** Without it, a few very large customers dominate the distance calculation.
- **k restricted to 3-6.** k = 2 often has the best silhouette but only separates "good" from "bad" customers.
- **Rule-based segment names** keep the mapping from cluster to business label transparent and easy to audit.
- **Quintile RFM scores** (`R_score`, `F_score`, `M_score`) are also computed as an interpretable baseline to compare with the clusters.

## Limitations and next steps

- K-means assumes roughly spherical clusters; try Gaussian Mixture Models or hierarchical clustering to compare.
- Only about a year of data, so seasonality (a strong Q4 effect) may influence recency.
- Add a Streamlit app to explore segments interactively.
- Extend with a predictive layer: customer lifetime value or churn prediction per segment.
