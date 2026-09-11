"""Optional exploratory clustering of match-level serve-difference profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analysis.features import SERVE_FEATURES, prepare_modeling_frame
from tennis_analysis.loading import load_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--sample-size", type=int, default=10_000)
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw, _ = load_matches(args.data_dir)
    frame, _ = prepare_modeling_frame(raw)
    transformed = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit_transform(frame[SERVE_FEATURES])

    rows = []
    labels_by_k = {}
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(transformed)
        score = silhouette_score(
            transformed,
            labels,
            sample_size=min(args.sample_size, len(frame)),
            random_state=42,
        )
        rows.append({"k": k, "silhouette": float(score)})
        labels_by_k[k] = labels

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    best_score = float(scores["silhouette"].max())
    scores.to_csv(reports / "clustering_silhouette.csv", index=False)
    frame.assign(cluster=labels_by_k[best_k]).groupby("cluster")[SERVE_FEATURES + ["game_margin"]].mean().to_csv(
        reports / "clustering_centroids.csv"
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(scores["k"], scores["silhouette"], marker="o")
    axis.set(xlabel="Number of clusters (k)", ylabel="Silhouette score", title="Exploratory K-Means separation")
    fig.tight_layout()
    fig.savefig(figures / "clustering_silhouette.png", dpi=160)
    plt.close(fig)
    payload = {
        "best_k": best_k,
        "best_silhouette": best_score,
        "rows": len(frame),
        "silhouette_sample_size": min(args.sample_size, len(frame)),
        "interpretation": "Exploratory match-level patterns only; clusters are not player types and are not causal.",
    }
    (reports / "clustering_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
