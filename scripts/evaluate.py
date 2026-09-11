"""Evaluate the previously selected model once on the latest test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from tennis_analysis.evaluation import bootstrap_intervals, grouped_error, prediction_frame, regression_metrics, save_diagnostic_plots
from tennis_analysis.explainability import NON_CAUSAL_NOTE, permutation_importance_table, save_optional_shap_summary, save_permutation_plot
from tennis_analysis.features import prepare_modeling_frame
from tennis_analysis.loading import load_matches
from tennis_analysis.models import FEATURE_GROUPS
from tennis_analysis.models import build_pipeline
from tennis_analysis.splitting import temporal_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); artifacts = Path(args.artifacts_dir); reports = Path(args.reports_dir)
    figures = reports / "figures"; figures.mkdir(parents=True, exist_ok=True)
    selection = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    model = joblib.load(artifacts / "selected_model.joblib")
    raw, _ = load_matches(args.data_dir); modeling, _ = prepare_modeling_frame(raw)
    split = temporal_split(modeling, selection["validation_year_count"], selection["test_year_count"])
    columns = FEATURE_GROUPS[selection["feature_group"]]
    predictions = model.predict(split.test[columns])
    metrics = regression_metrics(split.test["game_margin"], predictions)
    intervals = bootstrap_intervals(split.test["game_margin"], predictions)
    results = prediction_frame(split.test, predictions)
    results.to_csv(reports / "test_predictions.csv", index=False)
    baseline_rows = []
    train_and_validation = pd.concat([split.train, split.validation], ignore_index=True)
    refit_metrics = regression_metrics(
        train_and_validation["game_margin"], model.predict(train_and_validation[columns])
    )
    for baseline_name in ("dummy_mean", "dummy_median"):
        baseline = build_pipeline(baseline_name, columns)
        baseline.fit(train_and_validation[columns], train_and_validation["game_margin"])
        baseline_metrics = regression_metrics(
            split.test["game_margin"], baseline.predict(split.test[columns])
        )
        baseline_rows.append({"model": baseline_name, **baseline_metrics})
    pd.DataFrame(baseline_rows).to_csv(reports / "test_baselines.csv", index=False)
    grouped_error(results, "surface").to_csv(reports / "errors_by_surface.csv", index=False)
    grouped_error(results, "year").to_csv(reports / "errors_by_period.csv", index=False)
    grouped_error(results, "best_of").to_csv(reports / "errors_by_best_of.csv", index=False)
    save_diagnostic_plots(results, figures)

    fig, axis = plt.subplots(figsize=(7, 4)); axis.hist(modeling["game_margin"], bins=30, edgecolor="white")
    axis.set(xlabel="Winner game margin", ylabel="Matches", title="Target distribution")
    fig.tight_layout(); fig.savefig(figures / "target_distribution.png", dpi=160); plt.close(fig)

    importance = permutation_importance_table(model, split.test[columns], split.test["game_margin"])
    importance.to_csv(reports / "permutation_importance.csv", index=False)
    save_permutation_plot(importance, figures / "permutation_importance.png")
    shap_created = save_optional_shap_summary(model, split.test[columns], figures / "shap_summary.png")
    payload = {"model": selection["model_name"], "feature_group": selection["feature_group"], **metrics,
               "refit_train_validation_mae": refit_metrics["mae"],
               "mae_ci_95": intervals["mae"], "r2_ci_95": intervals["r2"],
               "test_rows": len(split.test), "shap_created": shap_created, "explainability_note": NON_CAUSAL_NOTE}
    (reports / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(reports / "results.csv").write_text(
        "model,feature_group,test_rows,mae,rmse,r2\n" +
        f"{payload['model']},{payload['feature_group']},{payload['test_rows']},{payload['mae']},{payload['rmse']},{payload['r2']}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
