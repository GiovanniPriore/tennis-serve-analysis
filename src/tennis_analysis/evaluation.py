"""Regression metrics, uncertainty estimates, and error diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_intervals(y_true, y_pred, n_bootstrap: int = 2_000, confidence: float = 0.95, random_seed: int = 42) -> dict[str, tuple[float, float]]:
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if len(truth) < 2:
        raise ValueError("At least two test observations are required")
    rng = np.random.default_rng(random_seed)
    maes, r2s = [], []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth, sampled_prediction = truth[indices], prediction[indices]
        maes.append(mean_absolute_error(sampled_truth, sampled_prediction))
        if np.unique(sampled_truth).size > 1:
            r2s.append(r2_score(sampled_truth, sampled_prediction))
    alpha = (1 - confidence) / 2
    quantiles = [alpha, 1 - alpha]
    return {
        "mae": tuple(float(value) for value in np.quantile(maes, quantiles)),
        "r2": tuple(float(value) for value in np.quantile(r2s, quantiles)),
    }


def prediction_frame(test: pd.DataFrame, predictions) -> pd.DataFrame:
    result = test.copy()
    result["prediction"] = np.asarray(predictions)
    result["residual"] = result["game_margin"] - result["prediction"]
    result["absolute_error"] = result["residual"].abs()
    result["year"] = result["match_date"].dt.year
    return result


def grouped_error(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return frame.groupby(group, dropna=False)["absolute_error"].agg(["count", "mean", "median"]).reset_index()


def save_diagnostic_plots(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(frame["game_margin"], frame["prediction"], alpha=0.25, s=12)
    bounds = [min(frame["game_margin"].min(), frame["prediction"].min()), max(frame["game_margin"].max(), frame["prediction"].max())]
    axis.plot(bounds, bounds, "--", color="black")
    axis.set(xlabel="Actual game margin", ylabel="Predicted game margin", title="Predicted vs actual")
    fig.tight_layout(); fig.savefig(output / "predicted_vs_actual.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(frame["residual"], bins=35, edgecolor="white"); axis.axvline(0, linestyle="--", color="black")
    axis.set(xlabel="Residual (actual - predicted)", ylabel="Matches", title="Residual distribution")
    fig.tight_layout(); fig.savefig(output / "residual_distribution.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(frame["prediction"], frame["residual"], alpha=0.25, s=12); axis.axhline(0, linestyle="--", color="black")
    axis.set(xlabel="Predicted game margin", ylabel="Residual", title="Residuals vs predictions")
    fig.tight_layout(); fig.savefig(output / "residual_plot.png", dpi=160); plt.close(fig)
