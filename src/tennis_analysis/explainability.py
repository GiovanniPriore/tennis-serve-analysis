"""Non-causal model explanation utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

NON_CAUSAL_NOTE = "SHAP explains how the trained model uses each feature; it does not establish a causal relationship between serve statistics and match outcomes."


def permutation_importance_table(model, X, y, n_repeats: int = 20, random_seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=random_seed, n_jobs=-1)
    return pd.DataFrame({
        "feature": list(X.columns), "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def save_permutation_plot(table: pd.DataFrame, output_path: str | Path) -> None:
    ordered = table.sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(8, max(4, len(ordered) * 0.35)))
    axis.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    axis.set(xlabel="Increase in MAE after permutation", title="Permutation importance")
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)


def save_optional_shap_summary(model, X: pd.DataFrame, output_path: str | Path) -> bool:
    try:
        import shap
    except ImportError:
        return False
    estimator = model.named_steps["model"]
    if estimator.__class__.__name__ not in {"RandomForestRegressor", "XGBRegressor"}:
        return False
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    values = shap.TreeExplainer(estimator).shap_values(transformed)
    shap.summary_plot(values, transformed, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.tight_layout(); plt.savefig(output_path, dpi=160, bbox_inches="tight"); plt.close()
    return True
