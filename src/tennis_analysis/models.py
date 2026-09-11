"""Leakage-safe sklearn pipelines and validation-based model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import COMBINED_FEATURES, OTHER_CATEGORICAL_FEATURES, OTHER_MATCH_FEATURES, SERVE_FEATURES

RANDOM_SEED = 42
FEATURE_GROUPS = {"serve": SERVE_FEATURES, "other": OTHER_MATCH_FEATURES, "combined": COMBINED_FEATURES}


@dataclass(frozen=True)
class ValidationResult:
    feature_group: str
    model_name: str
    validation_mae: float


def _preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in OTHER_CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def available_estimators(include_xgboost: bool = True) -> dict[str, Callable[[], object]]:
    estimators: dict[str, Callable[[], object]] = {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "dummy_median": lambda: DummyRegressor(strategy="median"),
        "linear": LinearRegression,
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
            estimators["xgboost"] = lambda: XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, objective="reg:squarederror",
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        except ImportError:
            pass
    return estimators


def build_pipeline(model_name: str, feature_columns: list[str], include_xgboost: bool = True) -> Pipeline:
    estimators = available_estimators(include_xgboost)
    if model_name not in estimators:
        raise ValueError(f"Unknown or unavailable model: {model_name}. Available: {sorted(estimators)}")
    return Pipeline([("preprocessor", _preprocessor(feature_columns)), ("model", estimators[model_name]())])


def compare_on_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str = "game_margin",
    include_xgboost: bool = True,
) -> tuple[pd.DataFrame, Pipeline, str, str]:
    rows: list[ValidationResult] = []
    best: tuple[float, Pipeline, str, str] | None = None
    for group_name, columns in FEATURE_GROUPS.items():
        for model_name in available_estimators(include_xgboost):
            pipeline = build_pipeline(model_name, columns, include_xgboost)
            pipeline.fit(train[columns], train[target])
            mae = float(mean_absolute_error(validation[target], pipeline.predict(validation[columns])))
            rows.append(ValidationResult(group_name, model_name, mae))
            if best is None or mae < best[0]:
                best = (mae, pipeline, group_name, model_name)
    assert best is not None
    table = pd.DataFrame([item.__dict__ for item in rows]).sort_values("validation_mae")
    return table, best[1], best[2], best[3]


def refit_selected(
    frame: pd.DataFrame,
    feature_group: str,
    model_name: str,
    target: str = "game_margin",
    include_xgboost: bool = True,
) -> Pipeline:
    columns = FEATURE_GROUPS[feature_group]
    pipeline = build_pipeline(model_name, columns, include_xgboost)
    pipeline.fit(frame[columns], frame[target])
    return pipeline
