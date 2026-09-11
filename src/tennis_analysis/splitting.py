"""Chronological split construction and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        return {
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
            "train_start": self.train["match_date"].min().isoformat(),
            "train_end": self.train["match_date"].max().isoformat(),
            "validation_start": self.validation["match_date"].min().isoformat(),
            "validation_end": self.validation["match_date"].max().isoformat(),
            "test_start": self.test["match_date"].min().isoformat(),
            "test_end": self.test["match_date"].max().isoformat(),
        }


def temporal_split(
    frame: pd.DataFrame,
    validation_years: int = 2,
    test_years: int = 2,
) -> TemporalSplit:
    if "match_date" not in frame:
        raise ValueError("match_date column is required")
    dated = frame.dropna(subset=["match_date"]).sort_values("match_date").copy()
    years = sorted(int(year) for year in dated["match_date"].dt.year.unique())
    required = validation_years + test_years + 1
    if len(years) < required:
        raise ValueError(f"At least {required} distinct years are required; found {len(years)}")

    test = tuple(years[-test_years:])
    validation = tuple(years[-(test_years + validation_years):-test_years])
    train_years = set(years).difference(validation).difference(test)
    split = TemporalSplit(
        train=dated.loc[dated["match_date"].dt.year.isin(train_years)].copy(),
        validation=dated.loc[dated["match_date"].dt.year.isin(validation)].copy(),
        test=dated.loc[dated["match_date"].dt.year.isin(test)].copy(),
        validation_years=validation,
        test_years=test,
    )
    validate_temporal_order(split)
    return split


def validate_temporal_order(split: TemporalSplit) -> None:
    if split.train.empty or split.validation.empty or split.test.empty:
        raise ValueError("All temporal partitions must be non-empty")
    if not split.train["match_date"].max() < split.validation["match_date"].min():
        raise ValueError("Training dates overlap validation dates")
    if not split.validation["match_date"].max() < split.test["match_date"].min():
        raise ValueError("Validation dates overlap test dates")
