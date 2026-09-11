"""Leakage-aware feature and target construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .score_parser import parse_score


SERVE_FEATURES = [
    "ace_diff",
    "double_fault_diff",
    "first_serve_pct_diff",
    "first_serve_points_won_pct_diff",
    "second_serve_points_won_pct_diff",
    "break_points_saved_pct_diff",
]

OTHER_NUMERIC_FEATURES = ["rank_diff", "age_diff", "height_diff", "best_of"]
OTHER_CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]
OTHER_MATCH_FEATURES = OTHER_NUMERIC_FEATURES + OTHER_CATEGORICAL_FEATURES
COMBINED_FEATURES = SERVE_FEATURES + OTHER_MATCH_FEATURES


@dataclass(frozen=True)
class PreparationSummary:
    initial_matches: int
    parsed_completed_scores: int
    negative_margin_scores: int
    usable_scores: int
    excluded_scores: int
    complete_common_population: int
    exclusion_by_status: dict[str, int]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    valid = denominator.notna() & numerator.notna() & denominator.gt(0)
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _difference(frame: pd.DataFrame, winner: str, loser: str) -> pd.Series:
    return pd.to_numeric(frame[winner], errors="coerce") - pd.to_numeric(frame[loser], errors="coerce")


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parsed = [parse_score(score, int(best_of) if pd.notna(best_of) else None) for score, best_of in zip(result["score"], result["best_of"])]
    result["score_status"] = [item.status.value for item in parsed]
    result["score_exclusion_reason"] = [item.reason for item in parsed]
    result["winner_games"] = [item.winner_games for item in parsed]
    result["loser_games"] = [item.loser_games for item in parsed]
    result["game_margin"] = [item.game_margin for item in parsed]
    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ace_diff"] = _difference(result, "w_ace", "l_ace")
    # Positive means the match winner committed fewer double faults.
    result["double_fault_diff"] = _difference(result, "l_df", "w_df")
    result["first_serve_pct_diff"] = safe_ratio(result["w_1stIn"], result["w_svpt"]) - safe_ratio(result["l_1stIn"], result["l_svpt"])
    result["first_serve_points_won_pct_diff"] = safe_ratio(result["w_1stWon"], result["w_1stIn"]) - safe_ratio(result["l_1stWon"], result["l_1stIn"])
    result["second_serve_points_won_pct_diff"] = safe_ratio(result["w_2ndWon"], result["w_svpt"] - result["w_1stIn"]) - safe_ratio(result["l_2ndWon"], result["l_svpt"] - result["l_1stIn"])
    result["break_points_saved_pct_diff"] = safe_ratio(result["w_bpSaved"], result["w_bpFaced"]) - safe_ratio(result["l_bpSaved"], result["l_bpFaced"])

    result["rank_diff"] = np.log(pd.to_numeric(result.get("loser_rank"), errors="coerce")) - np.log(pd.to_numeric(result.get("winner_rank"), errors="coerce"))
    result["age_diff"] = _difference(result, "winner_age", "loser_age")
    result["height_diff"] = _difference(result, "winner_ht", "loser_ht")
    return result


def prepare_modeling_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, PreparationSummary]:
    prepared = add_features(add_target(frame))
    parsed_ok = prepared["score_status"].eq("ok")
    negative_margin = parsed_ok & prepared["game_margin"].lt(0)
    usable = prepared.loc[prepared["score_status"].eq("ok") & prepared["game_margin"].ge(0)].copy()
    # Keep one common population for every principal feature-group comparison.
    # Feature missingness is handled inside train-fitted pipelines, not by using
    # a different dropna population for each experiment.
    common = usable.dropna(subset=["game_margin", "match_date"]).copy()
    counts = prepared["score_status"].value_counts().to_dict()
    summary = PreparationSummary(
        initial_matches=len(prepared),
        parsed_completed_scores=int(parsed_ok.sum()),
        negative_margin_scores=int(negative_margin.sum()),
        usable_scores=len(usable),
        excluded_scores=len(prepared) - len(usable),
        complete_common_population=len(common),
        exclusion_by_status={str(key): int(value) for key, value in counts.items() if key != "ok"},
    )
    return common, summary
