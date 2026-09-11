import numpy as np
import pandas as pd

from tennis_analysis.features import add_features, add_target, safe_ratio


def test_zero_denominator_is_nan_not_zero():
    result = safe_ratio(pd.Series([0, 1]), pd.Series([0, 0]))
    assert result.isna().all()


def test_winner_loser_feature_direction():
    frame = pd.DataFrame({
        "w_ace": [10], "l_ace": [4], "w_df": [2], "l_df": [5],
        "w_svpt": [100], "l_svpt": [80], "w_1stIn": [60], "l_1stIn": [40],
        "w_1stWon": [45], "l_1stWon": [20], "w_2ndWon": [20], "l_2ndWon": [20],
        "w_bpSaved": [4], "w_bpFaced": [5], "l_bpSaved": [1], "l_bpFaced": [2],
        "winner_rank": [10], "loser_rank": [20], "winner_age": [25], "loser_age": [27],
        "winner_ht": [190], "loser_ht": [185],
    })
    features = add_features(frame)
    assert features.loc[0, "ace_diff"] == 6
    assert features.loc[0, "double_fault_diff"] == 3
    assert np.isclose(features.loc[0, "first_serve_pct_diff"], 0.1)


def test_margin_uses_set_orientation():
    frame = pd.DataFrame({"score": ["6-4 3-6 6-2"], "best_of": [3]})
    target = add_target(frame)
    assert target.loc[0, "winner_games"] == 15
    assert target.loc[0, "loser_games"] == 12
    assert target.loc[0, "game_margin"] == 3
