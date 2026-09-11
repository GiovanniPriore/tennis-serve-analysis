import pandas as pd

from tennis_analysis.splitting import temporal_split


def test_temporal_order_is_strict():
    frame = pd.DataFrame({
        "match_date": pd.to_datetime([f"{year}-06-01" for year in range(2015, 2024)]),
        "game_margin": range(9),
    })
    split = temporal_split(frame, validation_years=2, test_years=2)
    assert split.train["match_date"].max() < split.validation["match_date"].min()
    assert split.validation["match_date"].max() < split.test["match_date"].min()
    assert split.validation_years == (2020, 2021)
    assert split.test_years == (2022, 2023)
