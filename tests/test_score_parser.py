import math

import pytest

from tennis_analysis.score_parser import ScoreStatus, parse_score


@pytest.mark.parametrize(
    ("score", "best_of", "winner_games", "loser_games", "margin"),
    [
        ("6-4 3-6 6-2", 3, 15, 12, 3),
        ("7-6(5) 6-4", 3, 13, 10, 3),
        ("4-6 6-3 7-5", 3, 17, 14, 3),
        ("6-2 6-2", 3, 12, 4, 8),
        ("6-3 3-6 6-4 4-6 6-2", 5, 25, 21, 4),
    ],
)
def test_completed_scores(score, best_of, winner_games, loser_games, margin):
    parsed = parse_score(score, best_of)
    assert parsed.status is ScoreStatus.OK
    assert (parsed.winner_games, parsed.loser_games, parsed.game_margin) == (winner_games, loser_games, margin)


@pytest.mark.parametrize(
    ("score", "status"),
    [("W/O", ScoreStatus.WALKOVER), ("RET", ScoreStatus.RETIREMENT), ("6-4 2-1 RET", ScoreStatus.RETIREMENT), (None, ScoreStatus.MISSING), (math.nan, ScoreStatus.MISSING)],
)
def test_non_completed_scores_are_not_invented(score, status):
    parsed = parse_score(score, 3)
    assert parsed.status is status
    assert parsed.game_margin is None


def test_incomplete_score_is_rejected():
    parsed = parse_score("6-4", best_of=3)
    assert parsed.status is ScoreStatus.INCOMPLETE
    assert not parsed.is_usable


def test_match_tiebreak_is_excluded_as_special_format():
    parsed = parse_score("6-4 3-6 10-8", best_of=3)
    assert parsed.status is ScoreStatus.SPECIAL_FORMAT
    assert parsed.game_margin is None
