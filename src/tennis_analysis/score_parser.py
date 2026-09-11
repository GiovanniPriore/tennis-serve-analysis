"""Robust parsing for winner-perspective ATP score strings.

ATP match files record each set from the match winner's perspective. Therefore a
token such as ``3-6`` is a set lost by the eventual match winner; the first value
must not be replaced by the maximum of the two values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import Any


class ScoreStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    WALKOVER = "walkover"
    RETIREMENT = "retirement"
    DEFAULT = "default"
    SPECIAL_FORMAT = "special_format"
    INCOMPLETE = "incomplete"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class ParsedScore:
    winner_games: int | None
    loser_games: int | None
    winner_sets: int
    loser_sets: int
    status: ScoreStatus
    raw_score: str | None
    reason: str | None = None

    @property
    def game_margin(self) -> int | None:
        if self.winner_games is None or self.loser_games is None:
            return None
        return self.winner_games - self.loser_games

    @property
    def is_usable(self) -> bool:
        return self.status is ScoreStatus.OK and self.game_margin is not None


_SET_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\((?:\d+|RET)\))?$", re.IGNORECASE)
_RET_RE = re.compile(r"(?:^|\s)(?:RET|ABD)(?:\s|$)", re.IGNORECASE)
_WALKOVER_RE = re.compile(r"^(?:W/?O|WO)$", re.IGNORECASE)
_DEFAULT_RE = re.compile(r"^(?:DEF|DEFAULTED)$", re.IGNORECASE)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _valid_completed_set(winner_games: int, loser_games: int) -> bool:
    high, low = max(winner_games, loser_games), min(winner_games, loser_games)
    if high == 7 and low in {5, 6}:
        return True
    if high == 6 and low <= 4:
        return True
    return False


def parse_score(score: Any, best_of: int | None = None) -> ParsedScore:
    """Parse a completed score and return explicit exclusion status on failure.

    Tie-break points in parentheses are not counted as games. Retirements,
    walkovers, defaults, missing values, incomplete matches, and unknown tokens
    are never assigned an invented final margin.
    """
    if _missing(score):
        return ParsedScore(None, None, 0, 0, ScoreStatus.MISSING, None, "missing score")

    raw = str(score).strip()
    if not raw:
        return ParsedScore(None, None, 0, 0, ScoreStatus.MISSING, raw, "empty score")
    if _WALKOVER_RE.fullmatch(raw):
        return ParsedScore(None, None, 0, 0, ScoreStatus.WALKOVER, raw, "walkover has no played score")
    if _DEFAULT_RE.fullmatch(raw):
        return ParsedScore(None, None, 0, 0, ScoreStatus.DEFAULT, raw, "defaulted match")
    if _RET_RE.search(raw):
        return ParsedScore(None, None, 0, 0, ScoreStatus.RETIREMENT, raw, "retirement/abandonment")

    tokens = raw.split()
    sets: list[tuple[int, int]] = []
    for token in tokens:
        match = _SET_RE.fullmatch(token)
        if match is None:
            return ParsedScore(None, None, 0, 0, ScoreStatus.UNPARSEABLE, raw, f"unrecognized token: {token}")
        winner_games, loser_games = map(int, match.groups())
        if max(winner_games, loser_games) >= 10:
            return ParsedScore(
                None,
                None,
                0,
                0,
                ScoreStatus.SPECIAL_FORMAT,
                raw,
                "match tie-break is not comparable with a standard game count",
            )
        if not _valid_completed_set(winner_games, loser_games):
            return ParsedScore(None, None, 0, 0, ScoreStatus.INCOMPLETE, raw, f"incomplete or invalid set: {token}")
        sets.append((winner_games, loser_games))

    if not sets:
        return ParsedScore(None, None, 0, 0, ScoreStatus.UNPARSEABLE, raw, "no sets found")

    winner_sets = sum(w > l for w, l in sets)
    loser_sets = sum(l > w for w, l in sets)
    required_sets = (best_of // 2 + 1) if best_of in {3, 5} else None
    if required_sets is not None:
        if winner_sets != required_sets or loser_sets >= required_sets:
            return ParsedScore(None, None, winner_sets, loser_sets, ScoreStatus.INCOMPLETE, raw, "score does not complete the declared match format")
    elif winner_sets <= loser_sets:
        return ParsedScore(None, None, winner_sets, loser_sets, ScoreStatus.INCOMPLETE, raw, "eventual winner did not win more sets")

    return ParsedScore(
        winner_games=sum(w for w, _ in sets),
        loser_games=sum(l for _, l in sets),
        winner_sets=winner_sets,
        loser_sets=loser_sets,
        status=ScoreStatus.OK,
        raw_score=raw,
    )
