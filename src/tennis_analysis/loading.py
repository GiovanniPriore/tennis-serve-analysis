"""Dataset discovery, loading, validation, and audit utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "tourney_id", "tourney_date", "match_num", "score", "best_of",
    "winner_id", "loser_id", "surface", "tourney_level", "round",
    "winner_rank", "loser_rank", "winner_age", "loser_age", "winner_ht", "loser_ht",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn",
    "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
}


@dataclass(frozen=True)
class DatasetAudit:
    files: int
    rows: int
    columns: int
    min_year: int | None
    max_year: int | None
    exact_duplicate_rows: int
    duplicate_match_keys: int


def discover_match_files(data_dir: str | Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("atp_matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No atp_matches_*.csv files found in {Path(data_dir).resolve()}")
    return files


def load_matches(data_dir: str | Path) -> tuple[pd.DataFrame, DatasetAudit]:
    files = discover_match_files(data_dir)
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    matches = pd.concat(frames, ignore_index=True)
    missing = sorted(REQUIRED_COLUMNS.difference(matches.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    matches["match_date"] = pd.to_datetime(
        matches["tourney_date"].astype("Int64").astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )
    match_key = ["tourney_id", "match_num"]
    valid_dates = matches["match_date"].dropna()
    audit = DatasetAudit(
        files=len(files),
        rows=len(matches),
        columns=len(matches.columns),
        min_year=int(valid_dates.dt.year.min()) if not valid_dates.empty else None,
        max_year=int(valid_dates.dt.year.max()) if not valid_dates.empty else None,
        exact_duplicate_rows=int(matches.duplicated().sum()),
        duplicate_match_keys=int(matches.duplicated(match_key).sum()),
    )
    return matches, audit


def missing_value_report(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    selected = frame if columns is None else frame[list(columns)]
    return pd.DataFrame({
        "missing_count": selected.isna().sum(),
        "missing_pct": selected.isna().mean().mul(100),
    }).sort_values("missing_pct", ascending=False)


def audit_as_dict(audit: DatasetAudit) -> dict[str, int | None]:
    return asdict(audit)
