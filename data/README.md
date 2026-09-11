# Data

Recover the ATP match CSV files used by the original notebook and place them in
a local directory, for example `data/raw/`. Files must follow the naming pattern
`atp_matches_*.csv`.

The pipeline expects Jeff Sackmann-style columns including:

- tournament identifiers, date, surface, level, round, and best-of format;
- `winner_*` and `loser_*` identifiers and attributes;
- the final `score` from the match winner's perspective;
- winner and loser serve counts (`w_*`, `l_*`);
- rankings, ages, and heights for the optional non-serve feature groups.

Run:

```bash
python scripts/train.py --data-dir data/raw
python scripts/evaluate.py --data-dir data/raw
```

The training audit records file count, row count, year coverage, exact
duplicates, duplicate tournament/match keys, missing values, score exclusion
statuses, final population size, and chronological split boundaries.

Do not commit the raw CSV files unless their license explicitly permits it.
