# Refactor plan

## Verified from the original notebook

- The source archive was later supplied and audited: 25 yearly CSV files, 74,906 matches, 2000–2024, with no exact or match-key duplicates.
- The saved Kaggle run loaded 74,906 rows and 49 columns, then retained 68,386 rows with complete serve statistics and 66,314 rows after score filtering.
- Available columns shown by the notebook include tournament metadata, match date, winner/loser identifiers and attributes, score, best-of format, match duration, serve statistics, rankings, and ranking points.
- The notebook concatenates files named `atp_matches_*.csv`. It later analyzes 2004–2013 and 2014–2023, but the exact minimum and maximum years cannot be re-audited without the source CSV files.
- Winner and loser are already encoded by the source columns (`winner_*`, `loser_*`). The task is therefore post-match explanatory regression, not pre-match forecasting.
- The target is the final game margin: `winner_games - loser_games`.
- The original score parser incorrectly assigns the larger game count in every set to the match winner, including sets lost by that player.
- The original serve features are winner/loser differences in aces, double faults, first-serve percentage, first-serve points won, second-serve points won, and break points saved.
- The original safe-division helper replaces undefined percentages with zero, conflating “no opportunity” with genuine 0% performance.
- The original primary split is random rather than temporal.
- The original notebook compares Linear Regression, Random Forest, and XGBoost, with later experiments for ranking, other match features, combined features, SHAP, and K-Means clustering.
- `minutes` is used in the later “other stats” model even though final duration is structurally close to the game-margin target.
- One-hot encoding for the later model is fitted before the split.
- Feature-group comparisons use different row populations after missing-value filtering and include manually copied historical metrics.
- The clustering silhouette score is about 0.15, indicating weak separation; cluster labels are manually assigned narrative names.
- The refactored pipeline now generates all reported metrics and figures from the supplied CSV files; the bundled historical `.pkl` was deliberately ignored.

## Planned changes

### Phase 1 — Audit

- Preserve the original notebook under `notebooks/original/`.
- Record verified facts and unavailable evidence in this file.
- Do not copy historical notebook metrics into the new results.

### Phase 2 — Core pipeline

- Add robust score parsing with explicit statuses and synthetic unit tests.
- Add deterministic CSV discovery/loading, schema validation, duplicate reporting, and audit summaries.
- Add NaN-preserving serve feature engineering and documented feature groups.
- Add chronological train/validation/test splitting and invariant checks.

### Phase 3 — Modeling

- Add mean and median DummyRegressor baselines, Linear Regression, Ridge, Random Forest, and optional XGBoost.
- Fit all imputers, encoders, scalers, and estimators on training data only through scikit-learn pipelines.
- Select the candidate model on validation MAE and evaluate the selected model once on the test period.
- Compare feature groups on one common eligible match population.

### Phase 4 — Analysis

- Add MAE, RMSE, R², bootstrap confidence intervals, residual diagnostics, and grouped error summaries.
- Add permutation importance and optional SHAP for the selected tree model.
- Keep clustering exploratory and explicitly report weak separation when applicable.

### Phase 5 — Portfolio repository

- Add packaging, CLI scripts, minimal CI, documentation, data instructions, and lightweight report notebooks.
- Keep generated datasets, models, caches, and large artifacts out of Git.
- Populate README results only after a successful run of the new pipeline on recovered ATP data.

## Required external data

Place Jeff Sackmann-style `atp_matches_*.csv` files under a gitignored local directory such as `data/raw`, then pass that path to the training script. Raw data are intentionally not committed.

## Open decisions

- Exact temporal boundaries will be chosen from the observed year range using complete calendar years. The default implementation reserves the two most recent years for test and the preceding two years for validation when enough years are available.
- XGBoost remains optional so the core repository can run without it.
- Match duration is excluded from all principal feature groups. Any duration experiment must be labeled as sensitivity analysis.
