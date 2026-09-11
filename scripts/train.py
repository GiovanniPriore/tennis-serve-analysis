"""Select a model on chronological validation data and refit on train+validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import pandas as pd

from tennis_analysis.features import prepare_modeling_frame
from tennis_analysis.loading import audit_as_dict, load_matches, missing_value_report
from tennis_analysis.models import compare_on_validation, refit_selected
from tennis_analysis.splitting import temporal_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing atp_matches_*.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--no-xgboost", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    raw, audit = load_matches(args.data_dir)
    modeling, preparation = prepare_modeling_frame(raw)
    split = temporal_split(modeling, args.validation_years, args.test_years)
    validation_results, _, feature_group, model_name = compare_on_validation(
        split.train, split.validation, include_xgboost=not args.no_xgboost
    )
    train_and_validation = pd.concat([split.train, split.validation], ignore_index=True)
    model = refit_selected(train_and_validation, feature_group, model_name, include_xgboost=not args.no_xgboost)
    joblib.dump(model, output / "selected_model.joblib")
    validation_results.to_csv(output / "validation_results.csv", index=False)
    missing_value_report(raw).to_csv(output / "missing_values.csv")
    selection = {
        "feature_group": feature_group,
        "model_name": model_name,
        "dataset_audit": audit_as_dict(audit),
        "preparation": asdict(preparation),
        "split": split.summary(),
        "validation_year_count": args.validation_years,
        "test_year_count": args.test_years,
        "note": "Selected exclusively by validation MAE; test data was not evaluated during selection.",
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
