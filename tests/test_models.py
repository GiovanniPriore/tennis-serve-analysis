import pandas as pd

from tennis_analysis.features import SERVE_FEATURES
from tennis_analysis.models import build_pipeline


def _frame():
    return pd.DataFrame({column: [0.1, None, 0.3, 0.4] for column in SERVE_FEATURES})


def test_preprocessing_is_unfitted_before_pipeline_fit():
    pipeline = build_pipeline("ridge", SERVE_FEATURES, include_xgboost=False)
    assert not hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    frame = _frame()
    pipeline.fit(frame.iloc[:3], [1, 2, 3])
    assert hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    assert len(pipeline.predict(frame.iloc[3:])) == 1


def test_random_forest_reproducibility():
    frame = _frame().fillna(0); target = [1, 2, 3, 4]
    first = build_pipeline("random_forest", SERVE_FEATURES, include_xgboost=False).fit(frame, target)
    second = build_pipeline("random_forest", SERVE_FEATURES, include_xgboost=False).fit(frame, target)
    assert first.predict(frame).tolist() == second.predict(frame).tolist()
