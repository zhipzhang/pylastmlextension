import os

import numpy as np
import pandas as pd
import pytest

from pylastmlextension.models.quantile_model import QuantileRegressionModel


@pytest.fixture
def quantile_data():
    np.random.seed(42)
    n_rows = 50
    df = pd.DataFrame(
        {
            "feature_A": np.random.rand(n_rows),
            "feature_B": np.random.randint(0, 100, n_rows),
            "feature_C": np.random.normal(0, 1, n_rows),
        }
    )
    # Target value y = 2*A + 0.5*B + noise
    y = 2 * df["feature_A"] + 0.5 * df["feature_B"] + np.random.normal(0, 0.1, n_rows)
    weights = np.random.rand(n_rows)
    return df, y, weights


@pytest.fixture
def schema():
    return {
        "feature_A": {"level": "L1", "description": "Float feat"},
        "feature_B": {"level": "L1", "description": "Int feat"},
        "feature_C": {"level": "L2", "description": "Noise feat"},
    }


def test_to_dict_quantile_model(quantile_data, schema, tmp_path):
    df, y, w = quantile_data
    model = QuantileRegressionModel(
        name="test_quantile_model",
        feature_schema=schema,
        n_trials=1,
        alpha=0.5,
        metric="quantile",
        cv=2,
    )
    model.train(df, y, weights=w)
    model.save_model(tmp_path / "test_quantile_model.txt")
    config = model.to_dict()
    assert config["meta"]["name"] == "test_quantile_model"
    assert config["meta"]["model_type"] == "quantile_regression"
    assert config["meta"]["alpha"] == 0.5
    assert config["meta"]["metric"] == "quantile"
