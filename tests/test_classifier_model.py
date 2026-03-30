import numpy as np
import pandas as pd
import pytest

from pylastmlextension.models.classifier_model import ClassifierModel


@pytest.fixture
def classification_data():
    """Generate binary classification data."""
    np.random.seed(42)
    n_rows = 100
    df = pd.DataFrame(
        {
            "feature_A": np.random.rand(n_rows),
            "feature_B": np.random.randint(0, 100, n_rows),
            "feature_C": np.random.normal(0, 1, n_rows),
        }
    )
    # Binary target: y = 1 if 2*A + 0.01*B > 1.5 else 0
    y = (2 * df["feature_A"] + 0.01 * df["feature_B"] > 1.5).astype(int)
    weights = np.random.rand(n_rows)
    return df, pd.Series(y), weights


@pytest.fixture
def schema():
    return {
        "feature_A": {"level": "L1", "description": "Float feat"},
        "feature_B": {"level": "L1", "description": "Int feat"},
        "feature_C": {"level": "L2", "description": "Noise feat"},
    }


def test_classifier_optuna_integration(classification_data, schema):
    """Test ClassifierModel with Optuna hyperparameter optimization."""
    x, y, w = classification_data

    model = ClassifierModel(
        name="test_classifier",
        feature_schema=schema,
        n_trials=1,
        metric="binary_logloss",
        cv=2,
    )
    model.train(x, y, weights=w)

    # Verify best_params_ is populated
    assert isinstance(model.best_params_, dict)
    assert "num_leaves" in model.best_params_
    assert "learning_rate" in model.best_params_

    # Verify best_iteration_ is set correctly
    assert model.best_iteration_ is not None
    assert model.best_iteration_ > 0

    # Verify model is trained
    assert model.model_ is not None

    # Verify predictions are probabilities between 0 and 1
    preds = model.model_.predict(x)
    assert len(preds) == len(x)
    assert all(0 <= p <= 1 for p in preds)


def test_classifier_to_dict_includes_optimization_meta(classification_data, schema, tmp_path):
    """Verify JSON output includes optimization metadata."""
    x, y, _ = classification_data
    model = ClassifierModel(
        name="test_classifier_meta",
        feature_schema=schema,
        n_trials=1,
        metric="binary_logloss",
        cv=2,
    )
    model.train(x, y)

    # Save model
    output_dir = str(tmp_path / "output")
    model.save_model(output_dir)

    # Generate config dict
    config = model.to_dict()

    # Verify meta information
    meta = config["meta"]

    assert meta["model_type"] == "classification"
    assert "best_params" in meta
    assert "best_iteration" in meta
    assert meta["best_iteration"] == model.best_iteration_
    assert meta["metric"] == "binary_logloss"

    # Verify best_params content
    assert "num_leaves" in meta["best_params"]
    assert "learning_rate" in meta["best_params"]


def test_classifier_model_type(schema):
    """Verify model type is correctly set to classification."""
    model = ClassifierModel(
        name="test_type",
        feature_schema=schema,
        n_trials=1,
        cv=2,
    )

    assert model._get_model_type() == "classification"
    assert model._get_objective() == "binary"
    assert model._get_cv_stratified() is True


def test_classifier_is_minimize_for_different_metrics(schema):
    """Test _is_minimize returns correct value for different metrics."""
    # binary_logloss should be minimized
    model_logloss = ClassifierModel(
        name="test_logloss",
        feature_schema=schema,
        n_trials=1,
        metric="binary_logloss",
    )
    assert model_logloss._is_minimize() is True

    # auc should be maximized
    model_auc = ClassifierModel(
        name="test_auc",
        feature_schema=schema,
        n_trials=1,
        metric="auc",
    )
    assert model_auc._is_minimize() is False


def test_classifier_is_unbalance_option(schema):
    """Test is_unbalance option for imbalanced datasets."""
    # Default: is_unbalance=False
    model_balanced = ClassifierModel(
        name="test_balanced",
        feature_schema=schema,
        n_trials=1,
    )
    assert model_balanced.is_unbalance is False
    assert model_balanced._get_extra_params() == {}

    # With is_unbalance=True
    model_unbalanced = ClassifierModel(
        name="test_unbalanced",
        feature_schema=schema,
        n_trials=1,
        is_unbalance=True,
    )
    assert model_unbalanced.is_unbalance is True
    assert model_unbalanced._get_extra_params() == {"is_unbalance": True}


def test_classifier_is_unbalance_in_meta(classification_data, schema, tmp_path):
    """Verify is_unbalance is included in model metadata."""
    x, y, _ = classification_data
    model = ClassifierModel(
        name="test_unbalance_meta",
        feature_schema=schema,
        n_trials=1,
        cv=2,
        is_unbalance=True,
    )
    model.train(x, y)

    # Save model
    output_dir = str(tmp_path / "output")
    model.save_model(output_dir)

    # Generate config dict
    config = model.to_dict()
    meta = config["meta"]

    assert meta["is_unbalance"] is True
