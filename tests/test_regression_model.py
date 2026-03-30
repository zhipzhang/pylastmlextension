import os

import numpy as np
import pandas as pd
import pytest

from pylastmlextension.models.regression_model import RegressionModel


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n_rows = 50
    df = pd.DataFrame(
        {
            "feature_A": np.random.rand(n_rows),
            "feature_B": np.random.randint(0, 100, n_rows),
            "feature_C": np.random.normal(0, 1, n_rows),
        }
    )
    # 目标值 y = 2*A + 0.5*B + noise
    y = 2 * df["feature_A"] + 0.5 * df["feature_B"] + np.random.normal(0, 0.1, n_rows)
    weights = np.random.rand(n_rows)  # 随机权重
    return df, y, weights


@pytest.fixture
def schema():
    return {
        "feature_A": {"level": "L1", "description": "Float feat"},
        "feature_B": {"level": "L1", "description": "Int feat"},
        "feature_C": {"level": "L2", "description": "Noise feat"},
    }


def test_optuna_integration(regression_data, schema):
    x, y, w = regression_data

    model = RegressionModel(
        name="test_model", feature_schema=schema, n_trials=1, metric="rmse", cv=2
    )
    model.train(x, y, weights=w)
    assert isinstance(model.best_params_, dict)
    assert "num_leaves" in model.best_params_
    assert "learning_rate" in model.best_params_

    # --- 验证 2: 那个复杂的 Bug (Best Iteration) 是否修复 ---
    # 检查 best_iteration_ 是否被正确从 trial user_attrs 中取出
    assert model.best_iteration_ is not None
    assert model.best_iteration_ > 0

    # --- 验证 3: 最终模型是否生成 ---
    assert model.model_ is not None
    # 简单验证一下模型能预测 (LightGBM Booster 对象)
    preds = model.model_.predict(x)
    assert len(preds) == len(x)


def test_to_dict_includes_optimization_meta(regression_data, schema, tmp_path):
    """
    验证：JSON 输出中是否包含了我们辛辛苦苦搜出来的参数，并打印 config
    """
    x, y, _ = regression_data
    model = RegressionModel(name="test_meta_output", feature_schema=schema, n_trials=1, cv=2)
    model.train(x, y)

    # 保存模型
    output_dir = str(tmp_path / "output")
    model.save_model(output_dir)

    # 生成配置字典
    config = model.to_dict()

    # --- 验证 Meta 信息 ---
    # 我们在 _get_specific_meta 里写的逻辑应该在这里体现
    meta = config["meta"]

    assert "best_params" in meta
    assert "best_iteration" in meta
    assert meta["best_iteration"] == model.best_iteration_
    assert meta["metric"] == "rmse"

    # 验证 best_params 里的内容
    assert "num_leaves" in meta["best_params"]
    assert "learning_rate" in meta["best_params"]
