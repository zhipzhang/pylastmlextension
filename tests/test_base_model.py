import os

import pandas as pd
import pytest

from pylastmlextension.base import BaseModel


class MockBooster:
    def save_model(self, model_path: str) -> None:
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("mock model")


class MockTestModel(BaseModel):
    def _train_impl(self, x: pd.DataFrame, y: pd.Series) -> MockBooster:
        return MockBooster()

    def _get_model_type(self) -> str:
        return "test_type"

    def _get_specific_meta(self):
        return {"test_param": 999}


@pytest.fixture
def sample_data():
    df = pd.DataFrame({"featureA": [1, 2, 3], "featureB": [4, 5, 6], "unused_feature": [0, 0, 0]})
    y = pd.Series([1, 2, 3])
    return df, y


@pytest.fixture
def schema_config():
    return {
        "featureA": {"level": "dl2", "description": "feature A"},
        "featureB": {"level": "dl2", "description": "feature B"},
    }


@pytest.fixture
def model_instance(schema_config):
    return MockTestModel(name="test_model", feature_schema=schema_config)


def test_initialization(model_instance, schema_config):
    assert model_instance.name == "test_model"
    assert model_instance.feature_schema == schema_config
    assert len(model_instance.features_) == 0


def test_train_logic(model_instance, sample_data):
    df, y = sample_data

    train_x = df[["featureA", "featureB"]]
    model_instance.train(train_x, y)

    assert model_instance.features_ == ["featureA", "featureB"]
    assert model_instance.model_ is not None


def test_save_model(model_instance, sample_data, tmp_path):
    df, y = sample_data
    model_instance.train(df, y)

    output_dir = str(tmp_path / "output")
    model_instance.save_model(output_dir)
    expected_file_path = os.path.join(output_dir, "test_model.txt")
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, "r") as f:
        assert f.read() == "mock model"


def test_to_dict(model_instance, sample_data, tmp_path):
    df, y = sample_data
    model_instance.train(df, y)

    output_dir = str(tmp_path / "output")
    model_instance.save_model(output_dir)

    config = model_instance.to_dict()
    assert config["meta"]["name"] == "test_model"
    assert config["meta"]["model_type"] == "test_type"
    assert config["model_path"] == os.path.join(output_dir, "test_model.txt")
    assert len(config["features"]) == 3

    assert config["features"][0]["name"] == "featureA"
    assert config["features"][0]["level"] == "dl2"
    assert config["features"][0]["description"] == "feature A"
    assert config["features"][1]["name"] == "featureB"
    assert config["features"][1]["level"] == "dl2"
    assert config["features"][1]["description"] == "feature B"
    assert config["features"][2]["name"] == "unused_feature"
    assert config["features"][2]["level"] == "unknown"
    assert config["features"][2]["description"] == "unknown"
    assert config["meta"]["test_param"] == 999
