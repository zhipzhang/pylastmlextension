"""
Write basic model class for training interface and method to write a
json configuration file for the model.
"""

from abc import ABC, abstractmethod
import os
from typing import Dict, List, Any
import pandas as pd


class BaseModel(ABC):
    """
    Base class for all models based on lightgbm.
    """

    def __init__(self, name: str, feature_schema: Dict[str, Dict[str, str]]) -> None:
        """
        params:
            name: model name
            feature_schema: schema of the features
                Example:
                {
                    "intensity": { "level": "dl2", "description": "intensity level" }
                }
        """
        self.name = name
        self.feature_schema = feature_schema

        self.features_: List[str] = []
        self.model_: Any = None
        self.model_path_: Any = None

    @abstractmethod
    def _train_impl(self, x: pd.DataFrame, y: pd.Series) -> Any:
        """
        Concrete implementation of the training logic.
        """

    @abstractmethod
    def _get_model_type(self) -> str:
        """
        Concrete implementation of the model type string.
        "regression", "quantile", "classification"
        """

    def _get_specific_meta(self) -> Dict[str, Any]:
        """
        Hook method to inject specific configuration information.
        Concrete implementation of the specific meta information.
        Example: Quantile: {'quantile_alpha': 0.8}
        """
        return {}

    def train(self, x: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        """
        self.features_ = x.columns.tolist()
        self.model_ = self._train_impl(x, y)
        print(f"Model {self.name} trained successfully with {len(self.features_)} features.")

    def save_model(self, output_dir: str) -> None:
        """
        Save the model to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model_path_ = os.path.join(output_dir, f"{self.name}.txt")
        self.model_.save_model(self.model_path_)
        print(f"Model {self.name} saved to {self.model_path_}.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        """

        if self.model_path_ is None:
            raise ValueError("Model path is not set. Please save the model first.")

        feature_info_list = []
        for f_name in self.features_:
            meta = self.feature_schema.get(f_name, {"level": "unknown", "description": "unknown"})
            feature_info_list.append({"name": f_name, "level": meta["level"], "description": meta["description"]})
        config = {
            "meta": {"name": self.name, "model_type": self._get_model_type()},
            "model_path": self.model_path_,
            "features": feature_info_list,
        }

        extra_meta = self._get_specific_meta()
        if extra_meta:
            config["meta"].update(extra_meta)
        return config
