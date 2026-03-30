"""
This model contains two quantile regression models to estimate the sigma
- quantile_model_16: 16th percentile
- quantile_model_84: 84th percentile
"""

from typing import Dict

from .models.quantile_model import QuantileRegressionModel
import pandas as pd
from typing import Optional, Any
import numpy as np
import os


class SigmaEstimator:
    def __init__(self, name: str, feature_schema: Dict[str, Dict[str, str]], n_trials: int, cv: int = 3):
        self.name = name
        self.quantile_model_16 = QuantileRegressionModel(f'{name}_quantile_16', feature_schema, n_trials, alpha=0.16, cv=cv)
        self.quantile_model_84 = QuantileRegressionModel(f'{name}_quantile_84', feature_schema, n_trials, alpha=0.84, cv=cv)
        self.metadata = {}

    def train(self, x: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series | np.ndarray] = None):
        self.quantile_model_16.train(x, y, weights)
        self.quantile_model_84.train(x, y, weights)
    
    def save_model(self, output_dir: str):
        self.quantile_model_16.save_model(output_dir)
        self.quantile_model_84.save_model(output_dir)
    
    def update_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    def save(self, output_dir: str):
        self.save_model(output_dir)
        import json
        with open(os.path.join(output_dir, f"{self.name}.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "name": self.name,
                "model_type": "sigma_estimator",
                **self.metadata
            },
            "quantile_model_16": self.quantile_model_16.to_dict(),
            "quantile_model_84": self.quantile_model_84.to_dict(),
        }
