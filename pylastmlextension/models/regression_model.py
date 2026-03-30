"""
Regression model using LightGBM with Optuna hyperparameter optimization.
"""

from typing import Dict

from .lightgbm_base import LightGBMBaseModel


class RegressionModel(LightGBMBaseModel):
    """
    Regression model using LightGBM.
    The hyperparameters are optimized by Optuna.
    """

    def __init__(
        self,
        name: str,
        feature_schema: Dict[str, Dict[str, str]],
        n_trials: int,
        metric: str = "rmse",
        cv: int = 5,
    ) -> None:
        super().__init__(name, feature_schema, n_trials, metric, cv)

    def _get_objective(self) -> str:
        """Return LightGBM objective for regression."""
        if self.metric == "rmse":
            return "regression_l2"
        elif self.metric == "mae" or self.metric == "l1":
            return "regression_l1"
        elif self.metric == "mse":
            return "regression_l2"
        elif self.metric == "r2":
            return "regression_l2"
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "regression"

    def _get_cv_stratified(self) -> bool:
        """Regression does not use stratified sampling."""
        return False

    def _is_minimize(self) -> bool:
        """Regression metrics like rmse, mae are minimized."""
        return True
