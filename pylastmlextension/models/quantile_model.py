"""
Quantile regression model using LightGBM with Optuna hyperparameter optimization.
"""

from typing import Any, Dict

from .lightgbm_base import LightGBMBaseModel


class QuantileRegressionModel(LightGBMBaseModel):
    """
    Quantile regression model using LightGBM.
    The hyperparameters are optimized by Optuna.

    This model predicts a specific quantile of the conditional distribution
    of the target variable, rather than the conditional mean.

    Parameters
    ----------
    name : str
        Model name.
    feature_schema : Dict[str, Dict[str, str]]
        Schema of the features.
    n_trials : int
        Number of Optuna trials for hyperparameter optimization.
    alpha : float, default=0.5
        The quantile to predict. Must be in (0, 1).
        For example, alpha=0.5 predicts the median,
        alpha=0.1 predicts the 10th percentile,
        alpha=0.9 predicts the 90th percentile.
    metric : str, default="quantile"
        Evaluation metric for optimization.
    cv : int, default=5
        Number of cross-validation folds.
    """

    def __init__(
        self,
        name: str,
        feature_schema: Dict[str, Dict[str, str]],
        n_trials: int,
        alpha: float = 0.5,
        metric: str = "quantile",
        cv: int = 5,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        super().__init__(name, feature_schema, n_trials, metric, cv)
        self.alpha = alpha

    def _get_objective(self) -> str:
        """Return LightGBM objective for quantile regression."""
        return "quantile"

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "quantile_regression"

    def _get_cv_stratified(self) -> bool:
        """Quantile regression does not use stratified sampling."""
        return False

    def _is_minimize(self) -> bool:
        """Quantile loss (pinball loss) is minimized."""
        return True

    def _get_extra_params(self) -> Dict[str, Any]:
        """Return extra LightGBM parameters for quantile regression."""
        return {"alpha": self.alpha}

    def _get_specific_meta(self) -> Dict[str, Any]:
        """Return quantile-regression-specific metadata."""
        meta = super()._get_specific_meta()
        meta["alpha"] = self.alpha
        return meta
