"""
Binary classification model using LightGBM with Optuna hyperparameter optimization.
"""

from typing import Any, Dict, Optional

from .lightgbm_base import LightGBMBaseModel


class ClassifierModel(LightGBMBaseModel):
    """
    Binary classification model using LightGBM.
    The hyperparameters are optimized by Optuna.

    Parameters
    ----------
    name : str
        Model name.
    feature_schema : Dict[str, Dict[str, str]]
        Schema of the features.
    n_trials : int
        Number of Optuna trials for hyperparameter optimization.
    metric : str, default="binary_logloss"
        Evaluation metric for optimization.
    cv : int, default=5
        Number of cross-validation folds.
    is_unbalance : bool, default=False
        Set to True if signal/background classes are unbalanced.
        This tells LightGBM to handle class imbalance automatically.
    """

    def __init__(
        self,
        name: str,
        feature_schema: Dict[str, Dict[str, str]],
        n_trials: int,
        metric: str = "binary_logloss",
        cv: int = 5,
        is_unbalance: bool = False,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__(name, feature_schema, n_trials, metric, cv)
        self.is_unbalance = is_unbalance
        self.scale_pos_weight = scale_pos_weight

    def _get_objective(self) -> str:
        """Return LightGBM objective for binary classification."""
        return "binary"

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "classification"

    def _get_cv_stratified(self) -> bool:
        """Classification uses stratified sampling to maintain class distribution."""
        return True

    def _is_minimize(self) -> bool:
        """
        Return optimization direction based on metric.
        binary_logloss should be minimized, auc should be maximized.
        """
        # AUC is maximized, most other metrics are minimized
        if self.metric in ("auc", "average_precision"):
            return False
        return True

    def _get_extra_params(self) -> Dict[str, Any]:
        """Return extra LightGBM parameters for classification."""
        params: Dict[str, Any] = {}
        if self.is_unbalance:
            params["is_unbalance"] = True
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        return params

    def _get_specific_meta(self) -> Dict[str, Any]:
        """Return classification-specific metadata."""
        meta = super()._get_specific_meta()
        meta["is_unbalance"] = self.is_unbalance
        return meta
