"""
LightGBM base model with Optuna hyperparameter optimization.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from pylastmlextension import BaseModel


class LightGBMBaseModel(BaseModel):
    """
    Base class for LightGBM models with Optuna hyperparameter optimization.
    Subclasses should implement task-specific methods like _get_objective().
    """

    def __init__(
        self,
        name: str,
        feature_schema: Dict[str, Dict[str, str]],
        n_trials: int,
        metric: str,
        cv: int = 5,
    ) -> None:
        super().__init__(name, feature_schema)
        self.n_trials = n_trials
        self.metric = metric
        self.cv = cv
        self.best_params_: Dict[str, Any] = {}
        self.best_iteration_: Optional[int] = None

    @abstractmethod
    def _get_objective(self) -> str:
        """
        Return the LightGBM objective string.
        Examples: "regression", "binary", "multiclass"
        """

    @abstractmethod
    def _get_cv_stratified(self) -> bool:
        """
        Return whether to use stratified sampling in cross-validation.
        Should be True for classification tasks to maintain class distribution.
        """

    @abstractmethod
    def _is_minimize(self) -> bool:
        """
        Return whether the optimization direction is minimize.
        Most metrics like rmse, binary_logloss should return True.
        Metrics like auc should return False.
        """

    def _get_extra_params(self) -> Dict[str, Any]:
        """
        Hook method to provide additional LightGBM parameters.
        Subclasses can override this to add task-specific parameters.
        Example: ClassifierModel can add is_unbalance=True for imbalanced data.
        """
        return {}

    def _get_specific_meta(self) -> Dict[str, Any]:
        """
        Return LightGBM-specific metadata for model configuration.
        """
        return {
            "metric": self.metric,
            "best_params": self.best_params_,
            "best_iteration": self.best_iteration_,
        }

    def _optimize_params(
        self, x: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series | np.ndarray]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with LightGBM CV.
        """
        objective_str = self._get_objective()
        stratified = self._get_cv_stratified()
        direction = "minimize" if self._is_minimize() else "maximize"

        extra_params = self._get_extra_params()

        def objective(trial: optuna.Trial):
            params = {
                "objective": objective_str,
                "metric": self.metric,
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_jobs": -1,
                "num_leaves": trial.suggest_int("num_leaves", 32, 512),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": -1,
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 400),
                **extra_params,
            }

            train_dataset = lgb.Dataset(x, y, weight=weights)
            cv_results = lgb.cv(
                params,
                train_dataset,
                num_boost_round=2000,
                nfold=self.cv,
                stratified=stratified,
                shuffle=True,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    optuna.integration.LightGBMPruningCallback(trial, self.metric),
                ],
                return_cvbooster=False,
            )
            metric_key = f"valid {self.metric}-mean"
            if self._is_minimize():
                best_score = np.min(cv_results[metric_key])  # pyright: ignore
                best_iteration = int(np.argmin(cv_results[metric_key]) + 1)  # pyright: ignore
            else:
                best_score = np.max(cv_results[metric_key])  # pyright: ignore
                best_iteration = int(np.argmax(cv_results[metric_key]) + 1)  # pyright: ignore
            trial.set_user_attr("n_estimators", best_iteration)
            return best_score

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100)
        study = optuna.create_study(direction=direction, pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params_ = study.best_params
        self.best_iteration_ = study.best_trial.user_attrs["n_estimators"]
        print(f"Best parameters: {self.best_params_}")
        print(f"Best score: {study.best_value}")

        return self.best_params_

    def _train_impl(
        self, x: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series | np.ndarray]
    ) -> Any:
        """
        Train the final model with optimized parameters.
        """
        if not self.best_params_:
            self._optimize_params(x, y, weights)

        final_params = self.best_params_.copy()
        final_params.update(
            {
                "objective": self._get_objective(),
                "metric": self.metric,
                **self._get_extra_params(),
            }
        )

        train_dataset = lgb.Dataset(x, y, weight=weights)
        if self.best_iteration_ is None:
            raise ValueError("Best iteration is not set. Please optimize the parameters first.")
        return lgb.train(final_params, train_dataset, num_boost_round=self.best_iteration_)
