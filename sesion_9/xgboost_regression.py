"""
xgboost_regression.py
=====================
Regresión con XGBoost. Opcionalmente busca hiperparámetros con
RandomizedSearchCV o GridSearchCV.

Uso mínimo
----------
    model = XGBoostRegression().fit(X, y)
    model.print_metrics()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Instala XGBoost: pip install xgboost")

from base_regressor import BaseRegressor


class XGBoostRegression(BaseRegressor):
    """
    Regresión con XGBoost usando Pipelines.

    Parámetros
    ----------
    n_estimators : int, default=300
    learning_rate : float, default=0.05
    max_depth : int, default=6
    subsample : float, default=0.8
    colsample_bytree : float, default=0.8
    reg_alpha : float, default=0.0   (L1)
    reg_lambda : float, default=1.0  (L2)
    tune : bool, default=False
    search : {'random', 'grid'}, default='random'
    param_grid : dict o None
    n_iter : int, default=20
    cv : int, default=5
    test_size : float, default=0.2
    random_state : int, default=42
    scale_features : bool, default=False

    Ejemplos
    --------
    # Uso mínimo
    model = XGBoostRegression().fit(X, y)

    # Con tuning
    model = XGBoostRegression(tune=True, n_iter=25).fit(X, y)

    # Encadenamiento
    fi = XGBoostRegression().fit(X, y).get_feature_importance()
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        tune: bool = False,
        search: str = "random",
        param_grid: dict | None = None,
        n_iter: int = 20,
        cv: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = False,
    ):
        super().__init__(test_size=test_size, random_state=random_state,
                         scale_features=scale_features)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tune = tune
        self.search = search
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_: dict | None = None

    def _default_grid(self) -> dict:
        return {
            "regressor__n_estimators":     [50, 100, 200, 300],
            "regressor__learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth":        [3, 5, 7, 9],
            "regressor__subsample":        [0.6, 0.8, 1.0],
            "regressor__colsample_bytree": [0.6, 0.8, 1.0],
            "regressor__reg_alpha":        [0.0, 0.1, 1.0],
            "regressor__reg_lambda":       [0.5, 1.0, 5.0],
        }

    def fit(self, X, y) -> "XGBoostRegression":
        """Ajusta el modelo y retorna self."""
        self.feature_names_ = self._names(X)
        X_arr, y_arr = self._arr(X), self._arr1d(y)
        self._split(X_arr, y_arr)

        xgb = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        pipeline = self._make_pipeline(xgb)

        if self.tune:
            grid = self.param_grid or self._default_grid()
            print(f"Buscando hiperparámetros ({self.search}) ...")
            searcher = (
                RandomizedSearchCV(pipeline, grid, n_iter=self.n_iter,
                                   cv=self.cv, scoring="neg_mean_squared_error",
                                   random_state=self.random_state, n_jobs=-1)
                if self.search == "random"
                else GridSearchCV(pipeline, grid, cv=self.cv,
                                  scoring="neg_mean_squared_error", n_jobs=-1)
            )
            searcher.fit(self.X_train_, self.y_train_)
            self.pipeline_    = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
            print(f"  Mejores parámetros: {self.best_params_}")
        else:
            print("Ajustando XGBoost ...")
            pipeline.fit(self.X_train_, self.y_train_)
            self.pipeline_ = pipeline

        self._store_metrics(self.y_test_, self.pipeline_.predict(self.X_test_))
        return self

    def print_metrics(self):
        self._check_fitted()
        title = "XGBoost Regression"
        if self.best_params_:
            title += f"\n  Params: {self.best_params_}"
        super().print_metrics(title)

    def get_feature_importance(self) -> pd.DataFrame:
        """Importancia de variables (gain), ordenada de mayor a menor."""
        self._check_fitted()
        xgb = self.pipeline_.named_steps["regressor"]
        return (
            pd.DataFrame({"Feature": self.feature_names_,
                          "Importancia": xgb.feature_importances_})
            .sort_values("Importancia", ascending=False)
            .reset_index(drop=True)
        )
