"""
random_forest_regression.py
============================
Regresión con Random Forest. Opcionalmente busca hiperparámetros
con RandomizedSearchCV o GridSearchCV.

Uso mínimo
----------
    model = RandomForestRegression().fit(X, y)
    model.print_metrics()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from base_regressor import BaseRegressor


class RandomForestRegression(BaseRegressor):
    """
    Regresión con Random Forest usando Pipelines.

    Parámetros
    ----------
    n_estimators : int, default=200
    max_depth : int o None, default=None
    tune : bool, default=False
        True → búsqueda de hiperparámetros con RandomizedSearchCV.
    search : {'random', 'grid'}, default='random'
    param_grid : dict o None
        Grilla personalizada. Si es None usa la predefinida.
    n_iter : int, default=20  (solo para search='random')
    cv : int, default=5
    test_size : float, default=0.2
    random_state : int, default=42
    scale_features : bool, default=False
        Random Forest no requiere escalar, pero la opción está disponible.

    Ejemplos
    --------
    # Uso mínimo
    model = RandomForestRegression().fit(X, y)

    # Con tuning automático
    model = RandomForestRegression(tune=True, n_iter=30).fit(X, y)

    # Encadenamiento
    fi = RandomForestRegression().fit(X, y).get_feature_importance()
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
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
        self.max_depth = max_depth
        self.tune = tune
        self.search = search
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_: dict | None = None

    def _default_grid(self) -> dict:
        return {
            "regressor__n_estimators":      [50, 100, 200, 300],
            "regressor__max_depth":         [None, 5, 10, 20],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf":  [1, 2, 4],
            "regressor__max_features":      ["sqrt", "log2", 0.5],
        }

    def fit(self, X, y) -> "RandomForestRegression":
        """Ajusta el modelo y retorna self."""
        self.feature_names_ = self._names(X)
        X_arr, y_arr = self._arr(X), self._arr1d(y)
        self._split(X_arr, y_arr)

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        pipeline = self._make_pipeline(rf)

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
            print("Ajustando Random Forest ...")
            pipeline.fit(self.X_train_, self.y_train_)
            self.pipeline_ = pipeline

        self._store_metrics(self.y_test_, self.pipeline_.predict(self.X_test_))
        return self

    def print_metrics(self):
        self._check_fitted()
        title = "Random Forest Regression"
        if self.best_params_:
            title += f"\n  Params: {self.best_params_}"
        super().print_metrics(title)

    def get_feature_importance(self) -> pd.DataFrame:
        """Importancia de variables ordenada de mayor a menor."""
        self._check_fitted()
        rf = self.pipeline_.named_steps["regressor"]
        return (
            pd.DataFrame({"Feature": self.feature_names_,
                          "Importancia": rf.feature_importances_})
            .sort_values("Importancia", ascending=False)
            .reset_index(drop=True)
        )
