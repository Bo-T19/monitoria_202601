"""
catboost_regression.py
======================
Regresión con CatBoost. Opcionalmente busca hiperparámetros con
RandomizedSearchCV o GridSearchCV.

Uso mínimo
----------
    model = CatBoostRegression().fit(X, y)
    model.print_metrics()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    from catboost import CatBoostRegressor
except ImportError:
    raise ImportError("Instala CatBoost: pip install catboost")

from base_regressor import BaseRegressor


class CatBoostRegression(BaseRegressor):
    """
    Regresión con CatBoost usando Pipelines.

    Parámetros
    ----------
    iterations : int, default=300
        Número de árboles (equivalente a n_estimators).
    learning_rate : float, default=0.05
    depth : int, default=6
        Profundidad máxima del árbol (max 16 en CatBoost).
    l2_leaf_reg : float, default=3.0
        Regularización L2 sobre las hojas.
    subsample : float, default=0.8
        Fracción de muestras por árbol (requiere bootstrap_type='Bernoulli').
    colsample_bylevel : float, default=0.8
        Fracción de features por nivel del árbol.
    random_strength : float, default=1.0
        Fuerza del ruido para evitar sobreajuste al seleccionar splits.
    cat_features : list[int | str] | None, default=None
        Índices o nombres de columnas categóricas. CatBoost las maneja de forma
        nativa sin necesidad de codificación previa.
    tune : bool, default=False
        Si True, busca hiperparámetros con search CV.
    search : {'random', 'grid'}, default='random'
    param_grid : dict o None
        Grilla personalizada. Si None, se usa la grilla por defecto.
    n_iter : int, default=20
        Iteraciones para RandomizedSearchCV.
    cv : int, default=5
    test_size : float, default=0.2
    random_state : int, default=42
    scale_features : bool, default=False
        Los modelos basados en árboles no requieren escalado.

    Ejemplos
    --------
    # Uso mínimo
    model = CatBoostRegression().fit(X, y)

    # Con variables categóricas
    model = CatBoostRegression(cat_features=["ciudad", "tipo"]).fit(X, y)

    # Con tuning
    model = CatBoostRegression(tune=True, n_iter=25).fit(X, y)

    # Encadenamiento
    fi = CatBoostRegression().fit(X, y).get_feature_importance()
    """

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        colsample_bylevel: float = 0.8,
        random_strength: float = 1.0,
        cat_features: list | None = None,
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
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.random_strength = random_strength
        self.cat_features = cat_features
        self.tune = tune
        self.search = search
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_: dict | None = None

    def _default_grid(self) -> dict:
        return {
            "regressor__iterations":        [100, 200, 300, 500],
            "regressor__learning_rate":     [0.01, 0.05, 0.1, 0.2],
            "regressor__depth":             [4, 6, 8, 10],
            "regressor__l2_leaf_reg":       [1.0, 3.0, 5.0, 10.0],
            "regressor__subsample":         [0.6, 0.8, 1.0],
            "regressor__colsample_bylevel": [0.6, 0.8, 1.0],
            "regressor__random_strength":   [0.5, 1.0, 2.0],
        }

    def fit(self, X, y) -> "CatBoostRegression":
        """Ajusta el modelo y retorna self."""
        self.feature_names_ = self._names(X)
        X_arr, y_arr = self._arr(X), self._arr1d(y)
        self._split(X_arr, y_arr)

        cb = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            random_strength=self.random_strength,
            bootstrap_type="Bernoulli",   # necesario para subsample < 1.0
            cat_features=self.cat_features,
            random_seed=self.random_state,
            verbose=0,
        )
        pipeline = self._make_pipeline(cb)

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
            print("Ajustando CatBoost ...")
            pipeline.fit(self.X_train_, self.y_train_)
            self.pipeline_ = pipeline

        self._store_metrics(self.y_test_, self.pipeline_.predict(self.X_test_))
        return self

    def print_metrics(self):
        self._check_fitted()
        title = "CatBoost Regression"
        if self.best_params_:
            title += f"\n  Params: {self.best_params_}"
        super().print_metrics(title)

    def get_feature_importance(self) -> pd.DataFrame:
        """Importancia de variables, ordenada de mayor a menor."""
        self._check_fitted()
        cb = self.pipeline_.named_steps["regressor"]
        return (
            pd.DataFrame({"Feature": self.feature_names_,
                          "Importancia": cb.get_feature_importance()})
            .sort_values("Importancia", ascending=False)
            .reset_index(drop=True)
        )
