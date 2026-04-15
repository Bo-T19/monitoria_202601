"""
base_regressor.py
=================
Clase base compartida por todos los modelos de regresión.
Centraliza: split, pipeline, métricas, predict, report y utilidades comunes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


class BaseRegressor:
    """
    Clase base para todos los modelos de regresión.

    Gestiona:
      - Conversión de inputs (DataFrame / array / lista)
      - Train/Test split
      - Construcción del Pipeline con scaler opcional
      - Cálculo y reporte de métricas
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features

        # Atributos llenados en fit()
        self.pipeline_: Pipeline | None = None
        self.feature_names_: list[str] = []
        self.metrics_: dict = {}
        self.X_train_: np.ndarray | None = None
        self.X_test_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.y_test_: np.ndarray | None = None

    # ── Conversión ────────────────────────────────────────────────────

    def _names(self, X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        n = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return [f"x{i}" for i in range(n)]

    def _arr(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)

    def _arr1d(self, y) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.to_numpy()
        return np.asarray(y).ravel()

    # ── Split ─────────────────────────────────────────────────────────

    def _split(self, X: np.ndarray, y: np.ndarray):
        (self.X_train_, self.X_test_,
         self.y_train_, self.y_test_) = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    # ── Pipeline ──────────────────────────────────────────────────────

    def _make_pipeline(self, estimator) -> Pipeline:
        """Construye Pipeline con scaler opcional."""
        steps = []
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("regressor", estimator))
        return Pipeline(steps)

    # ── Métricas ──────────────────────────────────────────────────────

    def _store_metrics(self, y_true, y_pred) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        self.metrics_ = {
            "R²":   round(float(r2_score(y_true, y_pred)), 6),
            "MSE":  round(float(mse), 6),
            "RMSE": round(float(np.sqrt(mse)), 6),
            "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 6),
        }
        return self.metrics_

    # ── API pública compartida ────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        """Genera predicciones para nuevos datos."""
        self._check_fitted()
        return self.pipeline_.predict(self._arr(X))

    def get_metrics(self) -> dict:
        """Retorna las métricas de Test como diccionario."""
        self._check_fitted()
        return self.metrics_.copy()

    def print_metrics(self, title: str = "Métricas en Test"):
        """Imprime un reporte de métricas en el conjunto de Test."""
        self._check_fitted()
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        for k, v in self.metrics_.items():
            print(f"  {k:<8}: {v}")
        print(sep)

    def summary(self) -> pd.DataFrame:
        """Retorna las métricas como DataFrame de una fila."""
        self._check_fitted()
        return pd.DataFrame([self.metrics_])

    def _check_fitted(self):
        if self.pipeline_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} no ha sido ajustado. "
                "Llama a .fit(X, y) primero."
            )
