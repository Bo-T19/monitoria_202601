"""
regularized_regression.py
==========================
Regresión Lineal regularizada con Lasso (L1), Ridge (L2) y Elastic Net (L1+L2).

Penalización de cada método:
  Lasso      : alpha · |coef|
  Ridge      : alpha · coef²
  Elastic Net: alpha · [l1_ratio · |coef|  +  (1 - l1_ratio) · coef²]

Uso mínimo
----------
    model = RegularizedRegression().fit(X, y)   # ajusta los tres
    model.print_metrics()
    model.compare()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Lasso,      LassoCV,
    Ridge,      RidgeCV,
    ElasticNet, ElasticNetCV,
)
from base_regressor import BaseRegressor

# Métodos válidos
VALID_METHODS = {"lasso", "ridge", "elasticnet", "all"}


class RegularizedRegression(BaseRegressor):
    """
    Regresión Lineal con regularización Lasso, Ridge y/o Elastic Net.

    Alpha y l1_ratio se buscan automáticamente con validación cruzada
    a menos que se especifiquen valores fijos.

    Parámetros
    ----------
    method : {'all', 'lasso', 'ridge', 'elasticnet'}, default='all'
        Qué modelos ajustar. 'all' ajusta los tres.
    alpha : float o None, default=None
        Parámetro de regularización. None → búsqueda automática con CV.
    l1_ratio : float o None, default=None
        Solo para Elastic Net. Balance entre L1 y L2:
          0.0 → puro Ridge, 1.0 → puro Lasso, 0.5 → mitad y mitad.
        None → búsqueda automática con CV.
    cv : int, default=5
    test_size : float, default=0.2
    random_state : int, default=42
    scale_features : bool, default=True

    Ejemplos
    --------
    # Ajusta Lasso + Ridge + Elastic Net de una vez
    model = RegularizedRegression().fit(X, y)

    # Solo Elastic Net
    model = RegularizedRegression(method='elasticnet').fit(X, y)

    # Elastic Net con parámetros fijos
    model = RegularizedRegression(method='elasticnet',
                                   alpha=0.1, l1_ratio=0.5).fit(X, y)
    # Encadenamiento
    df = RegularizedRegression().fit(X, y).compare()
    """

    def __init__(
        self,
        method: str = "all",
        alpha: float | None = None,
        l1_ratio: float | None = None,
        cv: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        super().__init__(test_size=test_size, random_state=random_state,
                         scale_features=scale_features)
        if method not in VALID_METHODS:
            raise ValueError(f"method debe ser uno de {VALID_METHODS}")
        if l1_ratio is not None and not (0.0 <= l1_ratio <= 1.0):
            raise ValueError("l1_ratio debe estar entre 0.0 y 1.0")

        self.method   = method
        self.alpha    = alpha
        self.l1_ratio = l1_ratio
        self.cv       = cv

        self.pipelines_:   dict = {}
        self.best_params_: dict = {}   # {name: {"alpha": ..., "l1_ratio": ...}}
        self.metrics_all_: dict = {}

    # ── Ajuste individual de cada estimador ───────────────────────────

    def _fit_lasso(self, Xtr, ytr) -> tuple:
        """Retorna (estimador, params_elegidos)."""
        alphas = np.logspace(-4, 2, 100)
        if self.alpha is not None:
            chosen_alpha = self.alpha
        else:
            cv_m = LassoCV(alphas=alphas, cv=self.cv,
                           max_iter=10_000, random_state=self.random_state)
            cv_m.fit(Xtr, ytr)
            chosen_alpha = cv_m.alpha_
            print(f"  Mejor alpha (Lasso): {chosen_alpha:.6f}")

        reg = Lasso(alpha=chosen_alpha, max_iter=10_000)
        return reg, {"alpha": chosen_alpha}

    def _fit_ridge(self, Xtr, ytr) -> tuple:
        alphas = np.logspace(-4, 2, 100)
        if self.alpha is not None:
            chosen_alpha = self.alpha
        else:
            cv_m = RidgeCV(alphas=alphas, cv=self.cv)
            cv_m.fit(Xtr, ytr)
            chosen_alpha = cv_m.alpha_
            print(f"  Mejor alpha (Ridge): {chosen_alpha:.6f}")

        reg = Ridge(alpha=chosen_alpha)
        return reg, {"alpha": chosen_alpha}

    def _fit_elasticnet(self, Xtr, ytr) -> tuple:
        alphas    = np.logspace(-4, 2, 50)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

        if self.alpha is not None and self.l1_ratio is not None:
            # Ambos fijos
            chosen_alpha    = self.alpha
            chosen_l1_ratio = self.l1_ratio
        else:
            # Buscar uno o ambos con CV
            cv_m = ElasticNetCV(
                alphas=alphas,
                l1_ratio=l1_ratios if self.l1_ratio is None else [self.l1_ratio],
                cv=self.cv,
                max_iter=10_000,
                random_state=self.random_state,
            )
            cv_m.fit(Xtr, ytr)
            chosen_alpha    = self.alpha    if self.alpha    is not None else cv_m.alpha_
            chosen_l1_ratio = self.l1_ratio if self.l1_ratio is not None else cv_m.l1_ratio_
            print(f"  Mejor alpha (ElasticNet)   : {chosen_alpha:.6f}")
            print(f"  Mejor l1_ratio (ElasticNet): {chosen_l1_ratio:.4f}  "
                  f"({'→ más Lasso' if chosen_l1_ratio > 0.5 else '→ más Ridge'})")

        reg = ElasticNet(alpha=chosen_alpha, l1_ratio=chosen_l1_ratio, max_iter=10_000)
        return reg, {"alpha": chosen_alpha, "l1_ratio": chosen_l1_ratio}

    # ── Orquestador común ─────────────────────────────────────────────

    def _fit_one(self, name: str, Xtr, ytr, Xte, yte):
        """Ajusta un modelo por nombre, guarda pipeline y métricas."""
        builders = {
            "lasso":      self._fit_lasso,
            "ridge":      self._fit_ridge,
            "elasticnet": self._fit_elasticnet,
        }
        reg, params = builders[name](Xtr, ytr)

        self.best_params_[name] = params
        pipeline = self._make_pipeline(reg)
        pipeline.fit(Xtr, ytr)
        self.pipelines_[name] = pipeline

        y_pred = pipeline.predict(Xte)
        self._store_metrics(yte, y_pred)
        self.metrics_all_[name] = {**self.metrics_.copy(), **params}

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, X, y) -> "RegularizedRegression":
        """Ajusta el/los modelos y retorna self."""
        self.feature_names_ = self._names(X)
        X_arr, y_arr = self._arr(X), self._arr1d(y)
        self._split(X_arr, y_arr)

        names = (["lasso", "ridge", "elasticnet"]
                 if self.method == "all" else [self.method])

        for n in names:
            print(f"Ajustando {n.capitalize()} ...")
            self._fit_one(n, self.X_train_, self.y_train_,
                             self.X_test_,  self.y_test_)

        self.pipeline_ = self.pipelines_[names[-1]]
        return self

    # ── predict ───────────────────────────────────────────────────────

    def predict(self, X, method: str | None = None) -> np.ndarray:
        """
        Parámetros
        ----------
        method : 'lasso', 'ridge' o 'elasticnet'.
                 Si es None usa el último ajustado.
        """
        self._check_fitted()
        m = method or list(self.pipelines_)[-1]
        return self.pipelines_[m].predict(self._arr(X))

    # ── Reportes ──────────────────────────────────────────────────────

    def print_metrics(self):
        """Reporte de métricas para cada modelo ajustado."""
        self._check_fitted()
        for name, metrics in self.metrics_all_.items():
            sep = "=" * 58
            # Línea de parámetros elegidos
            params = self.best_params_[name]
            param_str = "  |  ".join(f"{k} = {v:.6f}" for k, v in params.items())
            print(f"\n{sep}")
            print(f"  {name.capitalize()}  |  {param_str}")
            print(sep)
            for k, v in metrics.items():
                if k not in params:          # no repetir alpha/l1_ratio
                    print(f"  {k:<8}: {v}")
            print(sep)

    def compare(self) -> pd.DataFrame:
        """DataFrame comparativo con métricas y parámetros de todos los modelos."""
        self._check_fitted()
        rows = [{"Modelo": n.capitalize(), **m}
                for n, m in self.metrics_all_.items()]
        return pd.DataFrame(rows).set_index("Modelo")

    def get_metrics(self, method: str | None = None) -> dict:
        self._check_fitted()
        if method:
            return self.metrics_all_.get(method, {})
        if len(self.metrics_all_) == 1:
            return next(iter(self.metrics_all_.values())).copy()
        return {n: m.copy() for n, m in self.metrics_all_.items()}

    def get_coefficients(self, method: str | None = None) -> pd.DataFrame:
        """
        Coeficientes de un modelo o de todos en un mismo DataFrame.
        """
        self._check_fitted()
        methods = [method] if method else list(self.pipelines_)
        frames = [
            pd.DataFrame({
                "Feature": self.feature_names_,
                f"Coef_{m.capitalize()}":
                    self.pipelines_[m].named_steps["regressor"].coef_,
            })
            for m in methods
        ]
        if len(frames) == 1:
            return frames[0]
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on="Feature")
        return df