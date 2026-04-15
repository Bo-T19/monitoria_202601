"""
stepwise_regression.py
======================
Regresión Lineal con selección Stepwise Forward-Backward por AIC.

    AIC = n * ln(RSS / n) + 2 * (k + 1)

Uso mínimo
----------
    model = StepwiseLinearRegression().fit(X, y)
    model.print_metrics()
    predictions = model.predict(X_new)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from base_regressor import BaseRegressor


class StepwiseLinearRegression(BaseRegressor):
    """
    Regresión Lineal con selección de variables Stepwise por AIC.

    Parámetros
    ----------
    direction : {'both', 'forward', 'backward'}, default='both'
    test_size : float, default=0.2
    random_state : int, default=42
    scale_features : bool, default=True
    verbose : bool, default=True

    Ejemplos
    --------
    # Uso mínimo — una línea
    model = StepwiseLinearRegression().fit(X, y)

    # Encadenamiento
    metrics = StepwiseLinearRegression(direction='forward').fit(X, y).get_metrics()

    # Con parámetros
    model = StepwiseLinearRegression(test_size=0.25, verbose=False).fit(X, y)
    model.print_metrics()
    model.print_aic_trace()
    """

    def __init__(
        self,
        direction: str = "both",
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        verbose: bool = True,
    ):
        super().__init__(test_size=test_size, random_state=random_state,
                         scale_features=scale_features)
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("direction debe ser 'forward', 'backward' o 'both'")
        self.direction = direction
        self.verbose = verbose
        self.selected_features_: list[str] = []
        self.aic_trace_: list[tuple] = []

    # ── AIC helpers ───────────────────────────────────────────────────

    def _aic(self, X_sub: np.ndarray, y: np.ndarray) -> float:
        n, k = len(y), X_sub.shape[1]
        Xa = np.column_stack([np.ones(n), X_sub])
        beta, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
        rss = max(float(np.sum((y - Xa @ beta) ** 2)), 1e-12)
        return n * np.log(rss / n) + 2 * (k + 1)

    def _aic_null(self, y: np.ndarray) -> float:
        rss = max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
        return len(y) * np.log(rss / len(y)) + 2

    # ── Stepwise ──────────────────────────────────────────────────────

    def _run_stepwise(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        n_feat = X.shape[1]
        included: list[int] = []
        step = 0
        current_aic = self._aic_null(y)
        self.aic_trace_ = [(step, "inicio", "—", round(current_aic, 4))]

        if self.verbose:
            print(f"  Paso {step:>2} | AIC nulo: {current_aic:.4f}")

        while True:
            step += 1
            improved = False

            # FORWARD — agregar la variable que más reduce AIC
            if self.direction in ("forward", "both"):
                excluded = [i for i in range(n_feat) if i not in included]
                best_aic, best_f = current_aic, None
                for f in excluded:
                    aic = self._aic(X[:, included + [f]], y)
                    if aic < best_aic:
                        best_aic, best_f = aic, f
                if best_f is not None:
                    included.append(best_f)
                    current_aic = best_aic
                    fname = self.feature_names_[best_f]
                    self.aic_trace_.append((step, "añadir", fname, round(current_aic, 4)))
                    improved = True
                    if self.verbose:
                        print(f"  Paso {step:>2} | + {fname:<22} | AIC = {current_aic:.4f}")

            # BACKWARD — eliminar la variable cuya ausencia reduce AIC
            if self.direction in ("backward", "both") and len(included) > 1:
                best_aic, worst_i = current_aic, None
                for i in range(len(included)):
                    cand = [included[j] for j in range(len(included)) if j != i]
                    aic = self._aic(X[:, cand], y)
                    if aic < best_aic:
                        best_aic, worst_i = aic, i
                if worst_i is not None:
                    fname = self.feature_names_[included.pop(worst_i)]
                    current_aic = best_aic
                    self.aic_trace_.append((step, "eliminar", fname, round(current_aic, 4)))
                    improved = True
                    if self.verbose:
                        print(f"  Paso {step:>2} | - {fname:<22} | AIC = {current_aic:.4f}")

            if not improved:
                if self.verbose:
                    print(f"  → Convergencia. AIC final = {current_aic:.4f}")
                break

        return included

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, X, y) -> "StepwiseLinearRegression":
        """
        Ajusta el modelo y retorna self (permite encadenamiento).

        Parámetros
        ----------
        X : DataFrame, ndarray o lista  — (n_samples, n_features)
        y : Series, ndarray o lista     — (n_samples,)
        """
        self.feature_names_ = self._names(X)
        X_arr, y_arr = self._arr(X), self._arr1d(y)
        self._split(X_arr, y_arr)

        # Escalar antes del stepwise (estabilidad numérica del OLS)
        sc = StandardScaler().fit(self.X_train_) if self.scale_features else None
        Xtr_sc = sc.transform(self.X_train_) if sc else self.X_train_

        print(f"Stepwise {self.direction.capitalize()} por AIC ...")
        sel_idx = self._run_stepwise(Xtr_sc, self.y_train_)
        self.selected_features_ = [self.feature_names_[i] for i in sel_idx]
        print(f"  Seleccionadas ({len(self.selected_features_)}): {self.selected_features_}")

        # Pipeline final solo con las variables seleccionadas
        Xtr = self.X_train_[:, sel_idx]
        Xte = self.X_test_[:, sel_idx]
        self.pipeline_ = self._make_pipeline(LinearRegression())
        self.pipeline_.fit(Xtr, self.y_train_)
        self._store_metrics(self.y_test_, self.pipeline_.predict(Xte))
        self.metrics_["AIC_final"] = round(self.aic_trace_[-1][3], 4)
        return self

    # ── predict ───────────────────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        sel_idx = [self.feature_names_.index(f) for f in self.selected_features_]
        return self.pipeline_.predict(self._arr(X)[:, sel_idx])

    # ── Reportes ──────────────────────────────────────────────────────

    def print_metrics(self):
        title = (f"Stepwise AIC ({self.direction})  |  "
                 f"{len(self.selected_features_)} var(s): {self.selected_features_}")
        super().print_metrics(title)

    def print_aic_trace(self):
        """Historial paso a paso del proceso de selección."""
        self._check_fitted()
        sep = "=" * 62
        print(f"\n{sep}\n  Historial AIC — Stepwise {self.direction.capitalize()}\n{sep}")
        print(f"  {'Paso':>4}  {'Acción':<10}  {'Variable':<22}  {'AIC':>10}")
        print("-" * 62)
        for paso, accion, feat, aic in self.aic_trace_:
            print(f"  {paso:>4}  {accion:<10}  {feat:<22}  {aic:>10.4f}")
        print(sep)

    def get_aic_trace(self) -> pd.DataFrame:
        """Historial de AIC como DataFrame."""
        self._check_fitted()
        return pd.DataFrame(self.aic_trace_,
                            columns=["Paso", "Acción", "Variable", "AIC"])

    def get_coefficients(self) -> pd.DataFrame:
        """Coeficientes del modelo final ordenados por magnitud."""
        self._check_fitted()
        reg = self.pipeline_.named_steps["regressor"]
        return (
            pd.DataFrame({"Feature": self.selected_features_,
                          "Coeficiente": reg.coef_})
            .sort_values("Coeficiente", key=abs, ascending=False)
            .reset_index(drop=True)
        )
