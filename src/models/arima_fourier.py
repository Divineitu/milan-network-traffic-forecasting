"""Model 1 (statistical): ARIMA with Fourier seasonal terms.

Why this rather than a classical SARIMA: the seasonal periods here are 144 (a day) and
1008 (a week). A seasonal ARIMA carries one state per seasonal lag, so s = 144 is already
expensive and s = 1008 is impractical, and a single SARIMA cannot represent two seasonal
periods at once. Dynamic harmonic regression instead supplies the seasonality as a few
sine/cosine pairs as external regressors, while the ARIMA part models the short-term
dependence left over (Hyndman & Athanasopoulos, 2021, ch. 12; cf. Shu et al., 2005, who
use seasonal ARIMA with two periodicities for wireless traffic).

Structure:
    x(t) = beta' f(t) + e(t),  where f(t) are the Fourier terms and e(t) ~ ARIMA(p, d, q)

Forecasting is one step ahead with the parameters fixed: the fitted model is extended
with the observed evaluation data (`append(..., refit=False)`), and its one-step-ahead
in-sample predictions over that stretch are read off. Each prediction therefore uses the
true history up to t and never the model's own earlier predictions.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.dataset import fourier_terms


class ArimaFourierForecaster:
    def __init__(self, order: tuple[int, int, int] = (2, 0, 0),
                 harmonics_day: int = 4, harmonics_week: int = 2) -> None:
        self.order = order
        self.harmonics_day = harmonics_day
        self.harmonics_week = harmonics_week
        self.name = f"ARIMA{order}+Fourier(d{harmonics_day},w{harmonics_week})"
        self._fitted = None
        self._train_index: pd.DatetimeIndex | None = None

    def _exog(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Fourier terms on a common time origin, so train and evaluation stay in phase."""
        if self._train_index is None:
            raise RuntimeError("fit() must be called before exogenous terms are built")
        offset = (index[0] - self._train_index[0]) // pd.Timedelta("10min")
        terms = fourier_terms(index, self.harmonics_day, self.harmonics_week)
        if offset:   # regenerate with the correct phase shift
            shifted = pd.date_range(self._train_index[0], periods=offset + len(index), freq="10min")
            terms = fourier_terms(shifted, self.harmonics_day, self.harmonics_week).iloc[offset:]
            terms.index = index
        return terms

    def fit(self, history: pd.Series) -> "ArimaFourierForecaster":
        self._train_index = history.index
        endog = history.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # convergence chatter, non-invertibility notes
            model = SARIMAX(endog, exog=self._exog(history.index), order=self.order,
                            trend="c", enforce_stationarity=False, enforce_invertibility=False)
            self._fitted = model.fit(disp=False)
        return self

    def predict(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        """One-step-ahead predictions over `target_index`, with parameters held fixed."""
        if self._fitted is None:
            raise RuntimeError("fit() must be called first")
        future = series.loc[target_index].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = self._fitted.append(future, exog=self._exog(target_index), refit=False)
        return np.asarray(extended.fittedvalues.loc[target_index])

    @property
    def summary(self) -> dict:
        """Parameter count and information criteria, for the comparison table."""
        return {"params": int(self._fitted.params.size),
                "aic": float(self._fitted.aic),
                "bic": float(self._fitted.bic)}
