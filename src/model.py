"""Explain and predict late departures with models a station manager can read.

The explanatory model is a logistic regression: each coefficient becomes an
odds ratio ("this factor multiplies the odds of a late departure by X"), with
a confidence interval. Stops of the same train are not independent (a train
late at one station is usually late at the next), so standard errors are
clustered by run; treating 99k stops as 99k independent draws would make every
interval far too narrow.

A gradient-boosted model is fitted on the same variables only as a yardstick:
if it predicts much better, the readable model is missing structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

# Reference levels are the "ordinary" case, so every odds ratio reads as
# "compared with a daytime IC stop outside Brussels".
PLANNING_FORMULA = (
    "late ~ C(hour_band, Treatment('daytime (9-16h)'))"
    " + weekend"
    " + C(category, Treatment('IC'))"
    " + brussels_junction"
    " + log_station_stops_per_day"
    " + trip_progress"
)
LIVE_FORMULA = PLANNING_FORMULA + " + prev_stop_delay_min"

PLANNING_FEATURES = ["hour_band", "weekend", "category", "brussels_junction",
                     "log_station_stops_per_day", "trip_progress"]


def fit_logit(df: pd.DataFrame, formula: str, cluster: str | None = "run_id"):
    """Fit a logistic regression, clustering standard errors by `cluster`."""
    model = smf.logit(formula, data=df)
    if cluster is None:
        return model.fit(disp=False)
    groups = pd.factorize(df[cluster])[0]
    return model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups})


def odds_ratio_table(result) -> pd.DataFrame:
    """Odds ratios with their 95% interval and p-value, intercept dropped."""
    ci = result.conf_int()
    table = pd.DataFrame({
        "odds_ratio": np.exp(result.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": result.pvalues,
    }).drop(index="Intercept")
    table.index = [_readable(name) for name in table.index]
    return table


def _readable(term: str) -> str:
    """Turn patsy's `C(x, Treatment('ref'))[T.level]` into `x = level`."""
    if term.startswith("C(") and "[T." in term:
        var = term[2:].split(",")[0].split(")")[0]
        level = term.split("[T.", 1)[1].rstrip("]")
        return f"{var} = {level}"
    return term


def evaluate(y_true: pd.Series, p: np.ndarray) -> dict:
    """AUC for ranking, Brier score for calibrated probabilities."""
    return {"auc": roc_auc_score(y_true, p), "brier": brier_score_loss(y_true, p)}


def calibration_table(y_true: pd.Series, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed late rate per decile of predicted risk."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": p})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False) + 1
    return frame.groupby("decile").agg(predicted=("p", "mean"), observed=("y", "mean"),
                                       stops=("y", "size"))


def fit_boosting(train: pd.DataFrame, features: list[str]):
    """Gradient boosting on the same variables, categoricals handled natively."""
    X = _encode(train, features)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                           max_iter=300, learning_rate=0.05,
                                           random_state=0)
    return model.fit(X, train["late"])


def predict_boosting(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(_encode(df, features))[:, 1]


def as_categories(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Cast text features to categoricals ONCE, on the full frame.

    Casting train and test separately would give each its own category codes,
    and the booster would silently read "IC" in one as "L" in the other.
    """
    df = df.copy()
    for col in features:
        if _is_text(df[col]):
            df[col] = df[col].astype("category")
    return df


def _is_text(col: pd.Series) -> bool:
    """True for object and pandas-3 string columns, False for categoricals."""
    return pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)         and not isinstance(col.dtype, pd.CategoricalDtype)


def _encode(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features]
    if any(_is_text(X[c]) for c in features):
        raise ValueError("cast text features with as_categories() before splitting")
    return X
