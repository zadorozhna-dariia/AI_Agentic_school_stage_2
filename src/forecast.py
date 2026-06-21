"""
Phase 4 - Kyiv City forecasting (both grains), with an honest predictability
assessment.

CORE HONESTY POINT [F0]
Air-raid ONSETS are adversary-driven (strike decisions), not a natural
seasonal process. We therefore do NOT try to predict "an alert at 14:00 next
Tuesday." What is realistically modelable is the *rate / probability* and its
calendar seasonality. We quantify how far that gets us and where it plateaus.

DAILY grain  -> forecast expected alert COUNT per day.
  Series is low-count, overdispersed, ~stationary at city level. We use a
  Negative-Binomial GLM on calendar features (dow, month, recent activity)
  and benchmark it against two baselines. Gaussian ARIMA is inappropriate for
  mean-1.3 count data, so we avoid it [F1].

HOURLY grain -> forecast PROBABILITY of >=1 alert in an hour.
  95% of hours are empty; predicting a count is hopeless. We model P(alert)
  from hour-of-day + day-of-week via logistic regression and evaluate with
  ranking/probabilistic metrics (ROC-AUC, Brier), not point accuracy [F2].

Evaluation: strict TIME-BASED split (no shuffling) - train on the past,
test on the most recent held-out span. Metrics suited to each task.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

KYIV_TZ = "Europe/Kyiv"
VALID_END = "2025-10-31"
REGION = "Kyiv City"
TEST_FRACTION = 0.2          # most-recent 20% held out


# ----------------------- data builders -----------------------

def _kyiv_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["started_at"])
    df["local"] = df["started_at"].dt.tz_convert(KYIV_TZ)
    df["date"] = df["local"].dt.tz_localize(None).dt.normalize()
    df = df[(df["region"] == REGION) & (df["date"] <= VALID_END)]
    return df


def daily_series(path: str) -> pd.Series:
    df = _kyiv_events(path)
    s = df.groupby("date").size()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(grid, fill_value=0).rename("count")


def hourly_series(path: str) -> pd.Series:
    df = _kyiv_events(path)
    h = df["local"].dt.floor("h").dt.tz_localize(None)
    s = h.value_counts()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(grid, fill_value=0).sort_index().rename("count")


def time_split(idx_len: int, frac: float = TEST_FRACTION):
    cut = int(idx_len * (1 - frac))
    return cut


# ----------------------- DAILY: features + models -----------------------

def daily_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"count": s})
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    # lagged activity: yesterday's count and 7-day trailing mean (shifted)
    df["lag1"] = df["count"].shift(1)
    df["roll7"] = df["count"].shift(1).rolling(7).mean()
    return df.dropna()


def fit_daily_models(df: pd.DataFrame):
    """Return predictions dict on the held-out test span."""
    cut = time_split(len(df))
    train, test = df.iloc[:cut], df.iloc[cut:]

    # design matrix: dow & month as dummies + lag features
    def design(d):
        X = pd.get_dummies(d[["dow", "month"]].astype("category"), drop_first=True)
        X = X.astype(float)
        X["lag1"] = d["lag1"].values
        X["roll7"] = d["roll7"].values
        return sm.add_constant(X, has_constant="add")

    Xtr, Xte = design(train), design(test)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)  # align dummies
    ytr, yte = train["count"].values, test["count"].values

    preds = {}
    # Baseline 1: global mean of training
    preds["mean_baseline"] = np.full(len(yte), ytr.mean())
    # Baseline 2: seasonal-naive (same day-of-week mean from training)
    dow_mean = train.groupby("dow")["count"].mean()
    preds["seasonal_naive"] = test["dow"].map(dow_mean).values
    # Model: Negative Binomial GLM
    try:
        nb = NegativeBinomial(ytr, Xtr).fit(disp=0, maxiter=100)
        preds["neg_binomial"] = nb.predict(Xte)
    except Exception as e:                       # robust fallback to Poisson
        po = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit()
        preds["neg_binomial"] = po.predict(Xte)
    return test.index, yte, preds


def count_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mae = np.mean(np.abs(y - yhat))
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    # Poisson deviance (lower=better) - proper for count predictions
    eps = 1e-9
    dev = 2 * np.sum(np.where(y > 0, y * np.log((y + eps) / (yhat + eps)), 0)
                     - (y - yhat))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "PoissonDev": round(dev, 1)}


# ----------------------- HOURLY: P(alert) model -----------------------

def hourly_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": (s.values >= 1).astype(int)}, index=s.index)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    return df


def fit_hourly_model(df: pd.DataFrame):
    cut = time_split(len(df))
    train, test = df.iloc[:cut], df.iloc[cut:]
    Xtr = pd.get_dummies(train[["hour", "dow"]].astype("category"), drop_first=True).astype(float)
    Xte = pd.get_dummies(test[["hour", "dow"]].astype("category"), drop_first=True).astype(float)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, train["y"])
    p = clf.predict_proba(Xte)[:, 1]
    # baseline: constant base rate from train
    base = np.full(len(test), train["y"].mean())
    return test.index, test["y"].values, p, base


# ----------------------- plots -----------------------

def plot_daily(idx, y, preds, out):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(idx, y, color="0.6", lw=0.8, label="actual")
    ax.plot(idx, preds["neg_binomial"], color="tab:red", lw=1.4, label="NB model")
    ax.plot(idx, preds["seasonal_naive"], color="tab:green", lw=1.0,
            ls="--", label="seasonal-naive")
    ax.set_title("Kyiv City - daily alert count forecast (held-out test)")
    ax.set_ylabel("alerts / day"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_hourly_profile(df, clf_idx, y, p, out):
    """Show modeled P(alert) by hour vs empirical, the honest 'best we can do'."""
    test = pd.DataFrame({"hour": clf_idx.hour, "y": y, "p": p})
    emp = test.groupby("hour")["y"].mean()
    mod = test.groupby("hour")["p"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(emp.index, emp.values, color="0.8", label="empirical P(alert)")
    ax.plot(mod.index, mod.values, color="tab:red", marker="o", lw=1.6,
            label="model P(alert)")
    ax.set_title("Kyiv City - hourly P(>=1 alert): modeled vs empirical (test)")
    ax.set_xlabel("hour (Kyiv local)"); ax.set_ylabel("P(alert)")
    ax.set_xticks(range(0, 24, 2)); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ----------------------- driver -----------------------

def run_phase4(clean_path: str, fig_dir: str) -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)

    # DAILY
    ds = daily_series(clean_path)
    ddf = daily_features(ds)
    idx, yte, preds = fit_daily_models(ddf)
    daily_results = {name: count_metrics(yte, p) for name, p in preds.items()}
    plot_daily(idx, yte, preds, f"{fig_dir}/08_kyiv_daily_forecast.png")

    # HOURLY
    hs = hourly_series(clean_path)
    hdf = hourly_features(hs)
    hidx, hy, hp, hbase = fit_hourly_model(hdf)
    hourly_results = {
        "model_AUC": round(roc_auc_score(hy, hp), 3),
        "model_Brier": round(brier_score_loss(hy, hp), 4),
        "baseline_Brier": round(brier_score_loss(hy, hbase), 4),
        "test_base_rate": round(hy.mean(), 4),
    }
    plot_hourly_profile(hdf, hidx, hy, hp, f"{fig_dir}/09_kyiv_hourly_prob.png")

    return {"daily": daily_results, "hourly": hourly_results,
            "daily_test_n": len(yte), "hourly_test_n": len(hy)}


if __name__ == "__main__":
    r = run_phase4("../data/clean_oblast.csv", "../figures")
    print("=== DAILY count forecast (test n=%d) ===" % r["daily_test_n"])
    for m, met in r["daily"].items():
        print(f"  {m:16s} {met}")
    print("\n=== HOURLY P(alert) (test n=%d) ===" % r["hourly_test_n"])
    for k, v in r["hourly"].items():
        print(f"  {k:16s} {v}")
