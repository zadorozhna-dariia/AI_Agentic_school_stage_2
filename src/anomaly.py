"""
Phase 3 - Anomaly detection & special-date analysis.

Two analyses, both on daily alert-onset COUNTS (Phase 2 metric choice):

  (A) Anomaly detection: flag days whose count is unusual RELATIVE TO THE
      LOCAL TREND + WEEKLY SEASONALITY, not the global mean. Because the
      series is strongly non-stationary (war phases), a global z-score would
      just label the busy 2024 period as one long anomaly. We instead use STL
      to remove trend+weekly season, then z-score the remainder on a rolling
      local scale. Runs on the all-oblast aggregate and on a single region.

  (B) Special-date windows: for each holiday/anniversary, compare a +/-5 day
      window against that SAME YEAR's local baseline (+/-21 days, window
      excluded). Pooling across years against a global mean is invalid here
      because trend confounds it [H1]. We report per-year ratios and the mean.
      Fixed-date holidays plus movable feasts (Western Easter, Orthodox Pascha)
      with explicit per-year dates; Christmas split into Western (25 Dec) and
      Orthodox (7 Jan).

Data window: truncated at VALID_END = 2025-10-31 (feed degrades after) [P3].
Luhansk excluded from per-region work (1 record total).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

KYIV_TZ = "Europe/Kyiv"
VALID_END = "2025-10-31"          # [P3] last trustworthy date


# ----------------------- data prep -----------------------

def load_daily(path: str, region: str | None = None) -> pd.Series:
    """Daily onset counts on a gap-free grid, truncated at VALID_END.

    region=None -> all-oblast aggregate; else that region only.
    """
    df = pd.read_csv(path, parse_dates=["started_at"])
    local = df["started_at"].dt.tz_convert(KYIV_TZ)
    df["date_local"] = local.dt.tz_localize(None).dt.normalize()
    df = df[df["date_local"] <= VALID_END]
    if region is not None:
        df = df[df["region"] == region]
    s = df.groupby("date_local").size()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(grid, fill_value=0).rename("alerts")


# ----------------------- (A) anomaly detection -----------------------

def detect_anomalies(daily: pd.Series, period: int = 7,
                     roll: int = 30, z_thresh: float = 3.0) -> pd.DataFrame:
    """STL-residual rolling z-score anomaly flags.

    1. STL(period=7) removes trend + weekly seasonality.
    2. Standardize the remainder by a rolling local std (handles changing
       volatility across war phases).
    3. Flag |z| >= z_thresh. Positive = unusually intense day.
    """
    stl = STL(daily.astype(float), period=period, robust=True).fit()
    resid = stl.resid
    local_sd = resid.rolling(roll, center=True, min_periods=roll // 2).std()
    z = resid / local_sd
    out = pd.DataFrame({
        "alerts": daily,
        "trend": stl.trend,
        "resid": resid,
        "z": z,
        "anomaly": z.abs() >= z_thresh,
    })
    out["direction"] = np.where(out["z"] >= 0, "high", "low")
    return out


def plot_anomalies(adf: pd.DataFrame, out: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(adf.index, adf["alerts"], color="0.75", lw=0.7, label="daily")
    ax.plot(adf.index, adf["trend"], color="tab:blue", lw=1.5, label="STL trend")
    hi = adf[adf["anomaly"] & (adf["direction"] == "high")]
    ax.scatter(hi.index, hi["alerts"], color="tab:red", s=22, zorder=5,
               label="high anomaly")
    ax.set_title(title); ax.set_ylabel("alerts / day"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ----------------------- (B) special-date windows -----------------------

HOLIDAYS = {
    "New Year (1 Jan)": (1, 1),
    "Orthodox Christmas (7 Jan)": (1, 7),
    "Western Christmas (25 Dec)": (12, 25),
    "Invasion Anniversary (24 Feb)": (2, 24),
    "Women's Day (8 Mar)": (3, 8),
    "Victory/Europe Day (9 May)": (5, 9),
    "Constitution Day (28 Jun)": (6, 28),
    "Independence Day (24 Aug)": (8, 24),
    "Defenders Day (1 Oct)": (10, 1),
}

# Movable feasts: explicit per-year dates (month, day). Western vs Orthodox
# Pascha diverge except in 2025 when both fall on 20 Apr.
MOVABLE_HOLIDAYS = {
    "Western Easter": {2022: (4, 17), 2023: (4, 9), 2024: (3, 31), 2025: (4, 20)},
    "Orthodox Pascha": {2022: (4, 24), 2023: (4, 16), 2024: (5, 5), 2025: (4, 20)},
}

WINDOW_HALF = 5   # +/-5 day window (was 3) per user decision
CTX_HALF = 21     # +/-21 day local baseline


def holiday_effect(daily: pd.Series, mm: int, dd: int,
                   half: int = WINDOW_HALF, ctx: int = CTX_HALF) -> pd.DataFrame:
    """Per-year window-vs-local-baseline ratio for one FIXED date [H1]."""
    per_year = {yr: (mm, dd) for yr in range(2022, 2026)}
    return _effect_from_dates(daily, per_year, half, ctx)


def _effect_from_dates(daily: pd.Series, per_year: dict,
                       half: int, ctx: int) -> pd.DataFrame:
    """Shared engine: per_year maps year -> (month, day)."""
    rows = []
    for yr, md in per_year.items():
        if md is None:
            continue
        try:
            c = pd.Timestamp(yr, md[0], md[1])
        except ValueError:
            continue
        if c < daily.index.min() + pd.Timedelta(days=ctx):
            continue
        if c > daily.index.max() - pd.Timedelta(days=ctx):
            continue
        win = daily.loc[c - pd.Timedelta(days=half):c + pd.Timedelta(days=half)]
        ctxw = daily.loc[c - pd.Timedelta(days=ctx):c + pd.Timedelta(days=ctx)]
        base = (ctxw.sum() - win.sum()) / (len(ctxw) - len(win))
        rows.append({
            "year": yr,
            "window_mean": round(win.mean(), 1),
            "local_base": round(base, 1),
            "ratio_pct": round((win.mean() / base - 1) * 100, 0) if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def holiday_summary(daily: pd.Series) -> pd.DataFrame:
    out = []
    # fixed-date holidays
    specs = [(name, {yr: (mm, dd) for yr in range(2022, 2026)})
             for name, (mm, dd) in HOLIDAYS.items()]
    # movable holidays
    specs += [(name, per_year) for name, per_year in MOVABLE_HOLIDAYS.items()]

    for name, per_year in specs:
        eff = _effect_from_dates(daily, per_year, WINDOW_HALF, CTX_HALF)
        if len(eff):
            out.append({
                "holiday": name,
                "years": len(eff),
                "mean_effect_pct": round(eff["ratio_pct"].mean(), 0),
                "min_pct": eff["ratio_pct"].min(),
                "max_pct": eff["ratio_pct"].max(),
            })
    return pd.DataFrame(out).sort_values("mean_effect_pct", ascending=False)


# ----------------------- driver -----------------------

def run_phase3(clean_path: str, fig_dir: str) -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)

    agg = load_daily(clean_path, region=None)
    kyiv = load_daily(clean_path, region="Kyiv City")

    a_agg = detect_anomalies(agg)
    a_kyiv = detect_anomalies(kyiv)
    plot_anomalies(a_agg, f"{fig_dir}/06_anomalies_all.png",
                   "Daily-count anomalies - all oblasts (STL-residual z>=3)")
    plot_anomalies(a_kyiv, f"{fig_dir}/07_anomalies_kyiv.png",
                   "Daily-count anomalies - Kyiv City")

    hol = holiday_summary(agg)
    hol.to_csv(f"{fig_dir}/holiday_effects.csv", index=False)

    top = a_agg[a_agg["anomaly"] & (a_agg["direction"] == "high")] \
        .sort_values("z", ascending=False).head(8)
    return {
        "agg_anomaly_days": int(a_agg["anomaly"].sum()),
        "kyiv_anomaly_days": int(a_kyiv["anomaly"].sum()),
        "top_high_days": [(d.date().isoformat(), int(r.alerts))
                          for d, r in top.iterrows()],
        "holiday_table": hol,
    }


if __name__ == "__main__":
    res = run_phase3("../data/clean_oblast.csv", "../figures")
    print("All-oblast anomaly days:", res["agg_anomaly_days"])
    print("Kyiv anomaly days:      ", res["kyiv_anomaly_days"])
    print("\nTop high-intensity anomaly days (all oblasts):")
    for d, n in res["top_high_days"]:
        print(f"  {d}: {n} alerts")
    print("\nHoliday / special-date effect (window vs same-year local baseline):")
    print(res["holiday_table"].to_string(index=False))
