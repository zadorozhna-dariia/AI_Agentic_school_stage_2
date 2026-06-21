"""
Phase 2 - Exploratory data analysis & seasonality (all oblasts).

Consumes data/clean_oblast.csv from Phase 1 and produces:
  - daily & monthly alert-count trend (war-phase view)
  - hour-of-day seasonality (Kyiv local, hourly grain)
  - day-of-week seasonality
  - regional breakdown (counts + duration)
  - duration distribution (non-censored only)
  - STL decomposition of the daily count series

Metric choice [E1]: we analyze ALERT ONSET COUNTS as the primary signal.
Counts are robust to the right-censoring problem (a missed all-clear inflates
duration, not the fact that an alert started). Durations are shown separately
and ALWAYS exclude censored rows.

Time basis [E2]: hour-of-day and day-of-week use Kyiv-local time, because
operational/human rhythms follow local time, not UTC. The tz-aware column is
re-derived on load because CSV round-trips drop tz metadata.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

KYIV_TZ = "Europe/Kyiv"
sns.set_theme(style="whitegrid")


def load_clean(path: str) -> pd.DataFrame:
    """Load Phase 1 output and restore tz-aware / derived fields [E2]."""
    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    df["started_local"] = df["started_at"].dt.tz_convert(KYIV_TZ)
    df["hour_local"] = df["started_local"].dt.hour
    df["dow_local"] = df["started_local"].dt.dayofweek
    df["date_local"] = df["started_local"].dt.tz_localize(None).dt.normalize()
    return df


# ----------------------- aggregations -----------------------

def daily_counts(df: pd.DataFrame) -> pd.Series:
    """Alert onsets per local calendar day, reindexed to a gap-free grid."""
    s = df.groupby("date_local").size()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full, fill_value=0).rename("alerts")


def hour_of_day_share(df: pd.DataFrame) -> pd.Series:
    return (df["hour_local"].value_counts(normalize=True).sort_index() * 100)


def dow_share(df: pd.DataFrame) -> pd.Series:
    return (df["dow_local"].value_counts(normalize=True).sort_index() * 100)


def region_counts(df: pd.DataFrame) -> pd.Series:
    return df["region"].value_counts()


def region_duration_median(df: pd.DataFrame) -> pd.Series:
    nc = df[~df["censored"]]
    return nc.groupby("region")["duration_min"].median().sort_values(ascending=False)


# ----------------------- plots -----------------------

def plot_trend(daily: pd.Series, out: str) -> None:
    """Daily counts with a 7- and 30-day rolling mean to expose war phases."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(daily.index, daily.values, color="0.8", lw=0.6, label="daily")
    ax.plot(daily.index, daily.rolling(7).mean(), color="tab:blue", lw=1.3,
            label="7-day mean")
    ax.plot(daily.index, daily.rolling(30).mean(), color="tab:red", lw=1.8,
            label="30-day mean")
    ax.set_title("Air-raid alert onsets per day — all oblasts")
    ax.set_ylabel("alerts / day"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_hour_dow(hod: pd.Series, dow: pd.Series, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(hod.index, hod.values, color="tab:blue")
    axes[0].set_title("Hour-of-day seasonality (Kyiv local)")
    axes[0].set_xlabel("hour"); axes[0].set_ylabel("% of alerts")
    axes[0].set_xticks(range(0, 24, 2))

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(range(7), dow.values, color="tab:orange")
    axes[1].set_title("Day-of-week seasonality")
    axes[1].set_ylabel("% of alerts")
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(names)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_region(rc: pd.Series, out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    rc.sort_values().plot.barh(ax=ax, color="tab:green")
    ax.set_title("Total alert onsets by oblast (2022-03-15 - present)")
    ax.set_xlabel("alert count")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_duration(df: pd.DataFrame, out: str) -> None:
    """Duration distribution, non-censored, log-x to handle the heavy tail."""
    nc = df.loc[~df["censored"], "duration_min"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(np.log10(nc[nc > 0]), bins=60, color="tab:purple", alpha=.85)
    med = nc.median()
    ax.axvline(np.log10(med), color="k", ls="--",
               label=f"median {med:.0f} min")
    ax.set_title("Alert duration distribution (non-censored)")
    ax.set_xlabel("log10(duration minutes)"); ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_stl(daily: pd.Series, out: str, period: int = 7):
    """STL decomposition of the daily count series (weekly period) [E3].

    Period=7 isolates the weekly cycle; the remainder/trend show war phases.
    Annual seasonality is weak/irregular here (adversary-driven), so weekly is
    the meaningful periodic component to extract.
    """
    res = STL(daily.astype(float), period=period, robust=True).fit()
    fig = res.plot(); fig.set_size_inches(13, 8)
    fig.suptitle("STL decomposition of daily alert counts (weekly period)",
                 y=1.01)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return res


# ----------------------- driver -----------------------

def run_eda(clean_path: str, fig_dir: str) -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)
    df = load_clean(clean_path)
    daily = daily_counts(df)

    plot_trend(daily, f"{fig_dir}/01_trend.png")
    plot_hour_dow(hour_of_day_share(df), dow_share(df), f"{fig_dir}/02_hour_dow.png")
    plot_region(region_counts(df), f"{fig_dir}/03_region.png")
    plot_duration(df, f"{fig_dir}/04_duration.png")
    plot_stl(daily, f"{fig_dir}/05_stl.png")

    return {
        "n_alerts": len(df),
        "days": len(daily),
        "mean_per_day": round(daily.mean(), 1),
        "busiest_day": (daily.idxmax().date().isoformat(), int(daily.max())),
        "peak_hour": int(hour_of_day_share(df).idxmax()),
        "top_region": region_counts(df).index[0],
        "median_dur_min": round(df.loc[~df.censored, "duration_min"].median(), 1),
    }


if __name__ == "__main__":
    summary = run_eda("../data/clean_oblast.csv", "../figures")
    for k, v in summary.items():
        print(f"{k:16s}: {v}")
