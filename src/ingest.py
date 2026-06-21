"""
Phase 1 — Ingest & clean the official Ukrainian air-raid siren dataset.

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
We use the OFFICIAL feed at OBLAST level as the analytical spine.

Design decisions (each flagged so a domain expert can verify):
  [A1] We keep only level == 'oblast'. Raion/hromada rows are a finer grain
       that double-counts the same physical alert and contains the extreme
       duration outliers; oblast level is the consistent unit for region-level
       seasonality and rate modeling.
  [A2] Exact duplicate (oblast, started_at, finished_at) rows are dropped.
       ~50% of raw oblast rows are exact dups (collection artifact).
  [A3] Timestamps are UTC in the raw file. We KEEP a UTC copy and ADD a
       Kyiv-local copy (Europe/Kyiv, DST-aware) for hour-of-day / day-of-week
       analysis, because human/operational rhythms follow local time.
  [A4] Durations are right-censored when an "all-clear" was missed. At oblast
       level only 13 rows exceed 24h, and all sit in frontline/border oblasts
       (Kharkiv, Donetsk, Chernihiv, Sumy, Luhansk) where a prolonged threat
       state can be genuine. We FLAG ONLY (`censored=True`, flag-only per user
       decision) so duration stats can exclude them while counts (alert onsets)
       remain intact. We do NOT store a capped value.
  [A6] Coverage starts 2022-03-15 (first official record). No volunteer
       backfill of the Feb 24 - Mar 15 2022 gap (user decision: Official-only).
  [A5] Overlapping alerts within one oblast: none exist after dedup in current
       data, but we provide a defensive merge so the pipeline stays correct if
       a future refresh introduces them.
"""
from __future__ import annotations
import pandas as pd

KYIV_TZ = "Europe/Kyiv"  # IANA modern spelling (DST-aware; alias of Europe/Kiev)
DURATION_CAP_MIN = 24 * 60  # [A4] 24h censoring threshold


def load_official(path: str) -> pd.DataFrame:
    """Load the official CSV with UTC-aware timestamps."""
    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    # Raw file is tz-aware UTC already; normalize defensively.
    for col in ("started_at", "finished_at"):
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")
        else:
            df[col] = df[col].dt.tz_convert("UTC")
    return df


def filter_oblast(df: pd.DataFrame) -> pd.DataFrame:
    """[A1] Keep oblast-level rows only; standardize the region column name."""
    out = df[df["level"] == "oblast"].copy()
    out = out.rename(columns={"oblast": "region"})
    return out[["region", "started_at", "finished_at"]]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """[A2] Drop exact duplicate alert intervals per region."""
    return df.drop_duplicates(subset=["region", "started_at", "finished_at"])


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with null timestamps or non-positive duration."""
    out = df.dropna(subset=["started_at", "finished_at"])
    out = out[out["finished_at"] > out["started_at"]]
    return out


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """[A5] Defensive: merge overlapping/adjacent intervals within a region.

    No-op on current data (zero overlaps after dedup) but keeps the interval
    set disjoint per region for any future refresh.
    """
    df = df.sort_values(["region", "started_at"])
    merged = []
    for region, g in df.groupby("region", sort=False):
        cur_start = cur_end = None
        for s, e in zip(g["started_at"], g["finished_at"]):
            if cur_start is None:
                cur_start, cur_end = s, e
            elif s <= cur_end:                 # overlap -> extend
                cur_end = max(cur_end, e)
            else:                              # gap -> flush
                merged.append((region, cur_start, cur_end))
                cur_start, cur_end = s, e
        if cur_start is not None:
            merged.append((region, cur_start, cur_end))
    return pd.DataFrame(merged, columns=["region", "started_at", "finished_at"])


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration, censoring flag [A4], and Kyiv-local time fields [A3]."""
    out = df.copy()
    out["duration_min"] = (
        (out["finished_at"] - out["started_at"]).dt.total_seconds() / 60
    )
    out["censored"] = out["duration_min"] > DURATION_CAP_MIN  # [A4]

    # [A3] Kyiv-local start time for diurnal / weekday analysis.
    local = out["started_at"].dt.tz_convert(KYIV_TZ)
    out["started_local"] = local
    out["hour_local"] = local.dt.hour
    out["dow_local"] = local.dt.dayofweek          # 0=Mon
    out["date_local"] = local.dt.date
    return out


def build_clean_dataset(path: str) -> pd.DataFrame:
    """End-to-end Phase 1 pipeline returning the analysis-ready frame."""
    df = load_official(path)
    df = filter_oblast(df)
    df = dedupe(df)
    df = drop_invalid(df)
    df = merge_overlaps(df)
    df = add_derived(df)
    return df.sort_values(["region", "started_at"]).reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: str) -> None:
    """Persist the cleaned frame (timestamps serialized as ISO UTC)."""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    RAW = "../data/official_data_en.csv"
    OUT = "../data/clean_oblast.csv"

    clean = build_clean_dataset(RAW)

    # --- Verification summary ---
    print("=== Phase 1 verification ===")
    print("Clean rows:        ", len(clean))
    print("Regions:           ", clean.region.nunique())
    print("Date range (UTC):  ", clean.started_at.min(), "->", clean.started_at.max())
    print("Start == 2022-03-15:", str(clean.started_at.min().date()) == "2022-03-15")
    print("Censored (>24h):   ", int(clean.censored.sum()), "(flag-only, kept in counts)")
    print("Duplicates remain: ",
          int(clean.duplicated(subset=["region", "started_at", "finished_at"]).sum()))
    print("Null timestamps:   ",
          int(clean[["started_at", "finished_at"]].isna().sum().sum()))
    print("Non-positive dur:  ", int((clean.duration_min <= 0).sum()))
    print("Median dur (real): ",
          round(clean.loc[~clean.censored, "duration_min"].median(), 1), "min")
    print("Columns:           ", list(clean.columns))

    save_clean(clean, OUT)
    print(f"\nSaved -> {OUT}")
