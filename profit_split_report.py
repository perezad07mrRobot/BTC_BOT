"""Generate weekly profit split proposals from trade CSV reports."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
OUTPUT_TEMPLATE = "profit_split_{date}.csv"


def _find_time_column(columns: List[str]) -> str:
    candidates = [
        "timestamp",
        "time",
        "dt",
        "exit_time",
        "entry_time",
        "date",
    ]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError("No timestamp-like column found in report")


def _find_pnl_column(columns: List[str]) -> str:
    candidates = [
        "pnl",
        "profit",
        "net_pnl",
        "return",
    ]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError("No PnL column found in report")


def load_reports() -> pd.DataFrame:
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(f"Reports directory not found: {REPORTS_DIR}")

    frames = []
    for csv_file in sorted(REPORTS_DIR.glob("*.csv")):
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        ts_col = _find_time_column(df.columns.tolist())
        pnl_col = _find_pnl_column(df.columns.tolist())
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col, pnl_col])
        df = df[[ts_col, pnl_col]].rename(columns={ts_col: "timestamp", pnl_col: "pnl"})
        df["source_file"] = csv_file.name
        frames.append(df)

    if not frames:
        raise ValueError("No trade rows found across report CSVs")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    return combined


def prepare_weekly_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["timestamp"].dt.to_period("W-SUN")
    grouped = df.groupby("week", dropna=True)["pnl"].sum().reset_index()
    grouped["week_start"] = grouped["week"].dt.start_time.dt.date
    grouped["week_end"] = grouped["week"].dt.end_time.dt.date
    grouped["pnl"] = grouped["pnl"].round(2)
    grouped["withdraw_50pct"] = grouped["pnl"].apply(lambda v: round(max(v, 0) * 0.5, 2))
    grouped["reinvest_50pct"] = (grouped["pnl"] - grouped["withdraw_50pct"]).round(2)
    return grouped[["week_start", "week_end", "pnl", "withdraw_50pct", "reinvest_50pct"]]


def main() -> None:
    df = load_reports()
    weekly = prepare_weekly_split(df)
    today = dt.datetime.utcnow().strftime("%Y%m%d")
    output_path = REPORTS_DIR / OUTPUT_TEMPLATE.format(date=today)
    weekly.to_csv(output_path, index=False)

    print("Weekly profit split proposal")
    print("==============================")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(weekly.to_string(index=False))
    print("------------------------------")
    print(f"Saved to: {output_path}")
    print(
        "Totals -> PnL: {pnl:.2f} | Withdraw 50%: {wd:.2f} | Reinvest 50%: {ri:.2f}".format(
            pnl=weekly["pnl"].sum(),
            wd=weekly["withdraw_50pct"].sum(),
            ri=weekly["reinvest_50pct"].sum(),
        )
    )


if __name__ == "__main__":
    main()
