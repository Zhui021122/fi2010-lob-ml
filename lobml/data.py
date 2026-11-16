"""Input contract, causal features, and temporal partitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = "LABEL_5TICK"
PURGE_ROWS = 10
LEVELS = range(10)
BOOK_COLUMNS = [
    f"{side}_{kind}_{level}"
    for side in ("PRICE", "VOLUME")
    for kind in ("ASK", "BID")
    for level in LEVELS
]
REQUIRED = ["STOCK", "DAY", *BOOK_COLUMNS]
FEATURES = [
    "spread_bps",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice_offset_bps",
    "ask_depth_5_log",
    "bid_depth_5_log",
    "ask_slope_bps",
    "bid_slope_bps",
    "imbalance_1_lag_1",
    "imbalance_1_lag_5",
    "mid_return_1_bps",
    "mid_return_5_bps",
    "mid_volatility_20_bps",
]


def load_data(path: str | Path, *, labeled: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    needed = REQUIRED + ([HORIZON] if labeled else [])
    missing = sorted(set(needed).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input CSV is empty")
    for col in needed:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame[needed].isna().any().any():
        raise ValueError("Required columns contain null values")
    if not np.isfinite(frame[BOOK_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Order-book columns contain non-finite values")
    if (frame["STOCK"] < 1).any() or (frame["DAY"] < 1).any():
        raise ValueError("STOCK and DAY must be positive")
    if (frame[[f"VOLUME_{kind}_{i}" for kind in ("ASK", "BID") for i in LEVELS]] < 0).any().any():
        raise ValueError("Volumes must be non-negative")
    if (frame["PRICE_ASK_0"] <= frame["PRICE_BID_0"]).any():
        raise ValueError("Best ask must exceed best bid")
    if labeled and not frame[HORIZON].isin((-1, 0, 1)).all():
        raise ValueError(f"{HORIZON} must contain only -1, 0, or 1")
    # The source CSV is ordered by stock and day. Preserve event order inside each group.
    groups = frame.groupby(["STOCK", "DAY"], sort=False).indices
    for positions in groups.values():
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError("Each STOCK/DAY group must occupy a contiguous block")
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute features using the current and earlier snapshots only."""
    ask = frame["PRICE_ASK_0"].astype(float)
    bid = frame["PRICE_BID_0"].astype(float)
    mid = (ask + bid) / 2
    safe_mid = mid.clip(lower=1e-12)
    result = pd.DataFrame(index=frame.index)
    result["spread_bps"] = 10_000 * (ask - bid) / safe_mid
    for k in (1, 5, 10):
        ask_depth = frame[[f"VOLUME_ASK_{i}" for i in range(k)]].sum(axis=1)
        bid_depth = frame[[f"VOLUME_BID_{i}" for i in range(k)]].sum(axis=1)
        result[f"imbalance_{k}"] = (bid_depth - ask_depth) / (bid_depth + ask_depth).clip(lower=1)
        if k == 5:
            result["ask_depth_5_log"] = np.log1p(ask_depth)
            result["bid_depth_5_log"] = np.log1p(bid_depth)
    ask_volume = frame["VOLUME_ASK_0"].astype(float)
    bid_volume = frame["VOLUME_BID_0"].astype(float)
    microprice = (ask * bid_volume + bid * ask_volume) / (bid_volume + ask_volume).clip(lower=1)
    result["microprice_offset_bps"] = 10_000 * (microprice - mid) / safe_mid
    result["ask_slope_bps"] = 10_000 * (frame["PRICE_ASK_9"] - ask) / safe_mid
    result["bid_slope_bps"] = 10_000 * (bid - frame["PRICE_BID_9"]) / safe_mid

    group_keys = [frame["STOCK"], frame["DAY"]]
    imbalance = result["imbalance_1"]
    result["imbalance_1_lag_1"] = imbalance.groupby(group_keys, sort=False).shift(1)
    result["imbalance_1_lag_5"] = imbalance.groupby(group_keys, sort=False).shift(5)
    previous_mid = mid.groupby(group_keys, sort=False).shift(1)
    old_mid = mid.groupby(group_keys, sort=False).shift(5)
    result["mid_return_1_bps"] = 10_000 * (mid / previous_mid - 1)
    result["mid_return_5_bps"] = 10_000 * (mid / old_mid - 1)
    result["mid_volatility_20_bps"] = (
        result["mid_return_1_bps"]
        .groupby(group_keys, sort=False)
        .transform(lambda s: s.rolling(20, min_periods=2).std())
    )
    result = result[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result.astype(np.float32)


def eligible_rows(frame: pd.DataFrame, purge_rows: int = PURGE_ROWS) -> pd.Series:
    """Discard tail rows whose future label window may cross a day boundary."""
    within_group = frame.groupby(["STOCK", "DAY"], sort=False).cumcount()
    group_size = frame.groupby(["STOCK", "DAY"], sort=False)["DAY"].transform("size")
    return within_group < group_size - purge_rows


def split_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    days = set(frame["DAY"].unique())
    if days != set(range(1, 11)):
        raise ValueError(f"Expected all ten FI-2010 days, got {sorted(days)}")
    eligible = eligible_rows(frame)
    return {
        "train": eligible & frame["DAY"].between(1, 6),
        "validation": eligible & frame["DAY"].eq(7),
        "test": eligible & frame["DAY"].between(8, 10),
    }
