import numpy as np
import pandas as pd
import pytest

from lobml.data import BOOK_COLUMNS, build_features, eligible_rows, load_data, split_masks


def sample_frame() -> pd.DataFrame:
    rows = []
    for stock in (1, 2):
        for day in range(1, 11):
            for event in range(15):
                row = {"STOCK": stock, "DAY": day, "LABEL_5TICK": event % 3 - 1}
                for level in range(10):
                    row[f"PRICE_ASK_{level}"] = 101 + level + event * 0.01
                    row[f"PRICE_BID_{level}"] = 99 - level + event * 0.01
                    row[f"VOLUME_ASK_{level}"] = 100 + event + level
                    row[f"VOLUME_BID_{level}"] = 110 + event + level
                rows.append(row)
    return pd.DataFrame(rows)


def test_features_do_not_depend_on_future_rows():
    frame = sample_frame()
    original = build_features(frame)
    changed = frame.copy()
    changed.loc[20:, BOOK_COLUMNS] = changed.loc[20:, BOOK_COLUMNS] * 2
    updated = build_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:20], updated.iloc[:20])


def test_history_resets_at_stock_and_day_boundaries():
    frame = sample_frame()
    features = build_features(frame)
    for start in (0, 15, 150, 165):
        assert features.iloc[start]["mid_return_1_bps"] == 0
        assert features.iloc[start]["imbalance_1_lag_1"] == 0


def test_purge_and_temporal_partitions_are_disjoint():
    frame = sample_frame()
    masks = split_masks(frame)
    assert sum(mask.sum() for mask in masks.values()) == 100
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()
    for (_, _), group in frame.groupby(["STOCK", "DAY"], sort=False):
        assert eligible_rows(frame).loc[group.index].sum() == 5


def test_input_contract_rejects_crossed_book(tmp_path):
    frame = sample_frame()
    frame.loc[0, "PRICE_ASK_0"] = frame.loc[0, "PRICE_BID_0"]
    path = tmp_path / "book.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Best ask"):
        load_data(path)


def test_input_contract_rejects_missing_feature(tmp_path):
    frame = sample_frame().drop(columns=["VOLUME_BID_9"])
    path = tmp_path / "book.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_data(path)
