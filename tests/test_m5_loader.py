import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.m5_loader import load_m5_weekly


def write_m5(d, n_weeks=12):
    n_days = n_weeks * 7 + 3
    dates = pd.date_range("2015-01-03", periods=n_days)
    wk = 11501 + np.arange(n_days) // 7
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": wk, "month": dates.month, "year": dates.year,
                  "d": [f"d_{i + 1}" for i in range(n_days)],
                  "event_name_1": ["Ev" if i == 8 else None for i in range(n_days)]}).to_csv(d / "calendar.csv", index=False)
    rows, prices = [], []
    for store in ["CA_1", "TX_1"]:
        for i, (cat, units, swing) in enumerate([("FOODS", 5, 0.3), ("HOBBIES", 5, 0.1), ("HOUSEHOLD", 5, 0.0), ("FOODS", 1, 0.5)]):
            item = f"{cat}_1_{i:03d}"
            rows.append([item, f"{cat}_1", cat, store] + [units] * n_days)
            for w in np.unique(wk):
                prices.append((store, item, w, round(10 * (1 + swing * ((w % 2) - 0.5)), 2)))
    cols = ["item_id", "dept_id", "cat_id", "store_id"] + [f"d_{i + 1}" for i in range(n_days)]
    pd.DataFrame(rows, columns=cols).to_csv(d / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(d / "sell_prices.csv", index=False)


def test_weekly_panel_shape_and_selection(tmp_path):
    write_m5(tmp_path)
    df = load_m5_weekly(raw_dir=tmp_path, n_products=2, n_weeks=12, min_weekly_units=20)
    assert df["product_id"].nunique() == 2
    assert set(df["product_id"]) == {"FOODS_1_000", "HOBBIES_1_001"}
    assert df.groupby("product_id").size().eq(12).all()
    assert (df["units_sold"] == 35).all()
    assert df["week_num"].min() == 0 and df["week_num"].max() == 11


def test_partial_week_dropped_and_holiday_flag(tmp_path):
    write_m5(tmp_path)
    df = load_m5_weekly(raw_dir=tmp_path, n_products=1, n_weeks=12, min_weekly_units=20)
    assert df["is_holiday_week"].sum() == 1
    assert df["true_elasticity"].isna().all()
    assert set(df["category"]) <= {"Food", "Hobbies", "Household"}
