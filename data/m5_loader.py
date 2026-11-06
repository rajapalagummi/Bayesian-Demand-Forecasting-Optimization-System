import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CATEGORY = {"FOODS": "Food", "HOUSEHOLD": "Household", "HOBBIES": "Hobbies"}


def weekly_calendar(raw_dir):
    cal = pd.read_csv(raw_dir / "calendar.csv", usecols=["date", "wm_yr_wk", "d", "month", "year", "event_name_1"])
    weeks = cal.groupby("wm_yr_wk").agg(
        week_start=("date", "min"),
        month=("month", "first"),
        year=("year", "first"),
        is_holiday_week=("event_name_1", lambda s: int(s.notna().any())),
    ).reset_index()
    return cal[["d", "wm_yr_wk"]], weeks


def weekly_units(raw_dir, store, day_to_week):
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"] == store]
    days = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=["item_id", "dept_id", "cat_id"], value_vars=days, var_name="d", value_name="units")
    long = long.merge(day_to_week, on="d")
    weekly = long.groupby(["item_id", "dept_id", "cat_id", "wm_yr_wk"]).agg(units_sold=("units", "sum"), days=("units", "size")).reset_index()
    return weekly[weekly["days"] == 7].drop(columns="days")


def load_m5_weekly(raw_dir=RAW, store="CA_1", n_products=10, n_weeks=104, min_weekly_units=20):
    day_to_week, weeks = weekly_calendar(raw_dir)
    weekly = weekly_units(raw_dir, store, day_to_week)
    last_weeks = sorted(weekly["wm_yr_wk"].unique())[-n_weeks:]
    weekly = weekly[weekly["wm_yr_wk"].isin(last_weeks)]

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[(prices["store_id"] == store) & prices["wm_yr_wk"].isin(last_weeks)]
    df = weekly.merge(prices[["item_id", "wm_yr_wk", "sell_price"]], on=["item_id", "wm_yr_wk"])

    stats = df.groupby("item_id").agg(
        n=("sell_price", "size"),
        mean_units=("units_sold", "mean"),
        price_cv=("sell_price", lambda p: p.std() / p.mean()),
    )
    chosen = stats[(stats["n"] == n_weeks) & (stats["mean_units"] >= min_weekly_units)]
    chosen = chosen.sort_values("price_cv", ascending=False).head(n_products).index

    df = df[df["item_id"].isin(chosen)].merge(weeks, on="wm_yr_wk")
    df["week_num"] = df["wm_yr_wk"].map({w: i for i, w in enumerate(last_weeks)})
    df = df.rename(columns={"item_id": "product_id", "dept_id": "subcategory", "sell_price": "price"})
    df["category"] = df["cat_id"].map(CATEGORY)
    df["base_price"] = df.groupby("product_id")["price"].transform("median").round(2)
    df["revenue"] = (df["price"] * df["units_sold"]).round(2)
    df["true_elasticity"] = np.nan
    cols = ["product_id", "category", "subcategory", "week_start", "price", "base_price", "units_sold",
            "revenue", "true_elasticity", "week_num", "year", "month", "is_holiday_week"]
    return df[cols].sort_values(["product_id", "week_num"]).reset_index(drop=True)


def selection_summary(df):
    g = df.groupby(["product_id", "category"]).agg(
        weeks=("week_num", "nunique"),
        mean_units=("units_sold", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
    ).round(2).reset_index()
    return g
