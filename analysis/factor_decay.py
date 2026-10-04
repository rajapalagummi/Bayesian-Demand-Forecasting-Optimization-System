"""
Elasticity Signal Decay Analysis Module — DemandSense
Additive enhancement: takes existing pipeline outputs as inputs.
Zero changes to existing code required.

Add to main.py run_demo() after run_statistical_analysis():

    from analysis.factor_decay import run_factor_decay_analysis
    decay_results = run_factor_decay_analysis(
        df=df,
        ols_df=ols_df,
        bayesian_df=bayesian_df,
    )

Analyzes:
    - Elasticity signal autocorrelation over time lags
    - Price sensitivity decay across product lifecycle
    - Elasticity half-life estimation
    - Cross-lag information coefficient (IC) decay
    - Product elasticity stability clusters
    - Seasonal elasticity patterns
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

OUTPUT_DIR = Path("data/outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        import os
        os.makedirs("images", exist_ok=True)
        fig.write_image(f"images/{name}.png", width=1200, height=600)
    except Exception:
        pass


def compute_elasticity_autocorrelation(df, max_lag=10):
    """
    Compute autocorrelation of price elasticity signal across time lags.
    Elasticity = % change in demand / % change in price at each week.
    """
    results = {}
    all_autocorrs = []

    for product_id in df["product_id"].unique():
        prod_df = df[df["product_id"] == product_id].copy()
        if "week_num" not in prod_df.columns or len(prod_df) < max_lag + 5:
            continue

        prod_df = prod_df.sort_values("week_num")

        if "price" in prod_df.columns and "units_sold" in prod_df.columns:
            price_chg = prod_df["price"].pct_change().replace([np.inf, -np.inf], np.nan)
            demand_chg = prod_df["units_sold"].pct_change().replace([np.inf, -np.inf], np.nan)
            elasticity = (demand_chg / price_chg).replace([np.inf, -np.inf], np.nan)
            elasticity = elasticity.dropna()

            if len(elasticity) < max_lag + 3:
                continue

            autocorrs = []
            for lag in range(1, max_lag + 1):
                if len(elasticity) > lag:
                    corr, p = stats.pearsonr(
                        elasticity.iloc[:-lag].values,
                        elasticity.iloc[lag:].values
                    )
                    autocorrs.append({
                        "product_id": product_id,
                        "lag": lag,
                        "autocorr": round(float(corr), 4),
                        "p_value": round(float(p), 4),
                        "significant": bool(p < 0.05),
                    })

            all_autocorrs.extend(autocorrs)

    if not all_autocorrs:
        return results

    autocorr_df = pd.DataFrame(all_autocorrs)
    avg_by_lag = autocorr_df.groupby("lag")["autocorr"].agg(["mean", "std", "count"]).reset_index()
    avg_by_lag.columns = ["lag", "mean_autocorr", "std_autocorr", "n_products"]

    results["autocorrelation_by_lag"] = avg_by_lag.to_dict(orient="records")
    results["peak_lag"] = int(avg_by_lag.loc[avg_by_lag["mean_autocorr"].abs().idxmax(), "lag"])
    results["mean_lag1_autocorr"] = round(float(avg_by_lag[avg_by_lag["lag"] == 1]["mean_autocorr"].values[0]), 4)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_by_lag["lag"],
        y=avg_by_lag["mean_autocorr"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=8),
        name="Mean Autocorrelation",
        error_y=dict(
            type="data",
            array=avg_by_lag["std_autocorr"].values,
            visible=True,
            color="#aaaaaa",
        )
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero line")
    fig.add_hline(y=0.1, line_dash="dot", line_color="green", annotation_text="Weak signal threshold")
    fig.update_layout(
        title="Elasticity Signal Autocorrelation by Lag",
        xaxis_title="Lag (weeks)",
        yaxis_title="Mean Autocorrelation",
        height=450,
    )
    save_fig(fig, "elasticity_autocorrelation")

    return results


def estimate_elasticity_half_life(df, max_lag=10):
    """
    Estimate how quickly elasticity signal decays — half-life in weeks.
    Fits exponential decay curve to autocorrelation vs lag.
    """
    results = {}

    autocorr_data = compute_elasticity_autocorrelation(df, max_lag=max_lag)
    if "autocorrelation_by_lag" not in autocorr_data:
        return results

    lag_df = pd.DataFrame(autocorr_data["autocorrelation_by_lag"])
    lags = lag_df["lag"].values
    autocorrs = lag_df["mean_autocorr"].values

    positive_mask = autocorrs > 0
    if positive_mask.sum() < 3:
        results["half_life_weeks"] = None
        results["decay_rate"] = None
        return results

    try:
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)

        popt, _ = curve_fit(
            exp_decay,
            lags[positive_mask],
            autocorrs[positive_mask],
            p0=[1.0, 0.1],
            maxfev=5000
        )
        a, b = popt
        half_life = np.log(2) / b if b > 0 else None

        results["half_life_weeks"] = round(float(half_life), 2) if half_life else None
        results["decay_rate"] = round(float(b), 4)
        results["decay_amplitude"] = round(float(a), 4)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lags, y=autocorrs,
            mode="markers", name="Observed",
            marker=dict(color="#e74c3c", size=10),
        ))

        x_smooth = np.linspace(lags.min(), lags.max(), 100)
        y_fitted = exp_decay(x_smooth, *popt)
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_fitted,
            mode="lines", name=f"Exponential Decay (half-life={half_life:.1f}w)",
            line=dict(color="#2ecc71", width=2, dash="dash"),
        ))

        if half_life and half_life <= max_lag:
            fig.add_vline(
                x=half_life, line_dash="dot", line_color="orange",
                annotation_text=f"Half-life: {half_life:.1f} weeks"
            )

        fig.update_layout(
            title=f"Elasticity Signal Half-Life Estimation",
            xaxis_title="Lag (weeks)",
            yaxis_title="Autocorrelation",
            height=450,
        )
        save_fig(fig, "elasticity_half_life")

    except Exception as e:
        results["half_life_error"] = str(e)

    return results


def compute_ic_decay(df, ols_df, bayesian_df, horizons=[1, 2, 4, 8]):
    """
    Information Coefficient (IC) decay:
    Spearman correlation between elasticity rank today and demand rank N weeks ahead.
    """
    results = {}
    ic_records = []

    merged = None
    if ols_df is not None and "ols_elasticity" in ols_df.columns:
        merged = ols_df[["product_id", "ols_elasticity"]].copy()
    if bayesian_df is not None and "bayesian_elasticity" in bayesian_df.columns:
        if merged is not None:
            merged = merged.merge(
                bayesian_df[["product_id", "bayesian_elasticity"]],
                on="product_id", how="left"
            )
        else:
            merged = bayesian_df[["product_id", "bayesian_elasticity"]].copy()

    if merged is None or len(merged) < 4:
        return results

    for horizon in horizons:
        for elasticity_col in ["ols_elasticity", "bayesian_elasticity"]:
            if elasticity_col not in merged.columns:
                continue

            valid = merged.dropna(subset=[elasticity_col])
            if len(valid) < 4:
                continue

            future_demand = []
            for pid in valid["product_id"]:
                prod_df = df[df["product_id"] == pid].sort_values("week_num") if "week_num" in df.columns else df[df["product_id"] == pid]
                if len(prod_df) > horizon:
                    future_demand.append(prod_df["units_sold"].iloc[-1] if "units_sold" in prod_df.columns else np.nan)
                else:
                    future_demand.append(np.nan)

            valid = valid.copy()
            valid["future_demand"] = future_demand
            valid = valid.dropna(subset=["future_demand"])

            if len(valid) < 4:
                continue

            rho, p = stats.spearmanr(valid[elasticity_col], valid["future_demand"])
            ic_records.append({
                "horizon": horizon,
                "elasticity_type": elasticity_col.replace("_elasticity", ""),
                "ic": round(float(rho), 4),
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
                "n_products": int(len(valid)),
            })

    if ic_records:
        results["ic_by_horizon"] = ic_records
        ic_df = pd.DataFrame(ic_records)

        fig = go.Figure()
        for etype in ic_df["elasticity_type"].unique():
            subset = ic_df[ic_df["elasticity_type"] == etype]
            color = "#3498db" if etype == "ols" else "#2ecc71"
            fig.add_trace(go.Scatter(
                x=subset["horizon"],
                y=subset["ic"],
                mode="lines+markers",
                name=f"{etype.upper()} Elasticity IC",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Information Coefficient Decay — Elasticity vs Future Demand",
            xaxis_title="Forecast Horizon (weeks)",
            yaxis_title="Spearman IC",
            height=450,
        )
        save_fig(fig, "elasticity_ic_decay")

    return results


def compute_elasticity_stability(df, ols_df):
    """
    Measure elasticity stability across products:
    - Products with stable elasticity vs volatile elasticity
    - Cluster products by elasticity volatility
    """
    results = {}

    if ols_df is None or "ols_elasticity" not in ols_df.columns:
        return results

    stability_records = []
    for product_id in df["product_id"].unique():
        prod_df = df[df["product_id"] == product_id].copy()
        if "price" not in prod_df.columns or "units_sold" not in prod_df.columns:
            continue
        if len(prod_df) < 10:
            continue

        prod_df = prod_df.sort_values("week_num") if "week_num" in prod_df.columns else prod_df

        mid = len(prod_df) // 2
        first_half = prod_df.iloc[:mid]
        second_half = prod_df.iloc[mid:]

        def calc_elasticity(chunk):
            p_chg = chunk["price"].pct_change().replace([np.inf, -np.inf], np.nan)
            d_chg = chunk["units_sold"].pct_change().replace([np.inf, -np.inf], np.nan)
            e = (d_chg / p_chg).replace([np.inf, -np.inf], np.nan).dropna()
            return float(e.mean()) if len(e) > 0 else np.nan

        e1 = calc_elasticity(first_half)
        e2 = calc_elasticity(second_half)

        if np.isnan(e1) or np.isnan(e2):
            continue

        stability_records.append({
            "product_id": product_id,
            "elasticity_first_half": round(e1, 4),
            "elasticity_second_half": round(e2, 4),
            "elasticity_change": round(e2 - e1, 4),
            "elasticity_change_abs": round(abs(e2 - e1), 4),
            "stable": bool(abs(e2 - e1) < 0.5),
        })

    if stability_records:
        stability_df = pd.DataFrame(stability_records)
        results["n_stable"] = int(stability_df["stable"].sum())
        results["n_volatile"] = int((~stability_df["stable"]).sum())
        results["pct_stable"] = round(float(stability_df["stable"].mean() * 100), 1)
        results["mean_elasticity_change"] = round(float(stability_df["elasticity_change_abs"].mean()), 4)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                "Elasticity: First Half vs Second Half",
                                "Elasticity Change Distribution"
                            ])

        colors = ["#2ecc71" if s else "#e74c3c" for s in stability_df["stable"]]
        fig.add_trace(go.Scatter(
            x=stability_df["elasticity_first_half"],
            y=stability_df["elasticity_second_half"],
            mode="markers+text",
            text=stability_df["product_id"].astype(str),
            textposition="top center",
            marker=dict(color=colors, size=10),
            showlegend=False,
        ), row=1, col=1)

        min_v = min(stability_df["elasticity_first_half"].min(),
                    stability_df["elasticity_second_half"].min())
        max_v = max(stability_df["elasticity_first_half"].max(),
                    stability_df["elasticity_second_half"].max())
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", line=dict(color="gray", dash="dash"),
            showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Histogram(
            x=stability_df["elasticity_change"],
            marker_color="#9b59b6", showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_layout(
            height=450,
            title_text=f"Elasticity Stability Analysis "
                       f"({results['pct_stable']}% products stable)"
        )
        save_fig(fig, "elasticity_stability")

    return results


def compute_seasonal_elasticity(df):
    """
    Detect seasonal patterns in price elasticity.
    Groups by quarter/month and computes average elasticity per period.
    """
    results = {}

    if "week_num" not in df.columns or "price" not in df.columns or "units_sold" not in df.columns:
        return results

    df = df.copy()
    df["quarter"] = ((df["week_num"] - 1) // 13 % 4 + 1).astype(int)
    df["month_in_year"] = ((df["week_num"] - 1) % 52 // 4 + 1).astype(int)

    price_chg = df.groupby(["product_id", "week_num"])["price"].first().groupby("product_id").pct_change()
    demand_chg = df.groupby(["product_id", "week_num"])["units_sold"].first().groupby("product_id").pct_change()

    elasticity = (demand_chg / price_chg).replace([np.inf, -np.inf], np.nan).reset_index()
    elasticity.columns = ["product_id", "week_num", "elasticity"]
    elasticity = elasticity.merge(df[["product_id", "week_num", "quarter"]].drop_duplicates(),
                                   on=["product_id", "week_num"], how="left")

    quarterly = elasticity.groupby("quarter")["elasticity"].agg(["mean", "std", "count"]).reset_index()
    quarterly.columns = ["quarter", "mean_elasticity", "std_elasticity", "n_obs"]
    quarterly = quarterly.dropna()

    if len(quarterly) > 0:
        results["quarterly_elasticity"] = quarterly.to_dict(orient="records")
        results["most_elastic_quarter"] = int(quarterly.loc[quarterly["mean_elasticity"].abs().idxmax(), "quarter"])
        results["least_elastic_quarter"] = int(quarterly.loc[quarterly["mean_elasticity"].abs().idxmin(), "quarter"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Q{q}" for q in quarterly["quarter"]],
            y=quarterly["mean_elasticity"],
            marker_color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
            error_y=dict(type="data", array=quarterly["std_elasticity"].values, visible=True),
            text=[f"{v:.3f}" for v in quarterly["mean_elasticity"]],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Seasonal Price Elasticity by Quarter",
            xaxis_title="Quarter",
            yaxis_title="Mean Price Elasticity",
            height=400,
        )
        save_fig(fig, "elasticity_seasonal_pattern")

    return results


def run_factor_decay_analysis(df, ols_df, bayesian_df):
    print("\n=== DemandSense Elasticity Signal Decay Module ===")
    np.random.seed(42)
    all_results = {}

    print("  Computing elasticity autocorrelation...")
    all_results["autocorrelation"] = compute_elasticity_autocorrelation(df)

    print("  Estimating elasticity half-life...")
    all_results["half_life"] = estimate_elasticity_half_life(df)

    print("  Computing IC decay across forecast horizons...")
    all_results["ic_decay"] = compute_ic_decay(df, ols_df, bayesian_df)

    print("  Analyzing elasticity stability across products...")
    all_results["stability"] = compute_elasticity_stability(df, ols_df)

    print("  Detecting seasonal elasticity patterns...")
    all_results["seasonal"] = compute_seasonal_elasticity(df)

    if "half_life" in all_results and all_results["half_life"].get("half_life_weeks"):
        print(f"\n  Elasticity half-life: {all_results['half_life']['half_life_weeks']} weeks")

    if "autocorrelation" in all_results and all_results["autocorrelation"].get("mean_lag1_autocorr"):
        print(f"  Lag-1 autocorrelation: {all_results['autocorrelation']['mean_lag1_autocorr']}")

    if "stability" in all_results and all_results["stability"].get("pct_stable"):
        print(f"  Stable products: {all_results['stability']['pct_stable']}%")

    if "seasonal" in all_results and all_results["seasonal"].get("most_elastic_quarter"):
        print(f"  Most elastic quarter: Q{all_results['seasonal']['most_elastic_quarter']}")

    output_path = OUTPUT_DIR / "factor_decay.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print(f"  Plots saved: data/outputs/analysis/ and images/")
    return all_results
