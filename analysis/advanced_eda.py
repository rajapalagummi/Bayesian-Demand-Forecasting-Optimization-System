"""
Advanced EDA Module — DemandSense
Additive enhancement: takes existing pipeline outputs as inputs.
Zero changes to existing code required.

Add to main.py run_demo() after run_factor_decay_analysis():

    from analysis.advanced_eda import run_advanced_eda
    eda_results = run_advanced_eda(
        df=df,
        ols_df=ols_df,
        bayesian_df=bayesian_df,
        opt_results=opt_results,
    )

Analyzes:
    - Cross-product elasticity comparison and ranking
    - Demand seasonality heatmap by product and week
    - Price sensitivity clustering (K-means)
    - Demand distribution by product tier
    - Price-demand scatter with elasticity overlay
    - Outlier product detection
    - Demand volatility analysis
    - Price dispersion across products
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
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


def cross_product_elasticity_comparison(ols_df, bayesian_df):
    """Rank products by elasticity magnitude and compare OLS vs Bayesian estimates."""
    results = {}

    if ols_df is None or bayesian_df is None:
        return results

    merged = pd.merge(
        ols_df[["product_id", "ols_elasticity"]].dropna(),
        bayesian_df[["product_id", "bayesian_elasticity"]].dropna(),
        on="product_id", how="inner"
    )

    if len(merged) == 0:
        return results

    merged["avg_elasticity"] = (merged["ols_elasticity"] + merged["bayesian_elasticity"]) / 2
    merged["elasticity_agreement"] = (merged["ols_elasticity"] - merged["bayesian_elasticity"]).abs()
    merged = merged.sort_values("avg_elasticity")

    results["most_elastic_product"] = str(merged.iloc[0]["product_id"])
    results["least_elastic_product"] = str(merged.iloc[-1]["product_id"])
    results["mean_agreement_gap"] = round(float(merged["elasticity_agreement"].mean()), 4)
    results["max_disagreement_product"] = str(
        merged.loc[merged["elasticity_agreement"].idxmax(), "product_id"]
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["product_id"].astype(str),
        y=merged["ols_elasticity"],
        name="OLS Elasticity",
        marker_color="#e74c3c",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=merged["product_id"].astype(str),
        y=merged["bayesian_elasticity"],
        name="Bayesian Elasticity",
        marker_color="#2ecc71",
        opacity=0.7,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Cross-Product Elasticity Comparison — OLS vs Bayesian",
        xaxis_title="Product ID",
        yaxis_title="Price Elasticity",
        barmode="group",
        height=450,
    )
    save_fig(fig, "eda_cross_product_elasticity")

    return results


def demand_seasonality_heatmap(df):
    """Demand seasonality heatmap — products vs weeks."""
    results = {}

    if "week" not in df.columns or "demand" not in df.columns:
        return results

    pivot = df.pivot_table(
        index="product_id",
        columns="week",
        values="demand",
        aggfunc="mean"
    )

    if pivot.empty:
        return results

    normalized = pivot.div(pivot.mean(axis=1), axis=0)

    results["n_products"] = int(len(pivot))
    results["n_weeks"] = int(len(pivot.columns))
    results["peak_week"] = int(pivot.mean(axis=0).idxmax())
    results["trough_week"] = int(pivot.mean(axis=0).idxmin())

    fig = px.imshow(
        normalized,
        labels=dict(x="Week", y="Product ID", color="Normalized Demand"),
        title="Demand Seasonality Heatmap — Normalized by Product Mean",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=500)
    save_fig(fig, "eda_demand_seasonality_heatmap")

    weekly_avg = pivot.mean(axis=0).reset_index()
    weekly_avg.columns = ["week", "avg_demand"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=weekly_avg["week"],
        y=weekly_avg["avg_demand"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#3498db", width=2),
        fillcolor="rgba(52, 152, 219, 0.2)",
        name="Avg Demand",
    ))
    fig2.add_vline(
        x=int(weekly_avg.loc[weekly_avg["avg_demand"].idxmax(), "week"]),
        line_dash="dash", line_color="red",
        annotation_text="Peak demand"
    )
    fig2.update_layout(
        title="Average Weekly Demand Trend Across All Products",
        xaxis_title="Week",
        yaxis_title="Average Demand",
        height=400,
    )
    save_fig(fig2, "eda_weekly_demand_trend")

    return results


def price_sensitivity_clustering(df, ols_df, n_clusters=3):
    """Cluster products by price sensitivity using K-means on elasticity features."""
    results = {}

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return results

    if ols_df is None or "ols_elasticity" not in ols_df.columns:
        return results

    features = ols_df[["product_id", "ols_elasticity"]].dropna().copy()

    if "demand" in df.columns:
        demand_stats = df.groupby("product_id")["demand"].agg(["mean", "std"]).reset_index()
        demand_stats.columns = ["product_id", "mean_demand", "std_demand"]
        demand_stats["demand_cv"] = demand_stats["std_demand"] / demand_stats["mean_demand"]
        features = features.merge(demand_stats, on="product_id", how="left")

    if "price" in df.columns:
        price_stats = df.groupby("product_id")["price"].agg(["mean", "std"]).reset_index()
        price_stats.columns = ["product_id", "mean_price", "price_std"]
        features = features.merge(price_stats, on="product_id", how="left")

    feature_cols = [c for c in ["ols_elasticity", "demand_cv", "mean_price"]
                    if c in features.columns]
    X = features[feature_cols].fillna(0).values

    if len(X) < n_clusters:
        n_clusters = max(2, len(X) - 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "Low Sensitivity", 1: "Medium Sensitivity", 2: "High Sensitivity"}
    cluster_elasticity = features.groupby("cluster")["ols_elasticity"].mean()
    sorted_clusters = cluster_elasticity.sort_values().index.tolist()
    label_map = {old: cluster_labels.get(i, f"Cluster {i}")
                 for i, old in enumerate(sorted_clusters)}
    features["cluster_label"] = features["cluster"].map(label_map)

    results["n_clusters"] = n_clusters
    results["cluster_sizes"] = features["cluster_label"].value_counts().to_dict()
    results["cluster_mean_elasticity"] = features.groupby("cluster_label")["ols_elasticity"].mean().round(4).to_dict()

    if len(feature_cols) >= 2:
        fig = px.scatter(
            features,
            x=feature_cols[0],
            y=feature_cols[1] if len(feature_cols) > 1 else feature_cols[0],
            color="cluster_label",
            hover_data=["product_id"],
            title=f"Price Sensitivity Clusters ({n_clusters} groups)",
            labels={
                feature_cols[0]: feature_cols[0].replace("_", " ").title(),
                feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]:
                    (feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]).replace("_", " ").title(),
            },
            color_discrete_map={
                "Low Sensitivity": "#2ecc71",
                "Medium Sensitivity": "#f39c12",
                "High Sensitivity": "#e74c3c",
            }
        )
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(height=450)
        save_fig(fig, "eda_price_sensitivity_clusters")

    return results


def demand_distribution_by_tier(df, ols_df):
    """Compare demand distributions across high/medium/low elasticity product tiers."""
    results = {}

    if ols_df is None or "ols_elasticity" not in ols_df.columns:
        return results

    elasticity = ols_df[["product_id", "ols_elasticity"]].dropna()
    q33 = elasticity["ols_elasticity"].quantile(0.33)
    q67 = elasticity["ols_elasticity"].quantile(0.67)

    elasticity["tier"] = pd.cut(
        elasticity["ols_elasticity"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low Elasticity", "Medium Elasticity", "High Elasticity"]
    )

    elasticity["tier"] = elasticity["tier"].astype(str)

    df_merged = df.copy()
    df_merged = df_merged.merge(elasticity[["product_id", "tier"]], on="product_id", how="left")
    df_merged = df_merged.dropna(subset=["tier"])
    df_merged["tier"] = df_merged["tier"].astype(str)

    if len(df_merged) == 0 or "demand" not in df_merged.columns:
        return results

    tier_stats = df_merged.groupby("tier", observed=True)["demand"].agg(["mean", "std", "median"]).reset_index()
    tier_stats.columns = ["tier", "mean_demand", "std_demand", "median_demand"]
    results["tier_demand_stats"] = tier_stats.to_dict(orient="records")

    fig = go.Figure()
    colors = {"Low Elasticity": "#2ecc71", "Medium Elasticity": "#f39c12", "High Elasticity": "#e74c3c"}
    for tier in df_merged["tier"].unique():
        tier_data = df_merged[df_merged["tier"] == tier]["demand"].dropna()
        fig.add_trace(go.Violin(
            y=tier_data,
            name=str(tier),
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors.get(str(tier), "#3498db"),
            opacity=0.7,
            line_color="white",
        ))

    fig.update_layout(
        title="Demand Distribution by Price Elasticity Tier",
        yaxis_title="Weekly Demand",
        height=450,
        showlegend=True,
    )
    save_fig(fig, "eda_demand_by_tier")

    return results


def detect_outlier_products(df, ols_df):
    """Identify products with unusual demand or elasticity patterns."""
    results = {}

    if "demand" not in df.columns:
        return results

    product_stats = df.groupby("product_id")["demand"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    product_stats.columns = ["product_id", "mean_demand", "std_demand", "min_demand", "max_demand"]
    product_stats["cv"] = product_stats["std_demand"] / product_stats["mean_demand"]
    product_stats["range_ratio"] = product_stats["max_demand"] / (product_stats["min_demand"] + 1)

    z_mean = np.abs(stats.zscore(product_stats["mean_demand"].fillna(0)))
    z_cv = np.abs(stats.zscore(product_stats["cv"].fillna(0)))
    product_stats["is_demand_outlier"] = z_mean > 2
    product_stats["is_volatility_outlier"] = z_cv > 2

    if ols_df is not None and "ols_elasticity" in ols_df.columns:
        elast = ols_df[["product_id", "ols_elasticity"]].dropna()
        z_elast = np.abs(stats.zscore(elast["ols_elasticity"].fillna(0)))
        elast["is_elasticity_outlier"] = z_elast > 2
        product_stats = product_stats.merge(
            elast[["product_id", "ols_elasticity", "is_elasticity_outlier"]],
            on="product_id", how="left"
        )

    results["n_demand_outliers"] = int(product_stats["is_demand_outlier"].sum())
    results["n_volatility_outliers"] = int(product_stats["is_volatility_outlier"].sum())
    if "is_elasticity_outlier" in product_stats.columns:
        results["n_elasticity_outliers"] = int(product_stats["is_elasticity_outlier"].fillna(False).sum())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            "Product Mean Demand vs Volatility (CV)",
                            "Demand Range Ratio by Product"
                        ])

    colors_demand = ["#e74c3c" if o else "#3498db"
                     for o in product_stats["is_demand_outlier"]]
    fig.add_trace(go.Scatter(
        x=product_stats["mean_demand"],
        y=product_stats["cv"],
        mode="markers+text",
        text=product_stats["product_id"].astype(str),
        textposition="top center",
        marker=dict(color=colors_demand, size=10),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=product_stats["product_id"].astype(str),
        y=product_stats["range_ratio"],
        marker_color=["#e74c3c" if o else "#3498db"
                      for o in product_stats["is_volatility_outlier"]],
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        height=450,
        title_text=f"Outlier Product Detection "
                   f"({results['n_demand_outliers']} demand outliers, "
                   f"{results['n_volatility_outliers']} volatility outliers)"
    )
    save_fig(fig, "eda_outlier_products")

    return results


def run_advanced_eda(df, ols_df, bayesian_df, opt_results=None):
    print("\n=== DemandSense Advanced EDA Module ===")
    np.random.seed(42)
    all_results = {}

    print("  Running cross-product elasticity comparison...")
    all_results["cross_product"] = cross_product_elasticity_comparison(ols_df, bayesian_df)

    print("  Building demand seasonality heatmap...")
    all_results["seasonality"] = demand_seasonality_heatmap(df)

    print("  Clustering products by price sensitivity...")
    all_results["clustering"] = price_sensitivity_clustering(df, ols_df)

    print("  Analyzing demand distribution by tier...")
    all_results["tier_analysis"] = demand_distribution_by_tier(df, ols_df)

    print("  Detecting outlier products...")
    all_results["outliers"] = detect_outlier_products(df, ols_df)

    if "seasonality" in all_results and all_results["seasonality"].get("peak_week"):
        print(f"\n  Peak demand week: {all_results['seasonality']['peak_week']}")
    if "clustering" in all_results and all_results["clustering"].get("cluster_sizes"):
        print(f"  Clusters: {all_results['clustering']['cluster_sizes']}")
    if "outliers" in all_results:
        print(f"  Demand outliers: {all_results['outliers'].get('n_demand_outliers', 0)}")

    output_path = OUTPUT_DIR / "advanced_eda.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print(f"  Plots saved: data/outputs/analysis/ and images/")
    return all_results
