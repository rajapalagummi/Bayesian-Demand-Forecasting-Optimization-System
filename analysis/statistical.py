"""
Statistical Analysis Module — DemandSense
Additive enhancement: takes existing pipeline outputs as inputs.
Zero changes to existing code required.

Add to main.py run_demo() after Stage 4, before results dict:

    from analysis.statistical import run_statistical_analysis
    stat_results = run_statistical_analysis(
        df=df,
        ols_df=ols_df,
        bayesian_df=bayesian_df,
        forecast_results=forecast_results,
        opt_results=opt_results,
    )
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


def describe_series(series, label):
    s = series.dropna()
    if len(s) < 4:
        return {}
    _, p_ks = stats.kstest(s, stats.norm(loc=s.mean(), scale=s.std()).cdf)
    _, p_jb = stats.jarque_bera(s)
    return {
        "label": label,
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "q25": round(float(s.quantile(0.25)), 4),
        "q75": round(float(s.quantile(0.75)), 4),
        "is_normal_ks": bool(p_ks > 0.05),
        "is_normal_jb": bool(p_jb > 0.05),
        "ks_p": round(float(p_ks), 4),
        "jb_p": round(float(p_jb), 4),
    }


def detect_outliers(series, label):
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    z = np.abs(stats.zscore(s))
    z_outliers = s[z > 3]
    return {
        "label": label,
        "iqr_outliers": int(len(outliers)),
        "iqr_outlier_pct": round(float(len(outliers) / len(s) * 100), 2),
        "zscore_outliers": int(len(z_outliers)),
        "lower_fence": round(float(lower), 4),
        "upper_fence": round(float(upper), 4),
    }


def bootstrap_ci(series, n_bootstrap=1000, ci=0.95):
    s = series.dropna().values
    if len(s) < 5:
        return {}
    boot_means = [np.mean(np.random.choice(s, len(s), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return {
        "observed_mean": round(float(s.mean()), 4),
        "ci_lower": round(float(np.percentile(boot_means, alpha / 2 * 100)), 4),
        "ci_upper": round(float(np.percentile(boot_means, (1 - alpha / 2) * 100)), 4),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def elasticity_distribution_analysis(ols_df, bayesian_df):
    results = {}

    if ols_df is not None and "ols_elasticity" in ols_df.columns:
        ols_e = ols_df["ols_elasticity"].dropna()
        results["ols"] = describe_series(ols_e, "OLS Elasticity")
        results["ols_outliers"] = detect_outliers(ols_e, "OLS Elasticity")
        results["ols_bootstrap_ci"] = bootstrap_ci(ols_e)

    if bayesian_df is not None and "bayesian_elasticity" in bayesian_df.columns:
        bay_e = bayesian_df["bayesian_elasticity"].dropna()
        results["bayesian"] = describe_series(bay_e, "Bayesian Elasticity")
        results["bayesian_outliers"] = detect_outliers(bay_e, "Bayesian Elasticity")
        results["bayesian_bootstrap_ci"] = bootstrap_ci(bay_e)

    if (ols_df is not None and bayesian_df is not None and
            "ols_elasticity" in ols_df.columns and
            "bayesian_elasticity" in bayesian_df.columns):

        merged = pd.merge(
            ols_df[["product_id", "ols_elasticity"]],
            bayesian_df[["product_id", "bayesian_elasticity"]],
            on="product_id", how="inner"
        ).dropna()

        if len(merged) > 3:
            corr, p = stats.pearsonr(
                merged["ols_elasticity"], merged["bayesian_elasticity"]
            )
            t_stat, t_p = stats.ttest_rel(
                merged["ols_elasticity"], merged["bayesian_elasticity"]
            )
            results["bayesian_vs_ols_correlation"] = {
                "pearson_r": round(float(corr), 4),
                "pearson_p": round(float(p), 4),
                "paired_ttest_t": round(float(t_stat), 4),
                "paired_ttest_p": round(float(t_p), 4),
                "significant_difference": bool(t_p < 0.05),
                "n_products": int(len(merged)),
            }

            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=[
                                    "OLS vs Bayesian Elasticity Scatter",
                                    "Elasticity Distribution Comparison",
                                    "Difference (Bayesian - OLS)"
                                ])

            fig.add_trace(go.Scatter(
                x=merged["ols_elasticity"], y=merged["bayesian_elasticity"],
                mode="markers", marker=dict(color="#3498db", size=8),
                name="Products", showlegend=False,
            ), row=1, col=1)
            min_v = min(merged["ols_elasticity"].min(), merged["bayesian_elasticity"].min())
            max_v = max(merged["ols_elasticity"].max(), merged["bayesian_elasticity"].max())
            fig.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode="lines", line=dict(color="red", dash="dash"),
                name="y=x", showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Histogram(
                x=merged["ols_elasticity"], name="OLS",
                marker_color="#e74c3c", opacity=0.6, showlegend=True,
            ), row=1, col=2)
            fig.add_trace(go.Histogram(
                x=merged["bayesian_elasticity"], name="Bayesian",
                marker_color="#2ecc71", opacity=0.6, showlegend=True,
            ), row=1, col=2)

            diff = merged["bayesian_elasticity"] - merged["ols_elasticity"]
            fig.add_trace(go.Histogram(
                x=diff, name="Difference", marker_color="#9b59b6",
                showlegend=False,
            ), row=1, col=3)
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=3)

            fig.update_layout(
                height=450,
                title_text=f"Bayesian vs OLS Elasticity Analysis "
                           f"(r={corr:.3f}, paired t-test p={t_p:.4f})"
            )
            save_fig(fig, "elasticity_bayesian_vs_ols")

    if bayesian_df is not None and "ci_width" in bayesian_df.columns:
        ci_w = bayesian_df["ci_width"].dropna()
        results["ci_width"] = describe_series(ci_w, "Bayesian CI Width")
        fig = px.histogram(
            bayesian_df.dropna(subset=["ci_width"]),
            x="ci_width",
            title="Bayesian Credible Interval Width Distribution",
            labels={"ci_width": "CI Width (95%)"},
            color_discrete_sequence=["#2ecc71"],
        )
        fig.add_vline(
            x=float(ci_w.mean()), line_dash="dash",
            annotation_text=f"Mean: {ci_w.mean():.3f}"
        )
        save_fig(fig, "elasticity_ci_width_distribution")

    return results


def forecast_analysis(forecast_results):
    if forecast_results is None or len(forecast_results) == 0:
        return {}

    results = {}
    fr = forecast_results.copy()

    for col in ["sarima_mape", "elasticity_mape", "mape_improvement_pct"]:
        if col in fr.columns:
            results[col] = describe_series(fr[col], col)
            results[f"{col}_outliers"] = detect_outliers(fr[col], col)
            results[f"{col}_bootstrap_ci"] = bootstrap_ci(fr[col])

    if "sarima_mape" in fr.columns and "elasticity_mape" in fr.columns:
        paired = fr[["sarima_mape", "elasticity_mape"]].dropna()
        if len(paired) > 3:
            t_stat, p_val = stats.ttest_rel(
                paired["sarima_mape"], paired["elasticity_mape"]
            )
            u_stat, u_p = stats.mannwhitneyu(
                paired["sarima_mape"], paired["elasticity_mape"],
                alternative="two-sided"
            )
            results["forecast_significance"] = {
                "paired_ttest_t": round(float(t_stat), 4),
                "paired_ttest_p": round(float(p_val), 4),
                "mann_whitney_u": round(float(u_stat), 4),
                "mann_whitney_p": round(float(u_p), 4),
                "elasticity_wins": int(
                    (paired["elasticity_mape"] < paired["sarima_mape"]).sum()
                ),
                "sarima_wins": int(
                    (paired["sarima_mape"] < paired["elasticity_mape"]).sum()
                ),
                "significant": bool(p_val < 0.05),
            }

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                "MAPE Distribution: SARIMA vs Elasticity-Informed",
                                "MAPE Improvement per Product (%)"
                            ])

        fig.add_trace(go.Box(
            y=paired["sarima_mape"], name="SARIMA",
            marker_color="#e74c3c", boxmean=True,
        ), row=1, col=1)
        fig.add_trace(go.Box(
            y=paired["elasticity_mape"], name="Elasticity",
            marker_color="#2ecc71", boxmean=True,
        ), row=1, col=1)

        if "mape_improvement_pct" in fr.columns:
            improvement = fr["mape_improvement_pct"].dropna()
            colors = ["#2ecc71" if v > 0 else "#e74c3c"
                      for v in improvement.values]
            fig.add_trace(go.Bar(
                x=list(range(len(improvement))),
                y=improvement.values,
                marker_color=colors,
                name="MAPE Improvement %",
                showlegend=False,
            ), row=1, col=2)
            fig.add_hline(y=0, line_dash="dash", row=1, col=2)

        fig.update_layout(height=450, title_text="Forecast A/B Test Analysis")
        save_fig(fig, "forecast_ab_test_analysis")

    return results


def optimization_analysis(opt_results, df):
    if opt_results is None:
        return {}

    results = {}
    comp = opt_results.get("comparison", {})

    objectives = {
        "MIP": comp.get("mip_objective", 0),
        "Greedy": comp.get("greedy_objective", 0),
        "Equal Split": comp.get("equal_objective",
                                comp.get("mip_objective", 0) *
                                (1 - comp.get("mip_vs_equal_improvement_pct", 0) / 100)),
    }

    satisfaction = {
        "MIP": comp.get("mip_satisfaction_rate", 0),
        "Greedy": comp.get("greedy_satisfaction_rate", 0),
    }

    results["objectives"] = objectives
    results["satisfaction"] = satisfaction
    results["mip_improvement_pct"] = comp.get("mip_vs_greedy_improvement_pct", 0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Objective Value by Strategy", "Constraint Satisfaction Rate (%)"]
    )

    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    fig.add_trace(go.Bar(
        x=list(objectives.keys()),
        y=list(objectives.values()),
        marker_color=colors[:len(objectives)],
        text=[f"{v:.1f}" for v in objectives.values()],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=list(satisfaction.keys()),
        y=list(satisfaction.values()),
        marker_color=["#2ecc71", "#e74c3c"],
        text=[f"{v:.1f}%" for v in satisfaction.values()],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=100, line_dash="dash", line_color="blue",
                  annotation_text="100% target", row=1, col=2)

    fig.update_layout(
        height=450,
        title_text=f"MIP vs Greedy Optimization "
                   f"(MIP improves objective by "
                   f"{comp.get('mip_vs_greedy_improvement_pct', 0):.1f}%)"
    )
    save_fig(fig, "optimization_comparison")

    if df is not None and "price" in df.columns and "demand" in df.columns:
        demand_stats = df.groupby("product_id")["demand"].agg(
            ["mean", "std", "min", "max"]
        ).reset_index()
        demand_stats.columns = ["product_id", "mean_demand",
                                 "std_demand", "min_demand", "max_demand"]
        demand_stats["cv"] = demand_stats["std_demand"] / demand_stats["mean_demand"]
        results["demand_variability"] = describe_series(
            demand_stats["cv"], "Demand Coefficient of Variation"
        )

        fig2 = px.scatter(
            demand_stats, x="mean_demand", y="cv",
            title="Product Demand Variability — Mean vs Coefficient of Variation",
            labels={"mean_demand": "Mean Weekly Demand",
                    "cv": "Coefficient of Variation"},
            color="cv", color_continuous_scale="RdYlGn_r",
            hover_data=["product_id"],
        )
        save_fig(fig2, "demand_variability_scatter")

    return results


def price_demand_analysis(df):
    if df is None or "price" not in df.columns or "demand" not in df.columns:
        return {}

    results = {}

    corr, p = stats.pearsonr(
        df["price"].dropna(), df["demand"].dropna()
    )
    rho, rho_p = stats.spearmanr(
        df["price"].dropna(), df["demand"].dropna()
    )
    results["price_demand_correlation"] = {
        "pearson_r": round(float(corr), 4),
        "pearson_p": round(float(p), 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(rho_p), 4),
        "n_observations": int(len(df.dropna(subset=["price", "demand"]))),
    }

    results["price_stats"] = describe_series(df["price"].dropna(), "Price")
    results["demand_stats"] = describe_series(df["demand"].dropna(), "Demand")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Price Distribution", "Demand Distribution",
            "Price vs Demand Scatter", "Weekly Demand Trend"
        ]
    )

    fig.add_trace(go.Histogram(
        x=df["price"].dropna(), name="Price",
        marker_color="#3498db", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=df["demand"].dropna(), name="Demand",
        marker_color="#e74c3c", showlegend=False,
    ), row=1, col=2)

    sample = df.sample(min(2000, len(df)), random_state=42)
    fig.add_trace(go.Scatter(
        x=sample["price"], y=sample["demand"],
        mode="markers",
        marker=dict(color="#9b59b6", size=3, opacity=0.4),
        name="Price-Demand", showlegend=False,
    ), row=2, col=1)

    if "week" in df.columns:
        weekly = df.groupby("week")["demand"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=weekly["week"], y=weekly["demand"],
            mode="lines", line=dict(color="#2ecc71", width=2),
            name="Avg Weekly Demand", showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        title_text=f"Price-Demand Analysis "
                   f"(Pearson r={corr:.3f}, p={p:.4f})"
    )
    save_fig(fig, "price_demand_analysis")

    return results


def run_statistical_analysis(df, ols_df, bayesian_df, forecast_results, opt_results):
    print("\n=== DemandSense Statistical Analysis Module ===")
    np.random.seed(42)
    all_results = {}

    print("  Running price-demand correlation analysis...")
    all_results["price_demand"] = price_demand_analysis(df)

    print("  Running elasticity distribution analysis...")
    all_results["elasticity"] = elasticity_distribution_analysis(ols_df, bayesian_df)

    print("  Running forecast A/B test analysis...")
    all_results["forecast"] = forecast_analysis(forecast_results)

    print("  Running optimization analysis...")
    all_results["optimization"] = optimization_analysis(opt_results, df)

    if "elasticity" in all_results and "bayesian" in all_results["elasticity"]:
        bay = all_results["elasticity"]["bayesian"]
        print(f"\n  Bayesian Elasticity: mean={bay['mean']:.4f}, "
              f"skew={bay['skewness']:.4f}, normal={bay['is_normal_ks']}")

    if "forecast" in all_results and "forecast_significance" in all_results["forecast"]:
        sig = all_results["forecast"]["forecast_significance"]
        print(f"  Forecast significance: p={sig['paired_ttest_p']:.4f}, "
              f"elasticity wins={sig['elasticity_wins']}/{sig['elasticity_wins']+sig['sarima_wins']}")

    if "optimization" in all_results:
        imp = all_results["optimization"].get("mip_improvement_pct", 0)
        print(f"  MIP improvement over greedy: {imp:.2f}%")

    output_path = OUTPUT_DIR / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print(f"  Plots saved: data/outputs/analysis/ and images/")
    return all_results
