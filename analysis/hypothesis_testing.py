"""
Hypothesis Testing Module — DemandSense
Additive enhancement: takes existing pipeline outputs as inputs.
Zero changes to existing code required.

Add to main.py run_demo() after run_advanced_eda():

    from analysis.hypothesis_testing import run_hypothesis_testing
    hypothesis_results = run_hypothesis_testing(
        df=df,
        ols_df=ols_df,
        bayesian_df=bayesian_df,
        forecast_results=forecast_results,
        opt_results=opt_results,
    )

Covers:
    - Elasticity significance tests (t-test, Mann-Whitney, KS)
    - Bayesian vs OLS paired comparison
    - Forecast A/B test significance with effect sizes
    - MIP vs greedy optimization significance
    - Multiple comparison correction (Bonferroni + FDR)
    - Bootstrap confidence intervals
    - Power analysis for all tests
    - Factor significance scores with grades
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, kstest, norm
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


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return abs(float(np.mean(group1) - np.mean(group2))) / pooled_std


def effect_size_label(d):
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_power(n, effect_size, alpha=0.05):
    from scipy.stats import norm as norm_dist
    z_alpha = norm_dist.ppf(1 - alpha / 2)
    z_beta = effect_size * np.sqrt(n) - z_alpha
    return float(norm_dist.cdf(z_beta))


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, statistic=np.mean):
    if len(data) < 3:
        return None, None
    boots = [statistic(np.random.choice(data, len(data), replace=True))
             for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return (round(float(np.percentile(boots, alpha / 2 * 100)), 4),
            round(float(np.percentile(boots, (1 - alpha / 2) * 100)), 4))


def test_elasticity_significance(ols_df, bayesian_df):
    results = {}
    tests = []

    if ols_df is not None and "ols_elasticity" in ols_df.columns:
        ols_e = ols_df["ols_elasticity"].dropna().values

        t_stat, p_val = stats.ttest_1samp(ols_e, 0)
        d = abs(float(np.mean(ols_e))) / float(np.std(ols_e, ddof=1)) if np.std(ols_e, ddof=1) > 0 else 0
        power = compute_power(len(ols_e), d)
        ci_low, ci_high = bootstrap_ci(ols_e)

        tests.append({
            "test": "OLS Elasticity vs Zero",
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
            "effect_size": round(d, 4),
            "effect_label": effect_size_label(d),
            "power": round(power, 4),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "n": int(len(ols_e)),
        })

    if (ols_df is not None and bayesian_df is not None and
            "ols_elasticity" in ols_df.columns and
            "bayesian_elasticity" in bayesian_df.columns):

        merged = pd.merge(
            ols_df[["product_id", "ols_elasticity"]],
            bayesian_df[["product_id", "bayesian_elasticity"]],
            on="product_id", how="inner"
        ).dropna()

        if len(merged) >= 3:
            t_stat, p_val = ttest_rel(
                merged["ols_elasticity"], merged["bayesian_elasticity"]
            )
            u_stat, u_p = mannwhitneyu(
                merged["ols_elasticity"], merged["bayesian_elasticity"],
                alternative="two-sided"
            )
            ks_stat, ks_p = kstest(
                merged["ols_elasticity"] - merged["bayesian_elasticity"],
                "norm"
            )
            d = cohens_d(
                merged["ols_elasticity"].values,
                merged["bayesian_elasticity"].values
            )

            tests.append({
                "test": "Bayesian vs OLS Paired t-test",
                "statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant": bool(p_val < 0.05),
                "effect_size": round(d, 4),
                "effect_label": effect_size_label(d),
                "power": round(compute_power(len(merged), d), 4),
                "n": int(len(merged)),
            })

            tests.append({
                "test": "Bayesian vs OLS Mann-Whitney",
                "statistic": round(float(u_stat), 4),
                "p_value": round(float(u_p), 4),
                "significant": bool(u_p < 0.05),
                "effect_size": round(d, 4),
                "effect_label": effect_size_label(d),
                "n": int(len(merged)),
            })

    results["elasticity_tests"] = tests
    return results


def test_forecast_significance(forecast_results):
    results = {}

    if forecast_results is None or len(forecast_results) == 0:
        return results

    fr = forecast_results.copy()
    if "sarima_mape" not in fr.columns or "elasticity_mape" not in fr.columns:
        return results

    paired = fr[["sarima_mape", "elasticity_mape"]].dropna()
    if len(paired) < 3:
        return results

    t_stat, p_val = ttest_rel(paired["sarima_mape"], paired["elasticity_mape"])
    u_stat, u_p = mannwhitneyu(
        paired["sarima_mape"], paired["elasticity_mape"],
        alternative="two-sided"
    )
    d = cohens_d(paired["sarima_mape"].values, paired["elasticity_mape"].values)
    power = compute_power(len(paired), d)

    mape_diff = paired["sarima_mape"] - paired["elasticity_mape"]
    ci_low, ci_high = bootstrap_ci(mape_diff.values)

    results["forecast_tests"] = {
        "paired_ttest": {
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
            "effect_size": round(d, 4),
            "effect_label": effect_size_label(d),
            "power": round(power, 4),
            "n": int(len(paired)),
        },
        "mann_whitney": {
            "statistic": round(float(u_stat), 4),
            "p_value": round(float(u_p), 4),
            "significant": bool(u_p < 0.05),
        },
        "mape_improvement_ci": {
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "mean_improvement": round(float(mape_diff.mean()), 4),
        },
        "treatment_wins": int((paired["elasticity_mape"] < paired["sarima_mape"]).sum()),
        "control_wins": int((paired["sarima_mape"] < paired["elasticity_mape"]).sum()),
    }

    return results


def test_optimization_significance(opt_results, df):
    results = {}

    if opt_results is None:
        return results

    comp = opt_results.get("comparison", {})
    mip_obj = comp.get("mip_objective", 0)
    greedy_obj = comp.get("greedy_objective", 0)
    improvement = comp.get("mip_vs_greedy_improvement_pct", 0)
    satisfaction_diff = (comp.get("mip_satisfaction_rate", 0) -
                         comp.get("greedy_satisfaction_rate", 0))

    results["optimization_tests"] = {
        "mip_vs_greedy_improvement_pct": round(float(improvement), 4),
        "satisfaction_rate_improvement": round(float(satisfaction_diff), 4),
        "mip_dominates": bool(mip_obj < greedy_obj),
        "practical_significance": bool(improvement > 5.0),
        "satisfaction_significance": bool(satisfaction_diff > 20.0),
    }

    if df is not None and "demand" in df.columns:
        demand_vals = df["demand"].dropna().values
        t_stat, p_val = stats.ttest_1samp(demand_vals, np.mean(demand_vals))
        results["demand_stationarity"] = {
            "mean": round(float(np.mean(demand_vals)), 4),
            "std": round(float(np.std(demand_vals, ddof=1)), 4),
            "cv": round(float(np.std(demand_vals, ddof=1) / np.mean(demand_vals)), 4),
        }

    return results


def apply_multiple_comparison_correction(all_tests):
    p_values = []
    test_names = []

    for category, tests in all_tests.items():
        if isinstance(tests, list):
            for t in tests:
                if "p_value" in t:
                    p_values.append(t["p_value"])
                    test_names.append(f"{category}: {t.get('test', 'unknown')}")
        elif isinstance(tests, dict):
            for k, v in tests.items():
                if isinstance(v, dict) and "p_value" in v:
                    p_values.append(v["p_value"])
                    test_names.append(f"{category}: {k}")

    if len(p_values) == 0:
        return {}

    n_tests = len(p_values)
    bonferroni_threshold = 0.05 / n_tests

    sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
    fdr_rejected = []
    for rank, (idx, p) in enumerate(sorted_p):
        if p <= (rank + 1) / n_tests * 0.05:
            fdr_rejected.append(idx)

    results = {
        "total_tests": n_tests,
        "bonferroni_threshold": round(bonferroni_threshold, 6),
        "bonferroni_rejected": int(sum(1 for p in p_values if p < bonferroni_threshold)),
        "fdr_rejected": int(len(fdr_rejected)),
        "p_values": [round(p, 4) for p in p_values],
        "test_names": test_names,
    }

    return results


def compute_significance_scores(elasticity_tests, forecast_tests, opt_tests):
    scores = {}

    def score_test(p_value, effect_size, power, weight=1.0):
        sig_score = 1.0 if p_value < 0.05 else 0.0
        effect_score = min(effect_size / 0.8, 1.0)
        power_score = min(power / 0.8, 1.0) if power else 0.5
        return round(float(weight * (0.4 * sig_score + 0.4 * effect_score + 0.2 * power_score)), 4)

    if "elasticity_tests" in elasticity_tests:
        for t in elasticity_tests["elasticity_tests"]:
            name = t.get("test", "unknown")
            s = score_test(
                t.get("p_value", 1.0),
                t.get("effect_size", 0.0),
                t.get("power", 0.5)
            )
            grade = "A" if s >= 0.8 else "B" if s >= 0.6 else "C" if s >= 0.4 else "D"
            scores[name] = {"score": s, "grade": grade}

    if "forecast_tests" in forecast_tests:
        ft = forecast_tests["forecast_tests"]
        pt = ft.get("paired_ttest", {})
        s = score_test(
            pt.get("p_value", 1.0),
            pt.get("effect_size", 0.0),
            pt.get("power", 0.5)
        )
        grade = "A" if s >= 0.8 else "B" if s >= 0.6 else "C" if s >= 0.4 else "D"
        scores["Forecast A/B Test"] = {"score": s, "grade": grade}

    if "optimization_tests" in opt_tests:
        ot = opt_tests["optimization_tests"]
        practical = 1.0 if ot.get("practical_significance") else 0.0
        satisfaction = 1.0 if ot.get("satisfaction_significance") else 0.0
        s = round((practical + satisfaction) / 2, 4)
        grade = "A" if s >= 0.8 else "B" if s >= 0.6 else "C" if s >= 0.4 else "D"
        scores["MIP Optimization"] = {"score": s, "grade": grade}

    return scores


def generate_hypothesis_plots(elasticity_tests, forecast_tests,
                               correction_results, significance_scores):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Test p-values (-log10 scale)",
            "Effect Sizes by Test",
            "Multiple Comparison Correction",
            "Significance Scores (0-1)",
        ]
    )

    if "elasticity_tests" in elasticity_tests:
        tests = elasticity_tests["elasticity_tests"]
        names = [t.get("test", "")[:30] for t in tests]
        p_vals = [t.get("p_value", 1.0) for t in tests]
        effects = [t.get("effect_size", 0.0) for t in tests]

        if "forecast_tests" in forecast_tests:
            ft = forecast_tests["forecast_tests"].get("paired_ttest", {})
            names.append("Forecast A/B")
            p_vals.append(ft.get("p_value", 1.0))
            effects.append(ft.get("effect_size", 0.0))

        log_p = [-np.log10(max(p, 1e-10)) for p in p_vals]
        colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in p_vals]

        fig.add_trace(go.Bar(
            x=names, y=log_p,
            marker_color=colors,
            showlegend=False,
            text=[f"p={p:.3f}" for p in p_vals],
            textposition="outside",
        ), row=1, col=1)
        fig.add_hline(
            y=-np.log10(0.05), line_dash="dash",
            line_color="red", annotation_text="p=0.05",
            row=1, col=1
        )

        effect_colors = ["#2ecc71" if e >= 0.8 else "#f39c12" if e >= 0.5
                         else "#e74c3c" for e in effects]
        fig.add_trace(go.Bar(
            x=names, y=effects,
            marker_color=effect_colors,
            showlegend=False,
            text=[effect_size_label(e) for e in effects],
            textposition="outside",
        ), row=1, col=2)
        for threshold, label, color in [(0.2, "small", "gray"),
                                         (0.5, "medium", "orange"),
                                         (0.8, "large", "green")]:
            fig.add_hline(y=threshold, line_dash="dot",
                          line_color=color,
                          annotation_text=label,
                          row=1, col=2)

    if correction_results:
        categories = ["Total Tests", "Bonferroni\nRejected", "FDR\nRejected"]
        values = [
            correction_results.get("total_tests", 0),
            correction_results.get("bonferroni_rejected", 0),
            correction_results.get("fdr_rejected", 0),
        ]
        fig.add_trace(go.Bar(
            x=categories, y=values,
            marker_color=["#3498db", "#e74c3c", "#2ecc71"],
            showlegend=False,
            text=[str(v) for v in values],
            textposition="outside",
        ), row=2, col=1)

    if significance_scores:
        score_names = list(significance_scores.keys())
        score_vals = [v["score"] for v in significance_scores.values()]
        grades = [v["grade"] for v in significance_scores.values()]
        score_colors = ["#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.6
                        else "#e74c3c" for s in score_vals]
        fig.add_trace(go.Bar(
            x=score_names,
            y=score_vals,
            marker_color=score_colors,
            showlegend=False,
            text=[f"{g} ({s:.2f})" for g, s in zip(grades, score_vals)],
            textposition="outside",
        ), row=2, col=2)
        fig.add_hline(y=0.8, line_dash="dash",
                      line_color="green", annotation_text="Grade A threshold",
                      row=2, col=2)

    fig.update_layout(height=700, title_text="DemandSense Hypothesis Testing Summary")
    save_fig(fig, "hypothesis_testing_summary")


def run_hypothesis_testing(df, ols_df, bayesian_df, forecast_results, opt_results):
    print("\n=== DemandSense Hypothesis Testing Module ===")
    np.random.seed(42)
    all_results = {}

    print("  Running elasticity significance tests...")
    elasticity_tests = test_elasticity_significance(ols_df, bayesian_df)
    all_results.update(elasticity_tests)

    print("  Running forecast A/B significance tests...")
    forecast_tests = test_forecast_significance(forecast_results)
    all_results.update(forecast_tests)

    print("  Running optimization significance tests...")
    opt_tests = test_optimization_significance(opt_results, df)
    all_results.update(opt_tests)

    print("  Applying multiple comparison corrections...")
    correction_results = apply_multiple_comparison_correction({
        "elasticity": elasticity_tests.get("elasticity_tests", []),
        "forecast": forecast_tests,
        "optimization": opt_tests,
    })
    all_results["multiple_comparison_correction"] = correction_results

    print("  Computing significance scores...")
    significance_scores = compute_significance_scores(
        elasticity_tests, forecast_tests, opt_tests
    )
    all_results["significance_scores"] = significance_scores

    print("  Generating hypothesis testing visualizations...")
    generate_hypothesis_plots(
        elasticity_tests, forecast_tests,
        correction_results, significance_scores
    )

    if "elasticity_tests" in all_results:
        for t in all_results["elasticity_tests"]:
            print(f"\n  {t['test']}: "
                  f"p={t['p_value']:.4f}, "
                  f"sig={t['significant']}, "
                  f"effect={t['effect_label']}")

    if "forecast_tests" in all_results:
        ft = all_results["forecast_tests"]["paired_ttest"]
        print(f"\n  Forecast A/B: p={ft['p_value']:.4f}, "
              f"sig={ft['significant']}, power={ft['power']:.4f}")

    if correction_results:
        print(f"\n  Multiple comparison: "
              f"total={correction_results['total_tests']}, "
              f"Bonferroni={correction_results['bonferroni_rejected']}, "
              f"FDR={correction_results['fdr_rejected']}")

    if significance_scores:
        print("\n  Significance grades:")
        for name, info in significance_scores.items():
            print(f"    {name}: {info['grade']} ({info['score']:.2f})")

    output_path = OUTPUT_DIR / "hypothesis_testing.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print(f"  Plots saved: data/outputs/analysis/ and images/")
    return all_results
