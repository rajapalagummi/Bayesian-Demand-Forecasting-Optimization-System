"""
DemandSense — Bayesian Price Elasticity Model
Estimates price elasticity as a posterior distribution using PyMC.

Model: log(Q) = alpha + beta * log(P) + gamma * X + epsilon
where beta = price elasticity (posterior distribution, not point estimate)

Key advantage over OLS: provides full uncertainty quantification,
especially valuable for products with sparse observations (<50 weekly data points).

Compares Bayesian credible intervals vs OLS confidence intervals to
quantify the benefit of Bayesian approach on sparse vs dense products.
"""
import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_ols_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """OLS baseline: log-log regression per product."""
    results = []
    for product_id, group in df.groupby("product_id"):
        if len(group) < 10:
            continue
        X = group["log_price"].values
        y = group["log_demand"].values
        slope, intercept, r, p, se = stats.linregress(X, y)
        t_critical = stats.t.ppf(0.975, df=len(group) - 2)
        ci_width = t_critical * se
        results.append({
            "product_id": product_id,
            "category": group["category"].iloc[0],
            "ols_elasticity": round(slope, 4),
            "ols_ci_lower": round(slope - ci_width, 4),
            "ols_ci_upper": round(slope + ci_width, 4),
            "ols_ci_width": round(2 * ci_width, 4),
            "ols_r2": round(r ** 2, 4),
            "n_obs": len(group),
            "true_elasticity": group["true_elasticity"].iloc[0],
        })
    return pd.DataFrame(results)


def fit_bayesian_elasticity_single(product_df: pd.DataFrame,
                                    product_id: str) -> dict:
    """
    Fit Bayesian log-log elasticity model for a single product.
    Prior: beta ~ Normal(-1, 1) — weakly informative, centered on -1 (unit elastic)
    """
    log_price = product_df["log_price"].values
    log_demand = product_df["log_demand"].values
    n = len(log_price)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=np.mean(log_demand), sigma=2)
        beta = pm.Normal("beta", mu=-1.0, sigma=1.0)

        if product_df[["is_holiday_week"]].shape[1] > 0:
            gamma = pm.Normal("gamma", mu=0, sigma=0.5)
            holiday = product_df["is_holiday_week"].values
            mu = alpha + beta * log_price + gamma * holiday
        else:
            mu = alpha + beta * log_price

        sigma = pm.HalfNormal("sigma", sigma=0.5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=log_demand)

        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=2,
            progressbar=False,
            random_seed=42,
        )

    beta_posterior = trace.posterior["beta"].values.flatten()
    beta_samples = beta_posterior
    hdi_result = az.hdi(beta_samples, prob=0.95)
    hdi = hdi_result if isinstance(hdi_result, np.ndarray) else np.array([hdi_result[0], hdi_result[1]])

    return {
        "product_id": product_id,
        "bayesian_elasticity": float(np.median(beta_posterior)),
        "bayesian_mean": float(np.mean(beta_posterior)),
        "bayesian_ci_lower": float(hdi[0]),
        "bayesian_ci_upper": float(hdi[1]),
        "bayesian_ci_width": float(hdi[1] - hdi[0]),
        "bayesian_std": float(np.std(beta_posterior)),
        "n_obs": n,
    }


def fit_bayesian_elasticity_all(df: pd.DataFrame,
                                 max_products: int = 50) -> pd.DataFrame:
    """Fit Bayesian elasticity model for all products."""
    results = []
    products = df["product_id"].unique()[:max_products]

    for i, product_id in enumerate(products):
        product_df = df[df["product_id"] == product_id].copy()
        if len(product_df) < 10:
            continue
        try:
            result = fit_bayesian_elasticity_single(product_df, product_id)
            result["category"] = product_df["category"].iloc[0]
            result["true_elasticity"] = product_df["true_elasticity"].iloc[0]
            results.append(result)
            logger.info(f"[{i+1}/{len(products)}] {product_id}: "
                        f"elasticity={result['bayesian_elasticity']:.3f}, "
                        f"CI width={result['bayesian_ci_width']:.3f}")
        except Exception as e:
            logger.warning(f"Failed for {product_id}: {e}")

    return pd.DataFrame(results)


def compare_bayesian_vs_ols(bayesian_df: pd.DataFrame,
                              ols_df: pd.DataFrame) -> dict:
    """
    Compare Bayesian credible intervals vs OLS confidence intervals.
    Split by sparse (<50 obs) vs dense (>=50 obs) products.
    """
    merged = bayesian_df.merge(ols_df, on=["product_id", "category",
                                             "true_elasticity", "n_obs"],
                                how="inner")

    sparse = merged[merged["n_obs"] < 50]
    dense = merged[merged["n_obs"] >= 50]

    def compute_stats(subset, label):
        if len(subset) == 0:
            return {}
        bayesian_width = subset["bayesian_ci_width"].mean()
        ols_width = subset["ols_ci_width"].mean()
        improvement_pct = (ols_width - bayesian_width) / ols_width * 100

        bayesian_error = (subset["bayesian_elasticity"] - subset["true_elasticity"]).abs().mean()
        ols_error = (subset["ols_elasticity"] - subset["true_elasticity"]).abs().mean()

        return {
            f"{label}_n": len(subset),
            f"{label}_bayesian_ci_width": round(bayesian_width, 4),
            f"{label}_ols_ci_width": round(ols_width, 4),
            f"{label}_ci_improvement_pct": round(improvement_pct, 2),
            f"{label}_bayesian_mae": round(bayesian_error, 4),
            f"{label}_ols_mae": round(ols_error, 4),
        }

    results = {}
    results.update(compute_stats(sparse, "sparse"))
    results.update(compute_stats(dense, "dense"))
    results["total_products"] = len(merged)

    logger.info(f"\n=== Bayesian vs OLS Comparison ===")
    logger.info(f"Sparse products (<50 obs): n={results.get('sparse_n', 0)}")
    logger.info(f"  CI width improvement: {results.get('sparse_ci_improvement_pct', 0):.1f}%")
    logger.info(f"  Bayesian MAE: {results.get('sparse_bayesian_mae', 0):.4f}")
    logger.info(f"  OLS MAE: {results.get('sparse_ols_mae', 0):.4f}")
    logger.info(f"Dense products (>=50 obs): n={results.get('dense_n', 0)}")
    logger.info(f"  CI width improvement: {results.get('dense_ci_improvement_pct', 0):.1f}%")

    return results


def plot_elasticity_comparison(bayesian_df: pd.DataFrame,
                                ols_df: pd.DataFrame):
    """Plot Bayesian vs OLS elasticity estimates with uncertainty."""
    merged = bayesian_df.merge(ols_df, on=["product_id", "category",
                                             "true_elasticity", "n_obs"],
                                how="inner").head(20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Bayesian vs OLS Price Elasticity Estimates", fontsize=13)

    ax = axes[0]
    x = range(len(merged))
    ax.errorbar(x, merged["bayesian_elasticity"],
                yerr=[merged["bayesian_elasticity"] - merged["bayesian_ci_lower"],
                      merged["bayesian_ci_upper"] - merged["bayesian_elasticity"]],
                fmt="o", color="#2ecc71", label="Bayesian", alpha=0.7, capsize=3)
    ax.errorbar([xi + 0.3 for xi in x], merged["ols_elasticity"],
                yerr=[merged["ols_elasticity"] - merged["ols_ci_lower"],
                      merged["ols_ci_upper"] - merged["ols_elasticity"]],
                fmt="s", color="#e74c3c", label="OLS", alpha=0.7, capsize=3)
    ax.scatter(x, merged["true_elasticity"], marker="*", color="#3498db",
               s=100, label="True elasticity", zorder=5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Product")
    ax.set_ylabel("Elasticity Estimate")
    ax.set_title("Elasticity Estimates with Uncertainty")
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(merged["n_obs"], merged["bayesian_ci_width"],
                color="#2ecc71", label="Bayesian CI width", alpha=0.7)
    ax2.scatter(merged["n_obs"], merged["ols_ci_width"],
                color="#e74c3c", label="OLS CI width", alpha=0.7)
    ax2.axvline(50, color="gray", linestyle="--", label="Sparse/dense threshold")
    ax2.set_xlabel("Number of observations")
    ax2.set_ylabel("CI/Credible Interval Width")
    ax2.set_title("Uncertainty vs Sample Size")
    ax2.legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "elasticity_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Elasticity comparison plot saved to {path}")
    return str(path)


def run_elasticity_analysis(df: pd.DataFrame) -> dict:
    mlflow.set_experiment("demandsense_elasticity")
    with mlflow.start_run(run_name="bayesian_vs_ols"):
        logger.info("Fitting OLS elasticity models...")
        ols_df = fit_ols_elasticity(df)

        logger.info("Fitting Bayesian elasticity models...")
        bayesian_df = fit_bayesian_elasticity_all(df)

        comparison = compare_bayesian_vs_ols(bayesian_df, ols_df)

        for k, v in comparison.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        bayesian_df.to_csv(OUTPUT_DIR / "bayesian_elasticity.csv", index=False)
        ols_df.to_csv(OUTPUT_DIR / "ols_elasticity.csv", index=False)

        plot_elasticity_comparison(bayesian_df, ols_df)

        return {
            "bayesian_df": bayesian_df,
            "ols_df": ols_df,
            "comparison": comparison,
        }
