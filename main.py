"""
DemandSense — Main Pipeline Orchestrator
Runs full pipeline: Ingest → Bayesian Elasticity → Forecast A/B Test → MIP Optimization

Usage:
    python main.py --mode demo      # Quick demo on 10 products
    python main.py --mode full      # Full pipeline on all products
    python main.py --mode results   # Print saved results
"""
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_demo():
    """Quick demo on 10 products — runs in ~5 minutes."""
    from data.ingest import generate_synthetic_retail_data, compute_lag_features
    from models.bayesian_elasticity import fit_ols_elasticity, fit_bayesian_elasticity_all, compare_bayesian_vs_ols
    from forecasting.demand_forecast import run_ab_forecast_test
    from optimization.mip_allocator import run_optimization

    logger.info("=== DemandSense Demo (10 products) ===")

    df = generate_synthetic_retail_data(n_products=10, n_weeks=104)
    df = compute_lag_features(df)
    logger.info(f"Generated {len(df)} rows for {df['product_id'].nunique()} products")

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("demandsense_demo")

    with mlflow.start_run(run_name="demo_run"):
        logger.info("\n--- Stage 1: OLS Elasticity ---")
        ols_df = fit_ols_elasticity(df)
        logger.info(f"OLS elasticity fitted for {len(ols_df)} products")
        logger.info(f"Mean OLS elasticity: {ols_df['ols_elasticity'].mean():.4f}")

        logger.info("\n--- Stage 2: Bayesian Elasticity ---")
        bayesian_df = fit_bayesian_elasticity_all(df, max_products=10)
        logger.info(f"Bayesian elasticity fitted for {len(bayesian_df)} products")

        comparison = compare_bayesian_vs_ols(bayesian_df, ols_df)

        logger.info("\n--- Stage 3: Forecast A/B Test ---")
        forecast_results = run_ab_forecast_test(df, bayesian_df)

        logger.info("\n--- Stage 4: MIP Optimization ---")
        opt_results = run_optimization(df)

        for k, v in comparison.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                mlflow.log_metric(f"elasticity_{k}", v)

        if len(forecast_results) > 0:
            mlflow.log_metric("sarima_avg_mape", forecast_results["sarima_mape"].mean())
            mlflow.log_metric("elasticity_avg_mape", forecast_results["elasticity_mape"].mean())
            mlflow.log_metric("mape_improvement_pct", forecast_results["mape_improvement_pct"].mean())

        mlflow.log_metric("mip_vs_greedy_improvement_pct",
                           opt_results["comparison"]["mip_vs_greedy_improvement_pct"])
        mlflow.log_metric("mip_satisfaction_rate",
                           opt_results["comparison"]["mip_satisfaction_rate"])

        from analysis.statistical import run_statistical_analysis
        stat_results = run_statistical_analysis(
            df=df,
            ols_df=ols_df,
            bayesian_df=bayesian_df,
            forecast_results=forecast_results,
            opt_results=opt_results,
        )

        from analysis.factor_decay import run_factor_decay_analysis
        decay_results = run_factor_decay_analysis(
            df=df,
            ols_df=ols_df,
            bayesian_df=bayesian_df,
        )

        from analysis.advanced_eda import run_advanced_eda
        eda_results = run_advanced_eda(
            df=df,
            ols_df=ols_df,
            bayesian_df=bayesian_df,
            opt_results=opt_results,
        )

        from analysis.hypothesis_testing import run_hypothesis_testing
        hypothesis_results = run_hypothesis_testing(
            df=df,
            ols_df=ols_df,
            bayesian_df=bayesian_df,
            forecast_results=forecast_results,
            opt_results=opt_results,
        )



        results = {
            "mode": "demo",
            "n_products": df["product_id"].nunique(),
            "bayesian_vs_ols": {k: v for k, v in comparison.items()
                                  if isinstance(v, (int, float))},
            "forecast_ab_test": {
                "sarima_avg_mape": round(float(forecast_results["sarima_mape"].mean()), 4)
                if len(forecast_results) > 0 else None,
                "elasticity_avg_mape": round(float(forecast_results["elasticity_mape"].mean()), 4)
                if len(forecast_results) > 0 else None,
                "mape_improvement_pct": round(float(forecast_results["mape_improvement_pct"].mean()), 2)
                if len(forecast_results) > 0 else None,
                "n_products_tested": len(forecast_results),
                "treatment_wins": int((forecast_results["elasticity_mape"] < forecast_results["sarima_mape"]).sum())
                if len(forecast_results) > 0 else 0,
            },
            "optimization": {
                "mip_objective": opt_results["comparison"]["mip_objective"],
                "greedy_objective": opt_results["comparison"]["greedy_objective"],
                "mip_vs_greedy_improvement_pct": opt_results["comparison"]["mip_vs_greedy_improvement_pct"],
                "mip_vs_equal_improvement_pct": opt_results["comparison"]["mip_vs_equal_improvement_pct"],
                "mip_satisfaction_rate": opt_results["comparison"]["mip_satisfaction_rate"],
                "greedy_satisfaction_rate": opt_results["comparison"]["greedy_satisfaction_rate"],
            }
        }

        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n=== DEMO RESULTS ===")
        logger.info(f"Bayesian CI improvement over OLS (sparse products): "
                    f"{comparison.get('sparse_ci_improvement_pct', 'N/A')}%")
        logger.info(f"Forecast MAPE improvement: "
                    f"{results['forecast_ab_test']['mape_improvement_pct']}%")
        logger.info(f"MIP vs Greedy objective improvement: "
                    f"{results['optimization']['mip_vs_greedy_improvement_pct']}%")
        logger.info(f"MIP constraint satisfaction: "
                    f"{results['optimization']['mip_satisfaction_rate']}%")

        return results


def main():
    parser = argparse.ArgumentParser(description="DemandSense Pipeline")
    parser.add_argument("--mode", choices=["demo", "full", "results"],
                        default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        results = run_demo()
        print("\n=== Final Numbers ===")
        print(json.dumps({
            k: v for k, v in results.items()
            if k != "mode"
        }, indent=2, default=str))

    elif args.mode == "results":
        results_path = OUTPUT_DIR / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No results found. Run --mode demo first.")


if __name__ == "__main__":
    main()
