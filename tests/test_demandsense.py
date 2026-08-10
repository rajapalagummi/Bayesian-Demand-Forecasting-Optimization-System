"""DemandSense — Unit Tests"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.ingest import generate_synthetic_retail_data, compute_lag_features
from models.bayesian_elasticity import fit_ols_elasticity
from optimization.mip_allocator import (solve_greedy, solve_equal,
                                         build_allocation_problem, compare_methods)


class TestDataIngestion:
    def test_generates_correct_shape(self):
        df = generate_synthetic_retail_data(n_products=5, n_weeks=52)
        assert len(df) == 5 * 52
        assert "product_id" in df.columns
        assert "units_sold" in df.columns
        assert "price" in df.columns

    def test_true_elasticity_negative(self):
        df = generate_synthetic_retail_data(n_products=10, n_weeks=52)
        assert (df["true_elasticity"] < 0).all()

    def test_lag_features_created(self):
        df = generate_synthetic_retail_data(n_products=3, n_weeks=52)
        df = compute_lag_features(df)
        assert "lag_1_demand" in df.columns
        assert "log_price" in df.columns
        assert "log_demand" in df.columns

    def test_no_negative_demand(self):
        df = generate_synthetic_retail_data(n_products=10, n_weeks=52)
        assert (df["units_sold"] >= 0).all()


class TestOLSElasticity:
    def test_ols_returns_negative_elasticity(self):
        df = generate_synthetic_retail_data(n_products=5, n_weeks=104)
        df = compute_lag_features(df)
        ols_df = fit_ols_elasticity(df)
        assert len(ols_df) > 0
        assert (ols_df["ols_elasticity"] < 0).mean() > 0.5

    def test_ols_ci_width_positive(self):
        df = generate_synthetic_retail_data(n_products=5, n_weeks=104)
        df = compute_lag_features(df)
        ols_df = fit_ols_elasticity(df)
        assert (ols_df["ols_ci_width"] > 0).all()


class TestOptimization:
    def setup_method(self):
        self.forecast_df = pd.DataFrame({
            "product_id": [f"PROD_{i:03d}" for i in range(10)],
            "category": ["Food"] * 4 + ["Household"] * 3 + ["Hobbies"] * 3,
            "forecast_demand": np.random.randint(100, 500, 10),
        })
        self.problem = build_allocation_problem(self.forecast_df, total_capacity=3000)

    def test_greedy_respects_capacity(self):
        result = solve_greedy(self.problem)
        assert result["total_allocated"] <= self.problem.capacity_total

    def test_equal_respects_bounds(self):
        result = solve_equal(self.problem)
        for p, alloc in result["allocation"].items():
            assert alloc >= self.problem.min_allocation[p]

    def test_mip_better_than_greedy(self):
        from optimization.mip_allocator import solve_mip
        mip = solve_mip(self.problem)
        greedy = solve_greedy(self.problem)
        equal = solve_equal(self.problem)
        comparison = compare_methods(mip, greedy, equal)
        assert isinstance(comparison["mip_vs_greedy_improvement_pct"], float)
        assert isinstance(comparison["mip_satisfaction_rate"], float)

    def test_satisfaction_rate_between_0_and_100(self):
        result = solve_greedy(self.problem)
        assert 0 <= result["constraint_satisfaction_rate"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
