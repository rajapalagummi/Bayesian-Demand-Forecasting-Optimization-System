"""
DemandSense — Mixed-Integer Programming Resource Allocation
Solves optimal inventory allocation across product categories
given demand forecasts and capacity constraints.

Compares three approaches:
1. MIP solution (PuLP — optimal)
2. Greedy heuristic (proportional to forecast)
3. Equal allocation baseline

Quantifies MIP improvement over heuristics in:
- Constraint satisfaction rate
- Objective value (minimize total unmet demand)
- Computational feasibility
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import pulp

logger = logging.getLogger(__name__)


@dataclass
class AllocationProblem:
    products: list
    forecast_demand: dict
    capacity_total: int
    min_allocation: dict
    max_allocation: dict
    category_budgets: dict
    product_categories: dict
    holding_costs: dict


def build_allocation_problem(forecast_df: pd.DataFrame,
                              total_capacity: int = 10000) -> AllocationProblem:
    """
    Build allocation problem from demand forecasts.
    Constraints:
    - Total allocation <= total_capacity
    - Per-product min/max bounds
    - Per-category budget limits
    """
    products = forecast_df["product_id"].tolist()
    forecast_demand = dict(zip(forecast_df["product_id"],
                                forecast_df["forecast_demand"]))
    product_categories = dict(zip(forecast_df["product_id"],
                                   forecast_df["category"]))

    total_demand = sum(forecast_demand.values())
    scale = total_capacity / max(total_demand, 1)

    min_allocation = {p: max(1, int(forecast_demand[p] * scale * 0.3))
                      for p in products}
    max_allocation = {p: int(forecast_demand[p] * scale * 2.0)
                      for p in products}

    categories = forecast_df["category"].unique()
    category_totals = forecast_df.groupby("category")["forecast_demand"].sum()
    category_budgets = {
        cat: int(total_capacity * (category_totals.get(cat, 0) / max(total_demand, 1)) * 1.2)
        for cat in categories
    }

    np.random.seed(42)
    holding_costs = {p: round(np.random.uniform(0.1, 0.5), 2) for p in products}

    return AllocationProblem(
        products=products,
        forecast_demand=forecast_demand,
        capacity_total=total_capacity,
        min_allocation=min_allocation,
        max_allocation=max_allocation,
        category_budgets=category_budgets,
        product_categories=product_categories,
        holding_costs=holding_costs,
    )


def solve_mip(problem: AllocationProblem) -> dict:
    """
    Mixed-Integer Programming formulation:
    Minimize: sum(holding_cost_i * x_i) + sum(unmet_demand_i)
    Subject to:
    - sum(x_i) <= total_capacity
    - min_i <= x_i <= max_i
    - sum(x_i for i in category_c) <= category_budget_c
    - x_i integer >= 0
    """
    model = pulp.LpProblem("DemandAllocation", pulp.LpMinimize)

    x = {p: pulp.LpVariable(f"x_{p}", lowBound=problem.min_allocation[p],
                              upBound=problem.max_allocation[p], cat="Integer")
         for p in problem.products}

    unmet = {p: pulp.LpVariable(f"unmet_{p}", lowBound=0, cat="Continuous")
             for p in problem.products}

    model += (
        pulp.lpSum(problem.holding_costs[p] * x[p] for p in problem.products) +
        pulp.lpSum(unmet[p] for p in problem.products)
    )

    model += pulp.lpSum(x[p] for p in problem.products) <= problem.capacity_total

    for cat, budget in problem.category_budgets.items():
        cat_products = [p for p in problem.products
                        if problem.product_categories[p] == cat]
        if cat_products:
            model += pulp.lpSum(x[p] for p in cat_products) <= budget

    for p in problem.products:
        model += unmet[p] >= problem.forecast_demand[p] - x[p]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    status = model.solve(solver)

    if status != 1:
        logger.warning(f"MIP solver status: {pulp.LpStatus[status]}")

    allocation = {p: int(x[p].value() or 0) for p in problem.products}
    total_unmet = sum(
        max(0, problem.forecast_demand[p] - allocation[p])
        for p in problem.products
    )
    total_cost = sum(
        problem.holding_costs[p] * allocation.get(p, 0)
        for p in problem.products
    )

    return {
        "method": "MIP",
        "status": pulp.LpStatus[status],
        "allocation": allocation,
        "total_allocated": sum(allocation.values()),
        "total_unmet_demand": round(total_unmet, 2),
        "total_holding_cost": round(total_cost, 2),
        "objective_value": round(total_cost + total_unmet, 2),
        "constraint_satisfaction_rate": round(
            sum(1 for p in problem.products
                if allocation.get(p, 0) >= problem.forecast_demand[p]) /
            len(problem.products) * 100, 2
        ),
    }


def solve_greedy(problem: AllocationProblem) -> dict:
    """
    Greedy heuristic: allocate proportionally to forecasted demand.
    Respects total capacity but ignores category budgets and holding costs.
    """
    total_demand = sum(problem.forecast_demand.values())
    allocation = {}
    for p in problem.products:
        proportion = problem.forecast_demand[p] / max(total_demand, 1)
        raw = int(problem.capacity_total * proportion)
        allocation[p] = max(problem.min_allocation[p],
                             min(problem.max_allocation[p], raw))

    actual_total = sum(allocation.values())
    if actual_total > problem.capacity_total:
        scale = problem.capacity_total / actual_total
        allocation = {p: int(v * scale) for p, v in allocation.items()}

    total_unmet = sum(
        max(0, problem.forecast_demand[p] - allocation[p])
        for p in problem.products
    )
    total_cost = sum(
        problem.holding_costs[p] * allocation.get(p, 0)
        for p in problem.products
    )

    return {
        "method": "Greedy",
        "allocation": allocation,
        "total_allocated": sum(allocation.values()),
        "total_unmet_demand": round(total_unmet, 2),
        "total_holding_cost": round(total_cost, 2),
        "objective_value": round(total_cost + total_unmet, 2),
        "constraint_satisfaction_rate": round(
            sum(1 for p in problem.products
                if allocation.get(p, 0) >= problem.forecast_demand[p]) /
            len(problem.products) * 100, 2
        ),
    }


def solve_equal(problem: AllocationProblem) -> dict:
    """Equal allocation baseline: divide total capacity equally."""
    per_product = problem.capacity_total // len(problem.products)
    allocation = {p: max(problem.min_allocation[p],
                          min(problem.max_allocation[p], per_product))
                  for p in problem.products}

    total_unmet = sum(
        max(0, problem.forecast_demand[p] - allocation[p])
        for p in problem.products
    )
    total_cost = sum(
        problem.holding_costs[p] * allocation.get(p, 0)
        for p in problem.products
    )

    return {
        "method": "Equal",
        "allocation": allocation,
        "total_allocated": sum(allocation.values()),
        "total_unmet_demand": round(total_unmet, 2),
        "total_holding_cost": round(total_cost, 2),
        "objective_value": round(total_cost + total_unmet, 2),
        "constraint_satisfaction_rate": round(
            sum(1 for p in problem.products
                if allocation.get(p, 0) >= problem.forecast_demand[p]) /
            len(problem.products) * 100, 2
        ),
    }


def compare_methods(mip: dict, greedy: dict, equal: dict) -> dict:
    """Quantify MIP improvement over heuristic baselines."""
    mip_vs_greedy_obj = round(
        (greedy["objective_value"] - mip["objective_value"]) /
        max(greedy["objective_value"], 1) * 100, 2
    )
    mip_vs_equal_obj = round(
        (equal["objective_value"] - mip["objective_value"]) /
        max(equal["objective_value"], 1) * 100, 2
    )
    mip_vs_greedy_satisfaction = round(
        mip["constraint_satisfaction_rate"] - greedy["constraint_satisfaction_rate"], 2
    )
    mip_vs_equal_satisfaction = round(
        mip["constraint_satisfaction_rate"] - equal["constraint_satisfaction_rate"], 2
    )

    comparison = {
        "mip_objective": mip["objective_value"],
        "greedy_objective": greedy["objective_value"],
        "equal_objective": equal["objective_value"],
        "mip_vs_greedy_improvement_pct": mip_vs_greedy_obj,
        "mip_vs_equal_improvement_pct": mip_vs_equal_obj,
        "mip_satisfaction_rate": mip["constraint_satisfaction_rate"],
        "greedy_satisfaction_rate": greedy["constraint_satisfaction_rate"],
        "equal_satisfaction_rate": equal["constraint_satisfaction_rate"],
        "mip_vs_greedy_satisfaction_delta": mip_vs_greedy_satisfaction,
        "mip_vs_equal_satisfaction_delta": mip_vs_equal_satisfaction,
        "mip_total_unmet": mip["total_unmet_demand"],
        "greedy_total_unmet": greedy["total_unmet_demand"],
        "equal_total_unmet": equal["total_unmet_demand"],
    }

    logger.info(f"\n=== MIP vs Heuristic Comparison ===")
    logger.info(f"MIP objective: {mip['objective_value']:.2f}")
    logger.info(f"Greedy objective: {greedy['objective_value']:.2f}")
    logger.info(f"Equal objective: {equal['objective_value']:.2f}")
    logger.info(f"MIP vs Greedy improvement: {mip_vs_greedy_obj:.1f}%")
    logger.info(f"MIP vs Equal improvement: {mip_vs_equal_obj:.1f}%")
    logger.info(f"MIP constraint satisfaction: {mip['constraint_satisfaction_rate']:.1f}%")
    logger.info(f"Greedy constraint satisfaction: {greedy['constraint_satisfaction_rate']:.1f}%")

    return comparison


def run_optimization(df: pd.DataFrame,
                      forecast_df: Optional[pd.DataFrame] = None) -> dict:
    """Run full MIP optimization pipeline."""
    if forecast_df is None:
        forecast_df = df.groupby(["product_id", "category"]).agg(
            forecast_demand=("units_sold", "mean")
        ).reset_index()
        forecast_df["forecast_demand"] = (
            forecast_df["forecast_demand"] * 13
        ).round().astype(int)

    total_demand = int(forecast_df["forecast_demand"].sum())
    capacity = int(total_demand * 0.80)
    problem = build_allocation_problem(forecast_df, total_capacity=capacity)
    logger.info(f"Solving MIP for {len(problem.products)} products, "
                f"capacity={problem.capacity_total}")

    mip_result = solve_mip(problem)
    greedy_result = solve_greedy(problem)
    equal_result = solve_equal(problem)
    comparison = compare_methods(mip_result, greedy_result, equal_result)

    return {
        "mip": mip_result,
        "greedy": greedy_result,
        "equal": equal_result,
        "comparison": comparison,
        "problem": problem,
    }
