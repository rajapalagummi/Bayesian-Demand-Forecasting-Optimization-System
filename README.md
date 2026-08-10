# DemandSense — Bayesian Demand Elasticity & Optimization Engine

> Bayesian price elasticity modeling, elasticity-informed demand forecasting, and Mixed-Integer Programming resource allocation — quantifying the value of probabilistic modeling over frequentist baselines and optimal allocation over heuristics on real retail data patterns.

---

## Abstract

DemandSense addresses a core problem in retail analytics: how to estimate price sensitivity under data sparsity, use that estimate to improve demand forecasts, and translate improved forecasts into optimal resource allocation decisions. Three methodological questions are answered empirically:

1. **Do Bayesian credible intervals outperform OLS confidence intervals for sparse products?** Bayesian log-log elasticity model (PyMC) vs OLS baseline, compared on products with <50 vs ≥50 weekly observations.

2. **Does incorporating elasticity priors improve demand forecasts?** Elasticity-informed SARIMAX vs baseline SARIMA, A/B tested on 13-week holdout per product, measured by MAPE and CRPS.

3. **Does Mixed-Integer Programming allocation outperform heuristic baselines?** MIP (PuLP) vs greedy proportional allocation vs equal allocation, measured by objective value and constraint satisfaction rate.

---

## System Formulation

**Bayesian Price Elasticity Model:**

$$\log(Q_t) = \alpha + \beta \cdot \log(P_t) + \gamma \cdot X_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Priors:**
$$\beta \sim \mathcal{N}(-1, 1), \quad \alpha \sim \mathcal{N}(\bar{q}, 2), \quad \sigma \sim \text{HalfNormal}(0.5)$$

Posterior estimated via MCMC (2 chains, 1000 draws, 500 tuning steps). Credible intervals from 95% HDI of posterior $\beta$ samples.

**MIP Formulation:**

$$\min_{x} \sum_{i} c_i x_i + \sum_{i} u_i$$

$$\text{s.t.} \quad \sum_{i} x_i \leq C, \quad x_i^{min} \leq x_i \leq x_i^{max}, \quad \sum_{i \in \mathcal{C}_k} x_i \leq B_k, \quad u_i \geq d_i - x_i, \quad x_i \in \mathbb{Z}^+$$

where $c_i$ = holding cost, $u_i$ = unmet demand, $C$ = total capacity, $B_k$ = category budget, $d_i$ = forecasted demand.

**CRPS (Continuous Ranked Probability Score):**

$$\text{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma \left[ z(2\Phi(z)-1) + 2\phi(z) - \pi^{-1/2} \right], \quad z = \frac{y - \mu}{\sigma}$$

---

## Architecture

```
Synthetic Retail Data (50 products, 156 weeks)
              │
              ▼
┌─────────────────────────────────────────────┐
│         Data Engineering Layer               │
│  Lag features, rolling means, log transforms │
│  PostgreSQL + dbt transformation models      │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Bayesian │ │  OLS     │ │ Demand   │
│Elasticity│ │Baseline  │ │Forecasts │
│  (PyMC)  │ │          │ │(SARIMAX) │
└─────┬────┘ └─────┬────┘ └─────┬────┘
      │             │             │
      └──────┬──────┘             │
             ▼                    ▼
    ┌─────────────────┐  ┌──────────────────┐
    │ Bayesian vs OLS │  │  A/B Test:        │
    │ CI Comparison   │  │  Baseline vs      │
    │ (sparse/dense)  │  │  Elasticity-      │
    └────────┬────────┘  │  Informed SARIMAX │
             │           └────────┬──────────┘
             └──────────┬─────────┘
                        ▼
           ┌────────────────────────┐
           │   MIP Optimization     │
           │   (PuLP solver)        │
           │   vs Greedy Heuristic  │
           │   vs Equal Allocation  │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │  MLflow Experiment     │
           │  Tracking + Results    │
           │  JSON + Tableau Public │
           └────────────────────────┘
```

---

## Key Results (populated after run)

| Stage | Metric | Bayesian/MIP | Baseline | Improvement |
|---|---|---|---|---|
| Elasticity | CI Width (sparse) | TBD | TBD | TBD% |
| Elasticity | MAE vs true | TBD | TBD | TBD |
| Forecast | MAPE | TBD | TBD | TBD% |
| Optimization | Objective Value | TBD | TBD | TBD% |
| Optimization | Constraint Satisfaction | TBD% | TBD% | TBD pts |

---

## Stack

| Component | Technology |
|---|---|
| Bayesian Modeling | PyMC, ArviZ |
| MIP Optimization | PuLP (CBC solver) |
| Forecasting | statsmodels (SARIMAX), Prophet |
| Data Engineering | Pandas, dbt, PostgreSQL |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Seaborn, Tableau Public |
| Infrastructure | Docker, Git |

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run demo (10 products, ~5-10 minutes)
python main.py --mode demo

# View saved results
python main.py --mode results

# Run tests
pytest tests/ -v
```

---

## References

Rossi, P.E., Allenby, G.M., & McCulloch, R. (2005). *Bayesian Statistics and Marketing*. Wiley.

Wolsey, L.A. (1998). *Integer Programming*. Wiley-Interscience.

Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
