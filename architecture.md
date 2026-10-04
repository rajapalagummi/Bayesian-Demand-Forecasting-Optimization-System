# DemandSense — System Architecture

## Pipeline Overview

```mermaid
flowchart TD
    A([Synthetic Retail Data\n10 Products · 104 Weeks]) --> B[Data Ingestion\ndata/ingest.py]
    B --> C[Lag Feature Engineering\nlag_1, lag_4, rolling_4w, price_change_pct]

    C --> D[OLS Elasticity\nmodels/bayesian_elasticity.py]
    C --> E[Bayesian Elasticity\nPyMC Hierarchical Model]

    D & E --> F[Elasticity Comparison\nCI Width · MAE · Paired t-test]

    C --> G[Demand Forecast A/B Test\nforecasting/demand_forecast.py]
    G --> G1[SARIMA Baseline\nPure Time Series]
    G --> G2[Elasticity-Informed SARIMAX\nPrice Signal + Seasonality]
    G1 & G2 --> G3[A/B Test\nMAPE 0.2671 → 0.2118\n19.09% improvement · 10/10 wins]

    C --> H[MIP Optimization\noptimization/mip_allocator.py]
    H --> H1[MIP Solver\nPuLP Integer Programming]
    H --> H2[Greedy Baseline]
    H --> H3[Equal Split Baseline]
    H1 & H2 & H3 --> H4[Comparison\n60% vs 0% satisfaction\n6.48% objective improvement]

    F & G3 & H4 --> I[MLflow Experiment Tracking\nsqlite:///mlruns.db]
    F & G3 & H4 --> J[Results JSON\ndata/outputs/results.json]

    J --> K[Analysis Modules]
    K --> K1[Statistical Analysis\nstatistical.py]
    K --> K2[Elasticity Decay\nfactor_decay.py]
    K --> K3[Advanced EDA\nadvanced_eda.py]
    K --> K4[Hypothesis Testing\nhypothesis_testing.py]

    K1 & K2 & K3 & K4 --> L[(Images · HTML · JSON\ndata/outputs/analysis/)]

    style A fill:#4A90D9,color:#fff
    style B fill:#2ECC71,color:#fff
    style C fill:#2ECC71,color:#fff
    style E fill:#9B59B6,color:#fff
    style G3 fill:#E74C3C,color:#fff
    style H4 fill:#E74C3C,color:#fff
    style I fill:#F39C12,color:#fff
    style K fill:#1ABC9C,color:#fff
```

## Bayesian vs OLS Elasticity

```mermaid
flowchart LR
    subgraph OLS
        O1[Price · Demand Data] --> O2[Log-Log Regression\nlog demand = α + β log price]
        O2 --> O3[Point Estimate\nElasticity = β]
        O3 --> O4[Confidence Interval\nFrequentist · symmetric]
    end

    subgraph Bayesian
        B1[Price · Demand Data] --> B2[PyMC Hierarchical Model\nα ~ Normal · β ~ Normal · σ ~ HalfNormal]
        B2 --> B3[NUTS Sampler\n2 chains · 1000 draws]
        B3 --> B4[Posterior Distribution\nElasticity = E[β|data]]
        B4 --> B5[Credible Interval\n95% HDI · 14.61% narrower than OLS]
    end

    O4 & B5 --> C[Comparison\nPearson r · Paired t-test · Cohen's d]

    style B2 fill:#9B59B6,color:#fff
    style B5 fill:#2ECC71,color:#fff
    style O4 fill:#E74C3C,color:#fff
```

## Forecast A/B Test

```mermaid
flowchart TD
    DATA[Product Time Series\n104 weeks · price + demand] --> SPLIT[Train/Test Split\n80% train · 20% test]

    SPLIT --> CTRL[Control: SARIMA\nAutomatic order selection\nPure time series]
    SPLIT --> TREAT[Treatment: SARIMAX\nElasticity as exogenous regressor\nPrice signal + seasonality]

    CTRL --> M1[SARIMA MAPE\n0.2671 avg]
    TREAT --> M2[Elasticity MAPE\n0.2118 avg]

    M1 & M2 --> TEST[Statistical Testing\nPaired t-test · Mann-Whitney\nBootstrap CI on improvement]
    TEST --> RESULT[19.09% MAPE improvement\n10/10 products treatment wins\np < 0.05]

    style TREAT fill:#2ECC71,color:#fff
    style CTRL fill:#E74C3C,color:#fff
    style RESULT fill:#F39C12,color:#fff
```

## MIP Optimization

```mermaid
flowchart TD
    IN[10 Products\nDemand Forecasts\nCapacity Constraint] --> MIP

    subgraph MIP [MIP Solver — PuLP]
        OBJ[Minimize: Σ unmet_demand_i]
        CONS1[Subject to: Σ x_i ≤ total_capacity]
        CONS2[min_allocation_i ≤ x_i ≤ max_allocation_i]
        CONS3[x_i ∈ Integer]
    end

    subgraph GREEDY [Greedy Baseline]
        G1[Sort by demand descending]
        G2[Allocate until capacity exhausted]
    end

    subgraph EQUAL [Equal Split]
        E1[capacity / n_products per product]
    end

    MIP --> R1[Objective: 17926.17\nSatisfaction: 60%]
    GREEDY --> R2[Objective: 19168.03\nSatisfaction: 0%]
    EQUAL --> R3[Objective: ~22400\nSatisfaction: ~20%]

    R1 & R2 & R3 --> COMP[MIP improvement:\n6.48% vs Greedy\n24.99% vs Equal Split]

    style MIP fill:#2ECC71,color:#fff
    style R1 fill:#2ECC71,color:#fff
    style R2 fill:#E74C3C,color:#fff
    style COMP fill:#F39C12,color:#fff
```

## Analysis Pipeline

```mermaid
flowchart TD
    IN[Pipeline Outputs\nols_df · bayesian_df · forecast_results · opt_results] --> S[Statistical Analysis]
    IN --> D[Elasticity Decay]
    IN --> E[Advanced EDA]
    IN --> H[Hypothesis Testing]

    S --> S1[Distribution Analysis\nSkewness · Kurtosis · Normality]
    S --> S2[Outlier Detection\nIQR · Z-Score · Mahalanobis]
    S --> S3[Bootstrap CIs\n95% confidence on key metrics]
    S --> S4[Price-Demand Correlation\nPearson · Spearman]

    D --> D1[Autocorrelation\nLag 1-10 weeks]
    D --> D2[Half-Life Estimation\nExponential decay curve fit]
    D --> D3[IC Decay\n1 · 2 · 4 · 8 week horizons]
    D --> D4[Product Stability\nFirst vs second half comparison]
    D --> D5[Seasonal Patterns\nQuarterly elasticity analysis]

    E --> E1[Cross-Product Comparison\nOLS vs Bayesian ranked]
    E --> E2[Demand Heatmap\nProducts × weeks normalized]
    E --> E3[Price Sensitivity Clusters\nK-means: Low/Medium/High]
    E --> E4[Demand by Tier\nViolin plots across elasticity tiers]
    E --> E5[Outlier Detection\nZ-score + IQR flagging]

    H --> H1[t-tests\nOne-sample + paired]
    H --> H2[Mann-Whitney\nNon-parametric robustness]
    H --> H3[Multiple Comparison\nBonferroni + FDR correction]
    H --> H4[Power Analysis\nFor all significant tests]
    H --> H5[Significance Grades\nA/B/C/D scoring]

    S1 & S2 & S3 & S4 & D1 & D2 & D3 & D4 & D5 & E1 & E2 & E3 & E4 & E5 & H1 & H2 & H3 & H4 & H5 --> OUT[(17 Plots\nJSON Results\nHTML Dashboards)]

    style IN fill:#3498DB,color:#fff
    style OUT fill:#2ECC71,color:#fff
```
