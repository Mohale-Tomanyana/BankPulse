"""Statistical summaries, inference and an interpretable trend model."""

import numpy as np
import pandas as pd


def descriptive_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise daily withdrawal amounts by branch."""
    return (
        data.groupby("branch_id")["withdrawal_amount"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={"count": "days"})
    )


def mean_confidence_interval(values, confidence=0.95):
    """Normal-approximation confidence interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two observations are required.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    # 1.96 is the standard normal critical value for an approximate 95% interval.
    # For other confidence levels this project uses a normal quantile.
    from statistics import NormalDist
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2)
    return mean, mean - z_value * standard_error, mean + z_value * standard_error


def bootstrap_mean_interval(values, rng, samples=1000, confidence=0.95):
    """Percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two observations are required.")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    simulated_means = np.empty(samples)
    for index in range(samples):
        sample = rng.choice(values, size=values.size, replace=True)
        simulated_means[index] = np.mean(sample)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(simulated_means, [tail, 1 - tail])
    return float(np.mean(values)), float(lower), float(upper)


def least_squares_trend(values):
    """
    Fit y = intercept + slope*x using least squares.
    Returns intercept, slope and fitted values.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two observations are required.")
    x = np.arange(values.size, dtype=float)
    design = np.column_stack((np.ones(values.size), x))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    return float(coefficients[0]), float(coefficients[1]), fitted


def reserve_cost(reserve, demand, shortfall_rate, holding_rate):
    """Calculate simplified shortage and excess-cash cost."""
    demand = np.asarray(demand, dtype=float)
    if reserve < 0 or np.any(demand < 0):
        raise ValueError("Reserve and demand must be non-negative.")
    shortfall = np.maximum(demand - reserve, 0)
    excess = np.maximum(reserve - demand, 0)
    shortage_cost = shortfall_rate * shortfall
    holding_cost = holding_rate * excess
    return {
        "shortfall_probability": float(np.mean(shortfall > 0)),
        "average_shortfall": float(np.mean(shortfall)),
        "average_excess_cash": float(np.mean(excess)),
        "average_shortfall_cost": float(np.mean(shortage_cost)),
        "average_holding_cost": float(np.mean(holding_cost)),
        "average_total_cost": float(np.mean(shortage_cost + holding_cost)),
    }


def marginal_holding_cost(holding_rate):
    """
    Derivative of holding cost H(R) = c*R with respect to reserve R.
    This is a simple calculus interpretation: each extra rand of reserve
    adds c rand of holding cost under this simplified linear assumption.
    """
    if holding_rate < 0:
        raise ValueError("Holding-cost rate cannot be negative.")
    return float(holding_rate)
