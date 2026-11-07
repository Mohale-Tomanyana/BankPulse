"""Monte Carlo simulation of future daily withdrawal demand."""

import numpy as np


def simulate_daily_demand(
    average_withdrawal_count,
    average_withdrawal_size,
    number_of_days,
    rng,
    stress_multiplier=1.0,
):
    """
    Simulate total daily withdrawals.

    Counts follow a Poisson distribution. Individual withdrawal sizes
    follow a lognormal distribution with a chosen mean and moderate spread.
    """
    if average_withdrawal_count <= 0:
        raise ValueError("Average withdrawal count must be positive.")
    if average_withdrawal_size <= 0:
        raise ValueError("Average withdrawal size must be positive.")
    if number_of_days < 1:
        raise ValueError("Number of simulation days must be positive.")
    if stress_multiplier <= 0:
        raise ValueError("Stress multiplier must be positive.")

    counts = rng.poisson(
        average_withdrawal_count * stress_multiplier,
        size=number_of_days,
    )

    # Convert arithmetic mean and a 0.55 log-scale spread into lognormal mu.
    sigma = 0.55
    mu = np.log(average_withdrawal_size) - (sigma**2 / 2)

    totals = np.zeros(number_of_days, dtype=float)
    for day, count in enumerate(counts):
        if count:
            amounts = rng.lognormal(mu, sigma, size=count)
            totals[day] = amounts.sum()

    return totals
