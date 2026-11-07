"""Beta-Binomial Bayesian updating for high-demand day frequency."""

from statistics import NormalDist


def beta_posterior(alpha_prior, beta_prior, high_days, total_days):
    """Return posterior parameters and summaries for a high-demand probability."""
    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Beta prior parameters must be positive.")
    if total_days < 0 or high_days < 0 or high_days > total_days:
        raise ValueError("Counts must satisfy 0 <= high_days <= total_days.")

    alpha_post = alpha_prior + high_days
    beta_post = beta_prior + total_days - high_days
    total = alpha_post + beta_post
    mean = alpha_post / total
    variance = (alpha_post * beta_post) / (total**2 * (total + 1))
    standard_deviation = variance**0.5

    # A normal approximation is used for a readable interval.
    # The project labels this as approximate rather than an exact beta quantile.
    z = NormalDist().inv_cdf(0.975)
    lower = max(0.0, mean - z * standard_deviation)
    upper = min(1.0, mean + z * standard_deviation)

    return {
        "alpha_prior": float(alpha_prior),
        "beta_prior": float(beta_prior),
        "high_days": int(high_days),
        "normal_days": int(total_days - high_days),
        "alpha_posterior": float(alpha_post),
        "beta_posterior": float(beta_post),
        "posterior_mean": float(mean),
        "approx_95_lower": float(lower),
        "approx_95_upper": float(upper),
    }
