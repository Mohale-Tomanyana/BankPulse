"""Run the complete BankPulse workflow."""

from pathlib import Path
import numpy as np
import pandas as pd

from bankpulse.config import CONFIG
from bankpulse.analysis import (
    descriptive_summary,
    mean_confidence_interval,
    bootstrap_mean_interval,
    least_squares_trend,
    reserve_cost,
    marginal_holding_cost,
)
from bankpulse.bayes import beta_posterior
from bankpulse.simulation import simulate_daily_demand
from bankpulse.reporting import create_charts, write_report
from generate_data import generate_dataset


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "branch_daily.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = generate_dataset()

    rng = np.random.default_rng(CONFIG.random_seed + 1)
    summary = descriptive_summary(data)

    print("\nBANKPULSE — BRANCH CASH DEMAND & LIQUIDITY INTELLIGENCE")
    print("=" * 62)
    print(f"Branch-days analysed: {len(data)}")
    print(f"Branches: {data['branch_id'].nunique()}")
    print(f"Total withdrawals: R{data['withdrawal_amount'].sum():,.2f}")
    print(f"Total deposits: R{data['deposit_amount'].sum():,.2f}")
    print(f"Net cash outflow: R{data['net_cash_outflow'].sum():,.2f}")

    print("\nDESCRIPTIVE STATISTICS BY BRANCH")
    print(summary.round(2).to_string())

    bayes_rows = []
    policy_rows = []
    trend_rows = []

    for branch_id, group in data.groupby("branch_id"):
        group = group.sort_values("date")
        withdrawals = group["withdrawal_amount"].to_numpy(dtype=float)

        mean, ci_low, ci_high = mean_confidence_interval(withdrawals)
        _, boot_low, boot_high = bootstrap_mean_interval(
            withdrawals, rng, CONFIG.bootstrap_samples
        )

        print(f"\n{branch_id} — INFERENCE")
        print(f"Mean daily withdrawals: R{mean:,.2f}")
        print(f"Approx. 95% confidence interval: R{ci_low:,.2f} to R{ci_high:,.2f}")
        print(f"Bootstrap 95% interval: R{boot_low:,.2f} to R{boot_high:,.2f}")

        # Beta(2, 8) prior: prior mean high-demand probability = 0.20.
        # The first half sets the threshold; the second half supplies
        # the observations used for the Bayesian update.
        observed_high_demand = group["high_demand"].dropna()
        high_days = int(observed_high_demand.sum())
        observed_days = int(observed_high_demand.count())
        posterior = beta_posterior(2, 8, high_days, observed_days)
        bayes_rows.append({"branch_id": branch_id, **posterior})

        # Least-squares trend on the first 80% of dates, purely descriptive.
        training_size = int(len(withdrawals) * 0.80)
        intercept, slope, fitted = least_squares_trend(
            withdrawals[:training_size]
        )
        trend_rows.append({
            "branch_id": branch_id,
            "daily_trend_slope_R": slope,
            "trend_intercept_R": intercept,
        })

        # Future demand scenarios are based on historical sample averages.
        average_count = max(float(group["withdrawal_count"].mean()), 0.1)
        average_amount = max(
            float(group["withdrawal_amount"].sum())
            / max(int(group["withdrawal_count"].sum()), 1),
            1.0,
        )

        normal_demand = simulate_daily_demand(
            average_count,
            average_amount,
            CONFIG.simulation_days,
            rng,
            stress_multiplier=1.0,
        )
        stressed_demand = simulate_daily_demand(
            average_count,
            average_amount,
            CONFIG.simulation_days,
            rng,
            stress_multiplier=1.25,
        )

        for percentile in CONFIG.reserve_percentiles:
            reserve = float(np.quantile(normal_demand, percentile))
            metrics_normal = reserve_cost(
                reserve,
                normal_demand,
                CONFIG.shortfall_cost_per_rand,
                CONFIG.holding_cost_per_rand,
            )
            metrics_stress = reserve_cost(
                reserve,
                stressed_demand,
                CONFIG.shortfall_cost_per_rand,
                CONFIG.holding_cost_per_rand,
            )

            policy_rows.append({
                "branch_id": branch_id,
                "reserve_percentile": percentile,
                "reserve_amount_R": reserve,
                "normal_shortfall_probability": metrics_normal["shortfall_probability"],
                "normal_average_cost_R": metrics_normal["average_total_cost"],
                "stress_shortfall_probability": metrics_stress["shortfall_probability"],
                "stress_average_cost_R": metrics_stress["average_total_cost"],
            })

    bayes_table = pd.DataFrame(bayes_rows)
    policy_table = pd.DataFrame(policy_rows)
    trend_table = pd.DataFrame(trend_rows)

    print("\nBAYESIAN HIGH-DEMAND ESTIMATES")
    print(
        bayes_table[
            ["branch_id", "high_days", "posterior_mean",
             "approx_95_lower", "approx_95_upper"]
        ].round(4).to_string(index=False)
    )

    print("\nRESERVE POLICY COMPARISON")
    print(policy_table.round(2).to_string(index=False))

    print("\nLEAST-SQUARES TREND")
    print(trend_table.round(2).to_string(index=False))

    print("\nCALCULUS / SENSITIVITY INTERPRETATION")
    marginal_cost = marginal_holding_cost(CONFIG.holding_cost_per_rand)
    print(
        "Under the simplified linear holding-cost assumption, "
        f"each additional R1 of reserve adds R{marginal_cost:.4f} "
        "of holding cost per simulated day."
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    create_charts(data, summary, OUTPUT_DIR)
    write_report(
        OUTPUT_DIR / "bankpulse_report.txt",
        summary,
        bayes_table,
        policy_table,
        trend_table,
    )

    print(f"\nCharts and report saved in: {OUTPUT_DIR}")
    print("BankPulse analysis completed.")


if __name__ == "__main__":
    main()
