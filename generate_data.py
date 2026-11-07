"""Generate reproducible synthetic daily branch cash-flow data."""

from pathlib import Path
import numpy as np
import pandas as pd

from bankpulse.config import CONFIG
from bankpulse.models import Branch


def generate_dataset():
    rng = np.random.default_rng(CONFIG.random_seed)
    branches = [
        Branch("BR001", "Central", 520, 0.34),
        Branch("BR002", "Township", 430, 0.31),
        Branch("BR003", "Suburban", 390, 0.28),
        Branch("BR004", "Commercial", 610, 0.36),
        Branch("BR005", "Rural", 250, 0.25),
    ]

    dates = pd.date_range("2025-01-01", periods=CONFIG.number_of_days, freq="D")
    records = []

    for branch in branches:
        for date in dates:
            weekday = date.dayofweek
            month_end = date.day >= 25
            weekend = weekday >= 5

            weekday_factor = 0.62 if weekend else (1.18 if weekday in (0, 4) else 1.0)
            month_factor = 1.35 if month_end else 1.0

            expected_count = (
                branch.average_daily_customers
                * branch.base_withdrawal_rate
                * weekday_factor
                * month_factor
                / 10
            )
            withdrawal_count = int(rng.poisson(expected_count))

            # Average transaction size differs by branch type.
            branch_size_factor = {
                "Central": 1.15,
                "Township": 0.72,
                "Suburban": 1.05,
                "Commercial": 1.35,
                "Rural": 0.68,
            }[branch.region]

            mean_withdrawal = 1250 * branch_size_factor
            sigma = 0.60
            mu = np.log(mean_withdrawal) - sigma**2 / 2
            withdrawal_amount = (
                rng.lognormal(mu, sigma, withdrawal_count).sum()
                if withdrawal_count else 0.0
            )

            deposit_count = int(rng.poisson(expected_count * 0.78))
            mean_deposit = 1550 * branch_size_factor
            deposit_amount = (
                rng.lognormal(
                    np.log(mean_deposit) - sigma**2 / 2,
                    sigma,
                    deposit_count,
                ).sum()
                if deposit_count else 0.0
            )

            records.append({
                "branch_id": branch.branch_id,
                "region": branch.region,
                "date": date.date().isoformat(),
                "day_of_week": date.day_name(),
                "is_month_end_period": int(month_end),
                "withdrawal_count": withdrawal_count,
                "withdrawal_amount": round(float(withdrawal_amount), 2),
                "deposit_count": deposit_count,
                "deposit_amount": round(float(deposit_amount), 2),
                "net_cash_outflow": round(float(withdrawal_amount - deposit_amount), 2),
            })

    data = pd.DataFrame(records)
    # Use the first half as a calibration period and the second half
    # as an observation period. This avoids defining "high demand"
    # using the same observations whose frequency we later estimate.
    data["high_demand"] = np.nan

    for branch_id, group in data.groupby("branch_id"):
        group = group.sort_values("date")
        split_position = len(group) // 2

        calibration = group.iloc[:split_position]
        observation = group.iloc[split_position:]

        high_demand_threshold = calibration[
            "withdrawal_amount"
        ].quantile(0.80)

        data.loc[observation.index, "high_demand"] = (
            observation["withdrawal_amount"]
            > high_demand_threshold
        ).astype(int)

    output = Path(__file__).parent / "data" / "branch_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Created {len(data)} branch-day observations: {output}")
    return data


if __name__ == "__main__":
    generate_dataset()
