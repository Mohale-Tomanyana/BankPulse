"""Small domain classes used by the analysis and simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Branch:
    branch_id: str
    region: str
    average_daily_customers: int
    base_withdrawal_rate: float


@dataclass(frozen=True)
class ReservePolicy:
    branch_id: str
    percentile: float
    reserve_amount: float

    def __post_init__(self):
        if not 0 < self.percentile < 1:
            raise ValueError("Reserve percentile must be between 0 and 1.")
        if self.reserve_amount < 0:
            raise ValueError("Reserve amount cannot be negative.")
