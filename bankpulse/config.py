"""Central configuration for the BankPulse project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int = 20261107
    number_of_days: int = 240
    number_of_branches: int = 5
    simulation_days: int = 30
    reserve_percentiles: tuple = (0.80, 0.90, 0.95)
    shortfall_cost_per_rand: float = 0.08
    holding_cost_per_rand: float = 0.002
    bootstrap_samples: int = 1000


CONFIG = ProjectConfig()
