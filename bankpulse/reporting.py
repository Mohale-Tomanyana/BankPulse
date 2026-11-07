"""Charts and human-readable output files."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_charts(data, branch_summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for branch_id, group in data.groupby("branch_id"):
        daily = group.sort_values("date")
        plt.plot(daily["date"], daily["withdrawal_amount"], alpha=0.65, label=branch_id)
    plt.title("Daily cash withdrawals by branch")
    plt.xlabel("Date")
    plt.ylabel("Withdrawal amount (R)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "daily_withdrawals.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(data["withdrawal_amount"], bins=35, edgecolor="black")
    plt.title("Distribution of daily branch withdrawals")
    plt.xlabel("Withdrawal amount (R)")
    plt.ylabel("Number of branch-days")
    plt.tight_layout()
    plt.savefig(output_dir / "withdrawal_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(branch_summary.index, branch_summary["mean"])
    plt.title("Average daily withdrawals by branch")
    plt.xlabel("Branch")
    plt.ylabel("Average withdrawals (R)")
    plt.tight_layout()
    plt.savefig(output_dir / "branch_comparison.png", dpi=150)
    plt.close()


def write_report(path, summary, bayes_table, policy_table, trend_table):
    lines = [
        "BANKPULSE — BRANCH CASH DEMAND & LIQUIDITY REPORT",
        "=" * 58,
        "",
        "All figures are based on synthetic educational data.",
        "",
        "BRANCH WITHDRAWAL SUMMARY",
        summary.round(2).to_string(),
        "",
        "BAYESIAN HIGH-DEMAND MONITORING",
        bayes_table.round(4).to_string(index=False),
        "",
        "RESERVE POLICY SIMULATION",
        policy_table.round(2).to_string(index=False),
        "",
        "LEAST-SQUARES TREND",
        trend_table.round(2).to_string(index=False),
        "",
        "INTERPRETATION",
        "- A higher reserve generally reduces the chance and size of a cash shortfall.",
        "- A higher reserve also increases the amount of excess cash held on quieter days.",
        "- The lowest estimated cost depends on the illustrative cost assumptions.",
        "- Bayesian estimates quantify uncertainty about high-demand frequency.",
        "- A trend is descriptive and should not be treated as a reliable forecast by itself.",
        "",
        "LIMITATIONS",
        "- The data are simulated and do not represent a real bank.",
        "- Withdrawal sizes and counts follow simplified distributions.",
        "- Cost rates are illustrative and require operational validation.",
        "- The Bayesian interval is a normal approximation to the Beta posterior.",
        "- Results are not a regulatory liquidity measure or production recommendation.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
