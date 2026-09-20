"""
app/economics/profitability.py
────────────────────────────────
Cost, scrap, rework and profitability simulation.

Placeholder — all monetary figures come from .env / config, not invented.
Fill in the TODOs once the organizer dataset reveals actual cost structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.config import settings


@dataclass
class ProfitabilityResult:
    """
    Summary of a single profitability simulation run.

    Fields are typed broadly until the dataset defines exact semantics.
    """
    total_units: int = 0
    defective_units: int = 0
    scrap_units: int = 0
    rework_units: int = 0

    scrap_cost: float = 0.0
    rework_cost: float = 0.0
    total_defect_cost: float = 0.0

    revenue_lost: float = 0.0
    net_profitability_impact: float = 0.0

    currency: str = field(default_factory=lambda: settings.currency)
    notes: list[str] = field(default_factory=list)


def estimate_scrap_cost(scrap_units: int, cost_per_unit: float | None = None) -> float:
    """
    Total cost of scrapped units.

    Args:
        scrap_units:   Number of units scrapped.
        cost_per_unit: Override per-unit cost; defaults to
                       ``settings.cost_per_defect``.

    Returns:
        Total scrap cost.

    Raises:
        ValueError: If ``scrap_units`` < 0.
    """
    if scrap_units < 0:
        raise ValueError("scrap_units must be >= 0")
    unit_cost = cost_per_unit if cost_per_unit is not None else settings.cost_per_defect
    return scrap_units * unit_cost


def estimate_rework_cost(
    rework_units: int,
    rework_cost_per_unit: float,
) -> float:
    """
    Total cost of reworking defective units.

    TODO:
        - Confirm whether the organizer dataset includes rework time/cost data.

    Args:
        rework_units:        Number of units sent for rework.
        rework_cost_per_unit: Cost to rework a single unit.

    Returns:
        Total rework cost.

    Raises:
        ValueError: If either argument is negative.
    """
    if rework_units < 0 or rework_cost_per_unit < 0:
        raise ValueError("rework_units and rework_cost_per_unit must be >= 0")
    return rework_units * rework_cost_per_unit


def estimate_revenue_loss(
    lost_units: int,
    revenue_per_unit: float | None = None,
) -> float:
    """
    Revenue foregone due to scrapped or unshipped units.

    Args:
        lost_units:      Number of units that could not be sold.
        revenue_per_unit: Override; defaults to ``settings.revenue_per_unit``.

    Returns:
        Total revenue loss.
    """
    rev = revenue_per_unit if revenue_per_unit is not None else settings.revenue_per_unit
    return lost_units * rev


def run_profitability_simulation(
    total_units: int,
    defect_df: pd.DataFrame,
    rework_cost_per_unit: float = 0.0,
    scrap_fraction: float = 1.0,
) -> ProfitabilityResult:
    """
    Combine scrap, rework, and revenue-loss estimates into one result object.

    TODO:
        - Map defect_df columns to scrap vs. rework categories.
        - Confirm scrap_fraction vs. rework_fraction split from dataset.

    Args:
        total_units:          Total units produced in the analysis window.
        defect_df:            DataFrame of defect records.
        rework_cost_per_unit: Cost to rework one unit.
        scrap_fraction:       Fraction of defective units scrapped (rest reworked).

    Returns:
        :class:`ProfitabilityResult` with all cost fields populated.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("run_profitability_simulation — pending dataset schema inspection")


def profitability_summary_table(result: ProfitabilityResult) -> pd.DataFrame:
    """
    Convert a :class:`ProfitabilityResult` to a display-ready DataFrame.

    Args:
        result: Output from :func:`run_profitability_simulation`.

    Returns:
        DataFrame with columns ``metric`` and ``value``.
    """
    rows = [
        ("Total units",              result.total_units),
        ("Defective units",          result.defective_units),
        ("Scrap units",              result.scrap_units),
        ("Rework units",             result.rework_units),
        (f"Scrap cost ({result.currency})",         result.scrap_cost),
        (f"Rework cost ({result.currency})",        result.rework_cost),
        (f"Revenue lost ({result.currency})",       result.revenue_lost),
        (f"Net impact ({result.currency})",         result.net_profitability_impact),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
