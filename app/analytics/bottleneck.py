"""
app/analytics/bottleneck.py
─────────────────────────────
Production bottleneck and throughput-impact analysis.

Placeholder — implementation requires production/line dataset columns
(cycle time, queue depth, machine utilisation, downtime events, etc.).
Fill in the TODOs after the organizer dataset has been inspected.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def identify_bottlenecks(production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify stations or machines with the highest throughput constraint.

    Common approaches (pick after dataset inspection):
        - Longest average cycle time per station.
        - Highest queue / WIP accumulation upstream.
        - Lowest OEE (Overall Equipment Effectiveness).

    TODO:
        - Confirm column names (station_id, cycle_time, downtime_min, etc.).
        - Choose and implement the bottleneck detection algorithm.

    Args:
        production_df: Production records DataFrame.

    Returns:
        DataFrame of stations ranked by constraint severity.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("identify_bottlenecks — pending dataset schema inspection")


def compute_throughput_impact(
    production_df: pd.DataFrame,
    defect_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify how defect-related rework / scrap reduces effective throughput.

    TODO:
        - Link defect records to production records via a shared key.
        - Estimate rework time or scrap rate per defect type.
        - Propagate impact to line-level throughput figures.

    Args:
        production_df: Production records (cycle times, line speeds, etc.).
        defect_df:     Defect / inspection records.

    Returns:
        DataFrame with columns: station, baseline_throughput,
        defect_adjusted_throughput, throughput_loss_pct.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("compute_throughput_impact — pending dataset schema inspection")


def summarise_line_performance(production_df: pd.DataFrame) -> dict[str, Any]:
    """
    High-level KPIs for the overall production line.

    TODO:
        - OEE, yield rate, takt time vs. cycle time, etc.
        - Confirm which metrics are derivable from the organizer dataset.

    Args:
        production_df: Production records DataFrame.

    Returns:
        Dict of KPI name → value.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("summarise_line_performance — pending dataset schema inspection")
