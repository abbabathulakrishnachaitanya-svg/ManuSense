"""
app/analytics/root_cause.py
────────────────────────────
Root-cause analysis for detected defects.

Placeholder — logic depends on the organizer dataset schema.
Fill in the TODOs after inspecting available columns (machine ID,
shift, operator, material batch, process parameters, etc.).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_defect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate defect counts and rates from the inspection dataset.

    TODO:
        - Confirm column names for defect label, timestamp, line/station ID.
        - Decide aggregation granularity (per shift, per line, per SKU, etc.).

    Args:
        df: Raw inspection DataFrame loaded from the organizer dataset.

    Returns:
        Summary DataFrame with defect counts and rates per group.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("build_defect_summary — pending dataset schema inspection")


def correlate_defects_with_process(
    defect_df: pd.DataFrame,
    process_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join defect records with process/production parameters and compute
    correlations to surface likely root-cause candidates.

    TODO:
        - Identify join keys (timestamp, batch_id, machine_id, etc.).
        - Choose correlation method (Pearson, Spearman, Cramér's V for
          categorical variables).
        - Decide significance threshold.

    Args:
        defect_df:  Defect / inspection records.
        process_df: Process parameter records (temperature, speed, pressure…).

    Returns:
        DataFrame of (feature, correlation_score, p_value) sorted by
        absolute correlation descending.

    Raises:
        NotImplementedError: Until dataset schema is known.
    """
    raise NotImplementedError("correlate_defects_with_process — pending dataset schema inspection")


def rank_root_causes(correlation_df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """
    Return the top-N most likely root-cause factors as a ranked list.

    Args:
        correlation_df: Output of :func:`correlate_defects_with_process`.
        top_n:          Number of candidates to return.

    Returns:
        List of dicts with keys ``feature``, ``score``, ``direction``.

    Raises:
        NotImplementedError: Until correlation step is implemented.
    """
    raise NotImplementedError("rank_root_causes — pending correlation step")
