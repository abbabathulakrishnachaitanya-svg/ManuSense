"""
app/analytics/metrics.py
──────────────────────────
Quality and model-performance metrics.

Placeholder — exact metrics depend on the organizer dataset's
available ground-truth labels and production records.
"""

from __future__ import annotations

import pandas as pd


# ── Quality KPIs ─────────────────────────────────────────────────────────────

def defect_rate(total_inspected: int, total_defective: int) -> float:
    """
    Defective parts / total inspected.

    Args:
        total_inspected: Number of units inspected.
        total_defective: Number of units with at least one defect.

    Returns:
        Defect rate as a fraction in [0, 1].

    Raises:
        ValueError: If ``total_inspected`` is zero.
    """
    if total_inspected == 0:
        raise ValueError("total_inspected must be > 0")
    return total_defective / total_inspected


def first_pass_yield(total_inspected: int, total_defective: int) -> float:
    """
    Fraction of units passing inspection without rework (1 − defect_rate).

    Args:
        total_inspected: Number of units inspected.
        total_defective: Number of defective units.

    Returns:
        First-pass yield as a fraction in [0, 1].
    """
    return 1.0 - defect_rate(total_inspected, total_defective)


def defects_by_type(df: pd.DataFrame, defect_col: str) -> pd.Series:
    """
    Count occurrences of each defect class.

    TODO:
        - Confirm the column name for defect class in the organizer dataset.

    Args:
        df:         Inspection records DataFrame.
        defect_col: Column containing defect class labels.

    Returns:
        pd.Series indexed by defect class, values = count.

    Raises:
        KeyError: If ``defect_col`` is not present in ``df``.
    """
    if defect_col not in df.columns:
        raise KeyError(f"Column '{defect_col}' not found in DataFrame")
    return df[defect_col].value_counts()


# ── Model evaluation KPIs ────────────────────────────────────────────────────

def classification_report_df(y_true: list, y_pred: list) -> pd.DataFrame:
    """
    Precision, recall, F1, and support per class as a DataFrame.

    TODO:
        - Wire up once the model produces real predictions.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        DataFrame with columns: class, precision, recall, f1, support.

    Raises:
        NotImplementedError: Until model is implemented.
    """
    raise NotImplementedError("classification_report_df — pending model implementation")


def mean_average_precision(predictions: list, ground_truths: list) -> float:
    """
    mAP@0.5 for bounding-box localisation (if the dataset has bbox labels).

    TODO:
        - Confirm whether the organizer dataset includes bounding-box annotations.

    Args:
        predictions:   List of predicted bbox dicts per image.
        ground_truths: List of ground-truth bbox dicts per image.

    Returns:
        mAP score as a float in [0, 1].

    Raises:
        NotImplementedError: Until model and annotation format are confirmed.
    """
    raise NotImplementedError("mean_average_precision — pending annotation format confirmation")
