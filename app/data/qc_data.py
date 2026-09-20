"""
app/data/qc_data.py
────────────────────
Loader, normaliser, and analytical functions for the
Infoveave Product Quality Control sample dataset
(data/product-quality-control.csv, 944 records).

All normalisations are documented here.  The original CSV is never modified.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "product-quality-control.csv"

# ── Normalisation map for qc_result ──────────────────────────────────────────
# The CSV contains mixed-case variants: PASS / pass / FAIL / fail / PENDING / NaN
# We map these to canonical uppercase values for display/analysis.
_QC_NORM: dict[str, str] = {
    "pass":    "PASS",
    "fail":    "FAIL",
    "pending": "PENDING",
}

# Vision-model class → nearest QC defect type mapping
# Used to bridge the ResNet18 prediction to historical QC context.
VISION_TO_QC: dict[str, list[str]] = {
    "crack":   ["Crack"],
    "scratch": ["Surface Scratch"],
    "hole":    ["Dimensional Error", "Missing Part"],
    "rust":    ["Contamination", "Colour Deviation"],
    "normal":  [],   # no defect — no matching QC type
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADING & NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """
    Load the CSV exactly as-is.  Cached — disk is read only once per session.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"QC dataset not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    log.info("QC dataset loaded: %d rows, %d cols", *df.shape)
    return df


def load_normalised() -> pd.DataFrame:
    """
    Return a clean working copy of the dataset with:

    Normalisations applied (documented):
    1. qc_result: lowercase variants mapped to uppercase (PASS/FAIL/PENDING).
       NaN retained as NaN — not imputed.
    2. inspection_date: parsed to datetime; unparseable values → NaT.
    3. Numeric columns (units_inspected, units_passed, defect_count):
       remain float; NaN rows are NOT dropped — callers decide how to handle them.
    4. defect_type: stripped of leading/trailing whitespace where present.
       NaN retained (most records have no defect type recorded).

    The original DataFrame from load_raw() is NEVER mutated.
    """
    df = load_raw().copy()

    # 1. Normalise qc_result
    df["qc_result_norm"] = (
        df["qc_result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_QC_NORM)
    )
    # Where the original was NaN, map returns NaN — correct.

    # 2. Parse dates
    df["date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year

    # 3. Numeric coercion (already float64 from CSV read; just ensure)
    for col in ("units_inspected", "units_passed", "defect_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Strip defect_type whitespace
    df["defect_type"] = df["defect_type"].astype(str).str.strip()
    df.loc[df["defect_type"] == "nan", "defect_type"] = np.nan

    return df


def data_quality_report(df: pd.DataFrame | None = None) -> dict:
    """
    Return a summary of data-quality issues in the raw dataset.
    Used by the Data Explorer page to be transparent about data problems.
    """
    raw = load_raw() if df is None else df
    total = len(raw)

    # qc_result issues
    qc_vals = raw["qc_result"].astype(str).str.strip()
    mixed_case = int(qc_vals.str.lower().ne(qc_vals).sum())
    qc_null = int(raw["qc_result"].isna().sum())

    # defect_type missing
    defect_null = int(raw["defect_type"].isna().sum())

    # numeric nulls
    numeric_nulls = {
        col: int(raw[col].isna().sum())
        for col in ("units_inspected", "units_passed", "defect_count")
    }

    # date parse issues
    parsed_dates = pd.to_datetime(raw["inspection_date"], errors="coerce")
    date_errors = int(parsed_dates.isna().sum())

    return {
        "total_records":         total,
        "qc_result_null":        qc_null,
        "qc_result_mixed_case":  mixed_case,
        "defect_type_missing":   defect_null,
        "defect_type_pct_missing": round(defect_null / total * 100, 1),
        "numeric_nulls":         numeric_nulls,
        "date_parse_errors":     date_errors,
        "usable_records":        int((~raw["qc_result"].isna()).sum()),
        "normalisations_applied": [
            "qc_result: PASS/pass → PASS, FAIL/fail → FAIL, PENDING → PENDING",
            "inspection_date: parsed to datetime (errors → NaT)",
            "defect_type: whitespace stripped; 'nan' string → NaN",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    lines: list[str] | None = None,
    products: list[str] | None = None,
    defect_types: list[str] | None = None,
    qc_results: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """Filter the normalised dataframe. All parameters are optional."""
    out = df.copy()
    if lines:
        out = out[out["line_id"].isin(lines)]
    if products:
        out = out[out["product_code"].isin(products)]
    if defect_types:
        out = out[out["defect_type"].isin(defect_types)]
    if qc_results:
        out = out[out["qc_result_norm"].isin(qc_results)]
    if date_start:
        out = out[out["date"] >= pd.to_datetime(date_start, errors="coerce")]
    if date_end:
        out = out[out["date"] <= pd.to_datetime(date_end, errors="coerce")]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# KPI CALCULATIONS  — all derived from source data, never hard-coded
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute headline KPIs from a (filtered) normalised dataframe.

    Returns dict of scalar values safe to display in metric cards.
    """
    total_inspections  = len(df)
    total_units        = int(df["units_inspected"].sum(skipna=True))
    total_defects      = int(df["defect_count"].sum(skipna=True))
    total_passed_units = int(df["units_passed"].sum(skipna=True))

    pass_rate = (
        round(total_passed_units / total_units * 100, 2)
        if total_units > 0 else 0.0
    )
    defect_rate = (
        round(total_defects / total_units * 100, 2)
        if total_units > 0 else 0.0
    )

    qc_pass   = int((df["qc_result_norm"] == "PASS").sum())
    qc_fail   = int((df["qc_result_norm"] == "FAIL").sum())
    qc_pend   = int((df["qc_result_norm"] == "PENDING").sum())
    qc_unknown= total_inspections - qc_pass - qc_fail - qc_pend

    batch_fail_rate = (
        round(qc_fail / (qc_pass + qc_fail) * 100, 1)
        if (qc_pass + qc_fail) > 0 else 0.0
    )

    return {
        "total_inspections":   total_inspections,
        "total_units":         total_units,
        "total_defects":       total_defects,
        "total_passed_units":  total_passed_units,
        "pass_rate_pct":       pass_rate,
        "defect_rate_pct":     defect_rate,
        "qc_pass":             qc_pass,
        "qc_fail":             qc_fail,
        "qc_pending":          qc_pend,
        "qc_unknown":          qc_unknown,
        "batch_fail_rate_pct": batch_fail_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────

def defects_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Count records by defect type (excludes rows where defect_type is NaN)."""
    sub = df.dropna(subset=["defect_type"])
    return (
        sub.groupby("defect_type")
        .agg(count=("defect_type", "size"), total_defects=("defect_count", "sum"))
        .reset_index()
        .sort_values("count", ascending=False)
    )


def defects_by_line(df: pd.DataFrame) -> pd.DataFrame:
    """Defect count and defect rate per production line."""
    grp = (
        df.groupby("line_id")
        .agg(
            inspections=("batch_id", "count"),
            units_inspected=("units_inspected", "sum"),
            defect_count=("defect_count", "sum"),
        )
        .reset_index()
    )
    grp["defect_rate_pct"] = (
        grp["defect_count"] / grp["units_inspected"].replace(0, np.nan) * 100
    ).round(2)
    return grp.sort_values("defect_count", ascending=False)


def defects_by_product(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N products by defect count."""
    grp = (
        df.groupby("product_code")
        .agg(
            inspections=("batch_id", "count"),
            units_inspected=("units_inspected", "sum"),
            defect_count=("defect_count", "sum"),
        )
        .reset_index()
    )
    grp["defect_rate_pct"] = (
        grp["defect_count"] / grp["units_inspected"].replace(0, np.nan) * 100
    ).round(2)
    return grp.sort_values("defect_count", ascending=False).head(top_n)


def trend_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation: inspections, defects, pass/fail counts."""
    valid = df.dropna(subset=["date"])
    grp = (
        valid.groupby("year_month")
        .agg(
            inspections=("batch_id", "count"),
            units_inspected=("units_inspected", "sum"),
            defect_count=("defect_count", "sum"),
            qc_pass=("qc_result_norm", lambda x: (x == "PASS").sum()),
            qc_fail=("qc_result_norm", lambda x: (x == "FAIL").sum()),
        )
        .reset_index()
        .sort_values("year_month")
    )
    grp["defect_rate_pct"] = (
        grp["defect_count"] / grp["units_inspected"].replace(0, np.nan) * 100
    ).round(2)
    return grp


def defect_type_by_line(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation: defect type × production line."""
    sub = df.dropna(subset=["defect_type"])
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["defect_type", "line_id"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def qc_result_by_line(df: pd.DataFrame) -> pd.DataFrame:
    """PASS/FAIL counts per production line."""
    sub = df.dropna(subset=["qc_result_norm"])
    return (
        sub.groupby(["line_id", "qc_result_norm"])
        .size()
        .reset_index(name="count")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROOT-CAUSE ASSOCIATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def association_analysis(df: pd.DataFrame, defect_type: str) -> dict:
    """
    Descriptive association analysis for a selected defect type.

    Returns structured observations for the Root-Cause page.
    All findings are labelled as 'observed associations' — not proven causes.
    """
    total_with_defect = len(df.dropna(subset=["defect_type"]))
    sub = df[df["defect_type"] == defect_type].copy()
    n = len(sub)

    if n == 0 or total_with_defect == 0:
        return {"defect_type": defect_type, "count": 0, "observations": []}

    freq_pct = round(n / total_with_defect * 100, 1)

    observations: list[dict] = []

    # -- Line association
    line_counts = sub["line_id"].value_counts()
    if not line_counts.empty:
        top_line = line_counts.index[0]
        top_line_pct = round(line_counts.iloc[0] / n * 100, 1)
        observations.append({
            "type":       "production_line",
            "finding":    f"{defect_type} appears most frequently on {top_line} "
                          f"({line_counts.iloc[0]} of {n} occurrences, {top_line_pct}%)",
            "label":      "Observed association",
            "interpretation": f"May warrant investigation of process conditions, "
                              f"tooling, or handling on {top_line}.",
            "validation": "Confirm using actual process/maintenance records.",
            "line_counts": line_counts.to_dict(),
        })

    # -- Product association
    prod_counts = sub["product_code"].value_counts().head(5)
    if not prod_counts.empty:
        top_prod = prod_counts.index[0]
        observations.append({
            "type":       "product_code",
            "finding":    f"{defect_type} is recorded most on {top_prod} "
                          f"({prod_counts.iloc[0]} occurrences)",
            "label":      "Observed association",
            "interpretation": "Product-specific geometry or process may be relevant.",
            "validation": "Cross-check with product design and routing records.",
            "top_products": prod_counts.to_dict(),
        })

    # -- Time association
    if "date" in sub.columns and sub["date"].notna().any():
        monthly = sub.groupby("year_month").size()
        if len(monthly) > 1:
            peak_month = monthly.idxmax()
            observations.append({
                "type":       "time_trend",
                "finding":    f"Highest single-month occurrence: {peak_month} "
                              f"({monthly.max()} records)",
                "label":      "Observed pattern",
                "interpretation": "Elevated frequency in a specific period may indicate "
                                  "a batch, supplier, or seasonal process issue.",
                "validation": "Review batch records and supplier changes for that period.",
                "monthly": monthly.to_dict(),
            })

    # -- QC result association
    qc_dist = sub["qc_result_norm"].value_counts(dropna=True)
    if not qc_dist.empty:
        fail_n = int(qc_dist.get("FAIL", 0))
        fail_pct = round(fail_n / n * 100, 1) if n > 0 else 0
        observations.append({
            "type":       "qc_result",
            "finding":    f"{fail_pct}% of {defect_type} records resulted in FAIL "
                          f"({fail_n} of {n})",
            "label":      "QC outcome pattern",
            "interpretation": "High FAIL rate suggests the defect consistently "
                              "fails acceptance criteria.",
            "validation": "Review acceptance thresholds and inspector calibration.",
            "qc_dist": qc_dist.to_dict(),
        })

    return {
        "defect_type":     defect_type,
        "count":           n,
        "freq_pct":        freq_pct,
        "observations":    observations,
    }


def corrective_action_guidance(defect_type: str) -> dict:
    """
    Return generic, operationally reasonable investigation areas for a defect type.
    These are SUGGESTED INVESTIGATION AREAS — not proven root causes.
    """
    guidance = {
        "Crack": {
            "investigation_areas": [
                "Mechanical stress or vibration during processing or transport",
                "Material brittleness or batch variation",
                "Fixture clamping force or alignment",
                "Cycle-time deviation from standard",
                "Tool wear or tooling condition",
            ],
            "immediate_actions": [
                "Hold affected batch pending inspection",
                "Log defect against batch ID and production line",
                "Notify process engineer",
            ],
        },
        "Surface Scratch": {
            "investigation_areas": [
                "Contact surfaces: conveyor belts, fixtures, transfer guides",
                "Part-to-part contact during transport or packaging",
                "Handling procedures at transfer points",
                "Recent tooling or maintenance changes",
                "Packaging material condition",
            ],
            "immediate_actions": [
                "Inspect handling path for abrasive contact points",
                "Hold affected batch for visual inspection",
                "Review line maintenance log for recent changes",
            ],
        },
        "Contamination": {
            "investigation_areas": [
                "Environmental conditions: humidity, airborne particles",
                "Cleaning procedures and intervals",
                "Material storage conditions",
                "Cross-contamination from adjacent processes",
                "Consumable material quality",
            ],
            "immediate_actions": [
                "Quarantine affected batch",
                "Check environmental monitoring logs",
                "Review cleaning and housekeeping records",
            ],
        },
        "Dimensional Error": {
            "investigation_areas": [
                "Tooling wear or calibration drift",
                "Machine setup and parameter verification",
                "Material dimensional tolerance variation",
                "Fixture alignment and repeatability",
                "Temperature effects on dimensional stability",
            ],
            "immediate_actions": [
                "Measure sample units against specification",
                "Check machine calibration records",
                "Hold production pending parameter review",
            ],
        },
        "Wrong Label": {
            "investigation_areas": [
                "Labelling process controls and verification steps",
                "Work order / routing accuracy",
                "Label stock management and storage",
                "Operator training and procedure compliance",
                "Label printing equipment calibration",
            ],
            "immediate_actions": [
                "Quarantine all units from affected batch",
                "Verify label stock and print queue",
                "Review work order documentation",
            ],
        },
        "Missing Part": {
            "investigation_areas": [
                "Assembly process completeness checks",
                "Part supply and kitting accuracy",
                "Work instruction clarity and completeness",
                "In-process verification steps",
                "Parts presentation and ergonomics",
            ],
            "immediate_actions": [
                "100% inspection of affected batch",
                "Review assembly work instructions",
                "Check parts kitting records",
            ],
        },
        "Colour Deviation": {
            "investigation_areas": [
                "Coating or paint process parameters (temperature, time, thickness)",
                "Material batch variation in colour properties",
                "Process environmental conditions (humidity, temperature)",
                "Equipment calibration and maintenance",
                "Incoming material quality control",
            ],
            "immediate_actions": [
                "Colour measurement of sample units",
                "Review coating process parameters log",
                "Hold batch pending colour specification check",
            ],
        },
    }
    default = {
        "investigation_areas": [
            "Review production line process parameters",
            "Check tooling and equipment condition",
            "Review operator training records",
            "Examine material traceability records",
            "Check environmental conditions",
        ],
        "immediate_actions": [
            "Hold affected batch pending investigation",
            "Document defect details and batch information",
            "Notify quality engineer",
        ],
    }
    result = guidance.get(defect_type, default).copy()
    result["disclaimer"] = (
        "These are suggested investigation areas based on common manufacturing "
        "practice for this defect category. They do not constitute a proven "
        "root cause. Process validation is required before corrective action."
    )
    return result
