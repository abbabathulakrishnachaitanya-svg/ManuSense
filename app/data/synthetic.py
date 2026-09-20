"""
app/data/synthetic.py
──────────────────────
Reproducible synthetic manufacturing production and economics data generator.

IMPORTANT — DATA PROVENANCE
────────────────────────────
Everything produced by this module is SYNTHETIC DEMONSTRATION DATA.
It is NOT real factory data.
It is NOT based on any real production line.
It demonstrates how the analytics pipeline would behave when actual
production data is connected.

All synthetic associations are clearly documented below so no one
mistakes them for validated industrial causal relationships.

Synthetic associations built in
────────────────────────────────
  - Station S3 has an elevated defect rate (simulated quality issue)
  - High vibration correlates with crack and scratch defects
  - High pressure correlates with hole defects
  - High temperature + humidity correlates with rust defects
  - Shift C has a slightly higher overall defect rate
  - Changeover periods associated with transient quality reduction
  - Station S3 has the highest cycle time (simulated bottleneck)

Run as a script to regenerate CSVs:
    python -m app.data.synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
SEED          = 42
N_UNITS       = 500          # synthetic production records
CLASSES       = ["normal", "crack", "hole", "rust", "scratch"]
STATIONS      = ["S1", "S2", "S3", "S4", "S5"]
SHIFTS        = ["A", "B", "C"]
VARIANTS      = ["V1", "V2", "V3"]
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed"

# Economics constants (clearly synthetic)
SELLING_PRICE    = 45.00   # USD per unit
MATERIAL_COST    = 12.00
PROCESSING_COST  =  8.00
SCRAP_COST       = 25.00   # cost to scrap a defective unit
REWORK_COST      = 10.00   # cost to rework a defective unit (if reworkable)
DOWNTIME_COST_PM =  2.50   # cost per minute of downtime


def generate_production_df(seed: int = SEED) -> pd.DataFrame:
    """
    Generate N_UNITS rows of synthetic production records.

    Each row represents one inspected unit and includes:
    - Inspection metadata (unit_id, batch, station, variant, shift, timestamp)
    - A simulated defect label (true_defect_label) — NOT a model prediction
    - Process parameters (cycle_time, downtime, wip, temperature, pressure,
      machine_speed, vibration, humidity, changeover_minutes)

    All values are SYNTHETIC. Associations between process parameters and
    defect types are deliberately constructed for demonstration purposes only.

    Returns
    -------
    pd.DataFrame with columns documented in the column list below.
    """
    rng = np.random.default_rng(seed)

    # ── Unit IDs and metadata ─────────────────────────────────────────────────
    unit_ids   = [f"U{i:05d}" for i in range(1, N_UNITS + 1)]
    batch_ids  = [f"B{(i // 50) + 1:03d}" for i in range(N_UNITS)]
    stations   = rng.choice(STATIONS, N_UNITS, p=[0.22, 0.22, 0.28, 0.14, 0.14])
    variants   = rng.choice(VARIANTS, N_UNITS, p=[0.50, 0.30, 0.20])
    shifts     = rng.choice(SHIFTS,   N_UNITS, p=[0.36, 0.36, 0.28])

    # Timestamps: one unit every ~3 minutes over ~25 hours
    start = pd.Timestamp("2024-01-15 06:00:00")
    timestamps = pd.date_range(start=start, periods=N_UNITS, freq="3min")

    # ── Process parameters ────────────────────────────────────────────────────
    # Cycle time — Station S3 is slower (simulated bottleneck)
    cycle_time = np.where(
        stations == "S3",
        rng.normal(55, 5, N_UNITS),   # S3: slower
        rng.normal(38, 4, N_UNITS),   # others: normal
    ).clip(20, 90)

    downtime     = rng.exponential(scale=2.0, size=N_UNITS).clip(0, 30)   # minutes
    changeover   = np.where(rng.random(N_UNITS) < 0.08,
                            rng.uniform(10, 25, N_UNITS), 0.0)             # 8% of shifts
    wip          = rng.integers(2, 20, N_UNITS).astype(float)
    temperature  = rng.normal(22, 4, N_UNITS).clip(10, 40)                # °C
    pressure     = rng.normal(5.0, 0.8, N_UNITS).clip(2, 9)               # bar
    machine_speed = rng.normal(120, 15, N_UNITS).clip(70, 180)            # rpm
    vibration    = rng.normal(0.5, 0.2, N_UNITS).clip(0.1, 1.5)          # mm/s RMS
    humidity     = rng.normal(55, 10, N_UNITS).clip(20, 90)               # %RH

    # ── Defect label — SYNTHETIC ASSOCIATIONS ─────────────────────────────────
    # Base probability: 80 % normal, 20 % defect spread across 4 defect classes
    defect_label = []
    for i in range(N_UNITS):
        p_defect = 0.20   # baseline

        # Station S3 has elevated defect rate (simulated)
        if stations[i] == "S3":
            p_defect = 0.35

        # Shift C has slightly higher defect rate
        if shifts[i] == "C":
            p_defect = min(p_defect + 0.05, 0.50)

        # Changeover periods → transient quality reduction
        if changeover[i] > 0:
            p_defect = min(p_defect + 0.10, 0.60)

        if rng.random() > p_defect:
            defect_label.append("normal")
        else:
            # Which defect class? Driven by process conditions
            # High vibration → crack or scratch
            # High pressure  → hole
            # High temp+humidity → rust
            v   = vibration[i]
            pr  = pressure[i]
            tmp = temperature[i]
            hum = humidity[i]

            scores = {
                "crack":   max(0, v - 0.4) * 3.0,
                "scratch": max(0, v - 0.4) * 2.0,
                "hole":    max(0, pr - 4.5) * 2.0,
                "rust":    max(0, (tmp - 26) / 10 + (hum - 60) / 20),
            }
            # Normalise to probabilities
            total = sum(scores.values()) + 1e-6   # avoid zero-sum
            raw   = np.array([scores[c] for c in ["crack", "hole", "rust", "scratch"]], dtype=float)
            raw  += 1e-6                           # ensure every class has nonzero weight
            probs = raw / raw.sum()
            defect_label.append(
                rng.choice(["crack", "hole", "rust", "scratch"], p=probs)
            )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "unit_id":          unit_ids,
        "batch_id":         batch_ids,
        "station_id":       stations,
        "product_variant":  variants,
        "shift":            shifts,
        "timestamp":        timestamps,
        "true_defect_label": defect_label,
        "cycle_time_s":     cycle_time.round(1),
        "downtime_min":     downtime.round(2),
        "changeover_min":   changeover.round(1),
        "wip":              wip,
        "temperature_c":    temperature.round(1),
        "pressure_bar":     pressure.round(2),
        "machine_speed_rpm": machine_speed.round(0).astype(int),
        "vibration_mms":    vibration.round(3),
        "humidity_pct":     humidity.round(1),
        # Synthetic flag — always True; never suppress this column
        "is_synthetic":     True,
    })

    return df


def generate_economics_df(production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate per-unit economics records from the production DataFrame.

    All monetary values are SYNTHETIC.
    They do not represent real factory costs or revenues.

    Returns
    -------
    pd.DataFrame with columns:
        unit_id, true_defect_label, is_defective,
        selling_price, material_cost, processing_cost,
        scrap_cost_applied, rework_cost_applied,
        downtime_loss, estimated_contribution,
        disposition (sold / scrapped / reworked),
        is_synthetic
    """
    rng = np.random.default_rng(SEED + 1)

    records = []
    for _, row in production_df.iterrows():
        is_defective = row["true_defect_label"] != "normal"

        if not is_defective:
            disposition    = "sold"
            scrap_applied  = 0.0
            rework_applied = 0.0
        else:
            # 60 % of defects are scrapped, 40 % reworked (synthetic assumption)
            if rng.random() < 0.60:
                disposition    = "scrapped"
                scrap_applied  = SCRAP_COST
                rework_applied = 0.0
            else:
                disposition    = "reworked"
                scrap_applied  = 0.0
                rework_applied = REWORK_COST

        downtime_loss = row["downtime_min"] * DOWNTIME_COST_PM

        contribution = (
            SELLING_PRICE
            - MATERIAL_COST
            - PROCESSING_COST
            - scrap_applied
            - rework_applied
            - downtime_loss
        )

        records.append({
            "unit_id":               row["unit_id"],
            "true_defect_label":     row["true_defect_label"],
            "is_defective":          is_defective,
            "selling_price":         SELLING_PRICE,
            "material_cost":         MATERIAL_COST,
            "processing_cost":       PROCESSING_COST,
            "scrap_cost_applied":    round(scrap_applied, 2),
            "rework_cost_applied":   round(rework_applied, 2),
            "downtime_loss":         round(downtime_loss, 2),
            "estimated_contribution": round(contribution, 2),
            "disposition":           disposition,
            "is_synthetic":          True,
        })

    return pd.DataFrame(records)


def load_or_generate(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing CSVs if present, otherwise generate and save them.

    Args:
        force: If True, regenerate even if CSVs exist.

    Returns:
        (production_df, economics_df)
    """
    prod_path = OUTPUT_DIR / "synthetic_production.csv"
    econ_path = OUTPUT_DIR / "synthetic_economics.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and prod_path.exists() and econ_path.exists():
        prod_df = pd.read_csv(prod_path, parse_dates=["timestamp"])
        econ_df = pd.read_csv(econ_path)
        return prod_df, econ_df

    prod_df = generate_production_df()
    econ_df = generate_economics_df(prod_df)
    prod_df.to_csv(prod_path, index=False)
    econ_df.to_csv(econ_path, index=False)
    return prod_df, econ_df


# ── Bottleneck calculation ─────────────────────────────────────────────────────

def compute_station_metrics(production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-station capacity, utilisation, and defect rate.

    SIMULATED BOTTLENECK ANALYSIS — not real factory measurements.

    Returns DataFrame with columns:
        station_id, unit_count, avg_cycle_time_s, total_downtime_min,
        total_changeover_min, effective_available_min, est_capacity_units,
        utilisation_pct, defect_count, defect_rate_pct
    """
    scheduled_minutes = 24 * 60  # 24-hour window

    rows = []
    for station, grp in production_df.groupby("station_id"):
        n          = len(grp)
        avg_ct     = grp["cycle_time_s"].mean()
        total_down = grp["downtime_min"].sum()
        total_co   = grp["changeover_min"].sum()
        eff_avail  = max(scheduled_minutes - total_down - total_co, 1)
        capacity   = (eff_avail * 60) / avg_ct   # units per 24 h
        defects    = (grp["true_defect_label"] != "normal").sum()

        rows.append({
            "station_id":            station,
            "unit_count":            n,
            "avg_cycle_time_s":      round(avg_ct, 1),
            "total_downtime_min":    round(total_down, 1),
            "total_changeover_min":  round(total_co, 1),
            "effective_available_min": round(eff_avail, 1),
            "est_capacity_units":    round(capacity, 0),
            "utilisation_pct":       round(n / capacity * 100, 1),
            "defect_count":          int(defects),
            "defect_rate_pct":       round(defects / n * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("avg_cycle_time_s", ascending=False)
    return df


# ── Economics summary ─────────────────────────────────────────────────────────

def compute_economics_summary(economics_df: pd.DataFrame) -> dict:
    """
    Roll-up economics figures from the per-unit economics DataFrame.

    Returns a dict suitable for display in the dashboard.
    All values are SYNTHETIC.
    """
    total   = len(economics_df)
    defects = economics_df["is_defective"].sum()
    scrapped = (economics_df["disposition"] == "scrapped").sum()
    reworked = (economics_df["disposition"] == "reworked").sum()

    return {
        "total_units":           total,
        "defective_units":       int(defects),
        "defect_rate_pct":       round(defects / total * 100, 1),
        "scrapped_units":        int(scrapped),
        "reworked_units":        int(reworked),
        "total_scrap_cost":      round(economics_df["scrap_cost_applied"].sum(), 2),
        "total_rework_cost":     round(economics_df["rework_cost_applied"].sum(), 2),
        "total_downtime_loss":   round(economics_df["downtime_loss"].sum(), 2),
        "total_contribution":    round(economics_df["estimated_contribution"].sum(), 2),
        "avg_contribution_unit": round(economics_df["estimated_contribution"].mean(), 2),
        "currency":              "USD",
    }


if __name__ == "__main__":
    print("Generating synthetic data …")
    prod, econ = load_or_generate(force=True)
    print(f"Production records : {len(prod)}")
    print(f"Economics records  : {len(econ)}")
    station_metrics = compute_station_metrics(prod)
    print("\nStation metrics (SIMULATED):")
    print(station_metrics.to_string(index=False))
    summary = compute_economics_summary(econ)
    print("\nEconomics summary (SYNTHETIC):")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print(f"\nCSVs saved to {OUTPUT_DIR.resolve()}")
