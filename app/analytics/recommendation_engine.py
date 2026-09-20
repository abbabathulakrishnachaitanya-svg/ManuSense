"""
app/analytics/recommendation_engine.py
----------------------------------------
Dynamic recommendation engine for ManuSense.

Accepts the full inspection context (prediction, confidence, process metrics,
bottleneck, economics) and returns a structured InspectionContext dataclass
containing every piece of information the UI needs to render a complete,
connected workflow.

Nothing in this module calls Streamlit — it is pure Python.
The UI (streamlit_app.py) only renders what this module computes.

DATA PROVENANCE
    Vision results     : REAL  — trained ResNet18, challenge dataset
    Process/economics  : SYNTHETIC — demonstration data, clearly labelled
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── Per-defect knowledge base ─────────────────────────────────────────────────
# Each entry drives: observation text, contributing factors, inspection steps,
# corrective actions, verification, monitoring, and highlighted process variable.

DEFECT_KNOWLEDGE: dict[str, dict] = {
    "crack": {
        "observation": (
            "The model detected a crack-like surface pattern. "
            "Cracks typically appear as linear or branching discontinuities on the surface."
        ),
        "factors": [
            "Excessive mechanical stress or vibration during processing",
            "Process parameter deviation (cycle time, feed rate, machine speed)",
            "Tool wear or fixture misalignment introducing uneven force",
            "Post-changeover instability — quality often degrades transiently",
            "Material batch variation (hardness, brittleness)",
        ],
        "immediate": [
            "Hold this unit and adjacent units from the same batch pending verification",
            "Do NOT release to the next process step",
            "Log the defect against the current batch and station",
            "Notify the responsible process engineer",
        ],
        "inspect": [
            "Vibration levels at the responsible station — compare to approved baseline",
            "Cycle-time deviation from the standard operating range",
            "Tooling condition and wear — inspect for cracks, chips, or misalignment",
            "Fixture clamping force and alignment",
            "Defect rate trend across the last 3–5 batches (increasing trend = process drift)",
        ],
        "correct": [
            "Restore process parameters to the approved operating range if deviation is confirmed",
            "Recondition or replace worn tooling if confirmed as a contributing factor",
            "Implement vibration dampening or isolation if elevated vibration is confirmed",
            "Review and tighten fixturing if mechanical stress concentration is identified",
            "Increase post-changeover inspection frequency until stability is confirmed",
        ],
        "verify": [
            "Run a 10–20 unit verification batch after corrective action",
            "Inspect all verification units for crack indicators",
            "Compare crack defect rate: pre-correction vs. verification batch",
            "Document the corrective action and measured outcome",
        ],
        "monitor": [
            "Crack defect rate by station and batch (target: return to baseline)",
            "Vibration readings — set alert if above approved threshold",
            "Cycle-time deviation — flag if outside ±10% of standard",
        ],
        "process_var": "vibration_mms",
        "process_var_label": "Vibration (mm/s)",
        "highlight_station": True,
        "summary": "Inspect tooling, vibration, and process parameters. Verify after correction.",
    },

    "hole": {
        "observation": (
            "The model detected a hole-like pattern. "
            "Note: intentional geometric features (designed holes, slots, countersinks) "
            "in manufactured components can resemble the learned hole-defect class. "
            "Verify against the approved product drawing before concluding a defect is present."
        ),
        "factors": [
            "Intentional product geometry misidentified by the model (verify first)",
            "Tooling wear or breakage causing incorrect material removal",
            "Incorrect or drifted pressure/feed parameter",
            "Fixture misalignment causing off-location features",
            "Material thickness or hardness variation",
        ],
        "immediate": [
            "Compare this unit against the approved product drawing or reference image",
            "Verify whether the observed feature is intentional product geometry",
            "If geometry is confirmed as unexpected/defective: hold the batch",
            "Do NOT reject automatically without human confirmation",
        ],
        "inspect": [
            "Approved product drawing — is the observed feature expected?",
            "Tooling condition — check for wear, breakage, or misalignment",
            "Pressure and feed rate settings vs. approved operating range",
            "Station setup records — were any changes made to tooling or settings recently?",
            "Adjacent units — is the pattern consistent (tooling issue) or isolated (material)?",
        ],
        "correct": [
            "If intentional geometry: no corrective action — update reference/inspection process",
            "If defective: restore pressure/feed settings to approved range if deviation confirmed",
            "Replace or realign worn/broken tooling if confirmed",
            "Isolate and re-inspect full affected batch if a systematic error is suspected",
        ],
        "verify": [
            "Run a verification batch after any confirmed corrective action",
            "Compare to reference drawing at each verification unit",
            "Confirm defect rate has returned to baseline",
        ],
        "monitor": [
            "Hole/feature defect rate by station",
            "Tooling wear interval — log tool changes and associated quality outcome",
            "Pressure/feed deviation — alert if outside approved range",
        ],
        "process_var": "pressure_bar",
        "process_var_label": "Pressure (bar)",
        "highlight_station": True,
        "summary": (
            "First verify against product drawing. "
            "If defective: inspect tooling and process parameters. Verify after correction."
        ),
    },

    "rust": {
        "observation": (
            "The model detected a rust-like surface discolouration pattern. "
            "Rust typically indicates oxidation due to moisture exposure, "
            "coating failure, or adverse storage/handling conditions."
        ),
        "factors": [
            "Moisture or humidity exposure during storage or production",
            "Surface treatment / protective coating failure or insufficient coverage",
            "Extended storage beyond approved shelf life",
            "Condensation due to temperature fluctuation in storage area",
            "Handling-induced coating damage exposing bare metal",
        ],
        "immediate": [
            "Hold this unit and the associated batch pending verification",
            "Move affected units to a dry, controlled environment",
            "Log the defect against the batch and station",
            "Notify quality and materials engineering",
        ],
        "inspect": [
            "Surface treatment / coating process records for the affected batch",
            "Humidity and temperature logs for the production and storage areas",
            "Storage duration — has the material exceeded approved shelf life?",
            "Material handling records — inspect for coating damage points",
            "Environmental controls (dehumidifiers, seals) — check for failures",
        ],
        "correct": [
            "Quarantine affected batch for 100% inspection if rust is widespread",
            "Correct surface treatment or coating process if failure is confirmed",
            "Improve environmental controls (humidity target: <50% RH) if elevated humidity confirmed",
            "Update FIFO rotation procedures if shelf-life exceedance is identified",
            "Retrain handling personnel if mechanical coating damage is the root cause",
        ],
        "verify": [
            "Inspect a sample batch produced after corrective action",
            "Measure surface coating thickness and continuity if applicable",
            "Check humidity logs for 48 hours after environmental correction",
            "Compare rust incidence rate: pre- and post-correction",
        ],
        "monitor": [
            "Rust defect rate by batch and storage duration",
            "Humidity and temperature in storage areas — daily log",
            "Surface treatment process records — track yield per coating run",
        ],
        "process_var": "humidity_pct",
        "process_var_label": "Humidity (%)",
        "highlight_station": False,
        "summary": (
            "Check humidity, coating records, and storage conditions. "
            "Quarantine batch if widespread. Verify environment after correction."
        ),
    },

    "scratch": {
        "observation": (
            "The model detected a scratch-like surface mark. "
            "Scratches indicate physical contact damage — typically from handling, "
            "conveyor surfaces, fixtures, or part-to-part contact."
        ),
        "factors": [
            "Damaged or rough contact surface on conveyor, fixture, or tooling",
            "Improper handling during transfer between process steps",
            "Part-to-part contact during transport or packaging",
            "Excessive vibration causing relative movement between part and contact surface",
            "Recent maintenance or tooling change introducing a new contact edge",
        ],
        "immediate": [
            "Hold this unit pending verification",
            "Visually inspect the handling path for the last process step",
            "Identify whether the scratch pattern is consistent (fixed point) or random",
            "Log the defect against the current batch and station",
        ],
        "inspect": [
            "All contact surfaces: conveyor belts, fixtures, transfer guides, packaging inserts",
            "Scratch location and orientation — consistent location suggests a fixed contact point",
            "Vibration data — excessive vibration can cause relative part movement",
            "Recent maintenance or tool changes that introduced new contact edges",
            "Scratch rate trend across recent batches — sudden increase suggests a new contact point",
        ],
        "correct": [
            "Remove or repair the identified abrasive/damaged contact surface",
            "Add protective padding, separators, or liners at the identified contact point",
            "Replace damaged conveyor components or fixture inserts",
            "Adjust handling procedures to minimise direct part contact",
            "Verify tooling surface finish is within approved roughness specification",
        ],
        "verify": [
            "Run a 10–20 unit verification batch after removing/repairing the contact point",
            "Inspect verification units for scratch indicators",
            "Confirm scratch rate has returned to baseline",
            "Check the repaired surface condition after first production run",
        ],
        "monitor": [
            "Scratch defect rate by station — alert if above baseline",
            "Periodic inspection of high-risk contact surfaces",
            "Vibration readings at the responsible station",
        ],
        "process_var": "vibration_mms",
        "process_var_label": "Vibration (mm/s)",
        "highlight_station": True,
        "summary": (
            "Inspect all contact surfaces and handling path. "
            "Identify fixed contact point. Repair and verify with a batch run."
        ),
    },

    "normal": {
        "observation": "No defect indicators detected. The part appears to be within normal visual quality.",
        "factors":     [],
        "immediate":   ["Continue production — no hold required"],
        "inspect":     ["Continue standard sampling inspection per your quality plan"],
        "correct":     ["No corrective action required"],
        "verify":      ["Maintain standard monitoring frequency"],
        "monitor":     ["Defect rate trend — watch for gradual increase over time"],
        "process_var": None,
        "process_var_label": None,
        "highlight_station": False,
        "summary": "No defect detected. Continue normal operations and monitoring.",
    },
}

# ── Inspection context ────────────────────────────────────────────────────────

@dataclass
class InspectionContext:
    """
    Complete computed result for one inspection.
    The UI renders exactly this — it does no independent computation.
    """
    # ── Vision ─────────────────────────────────────────────────────────────
    inspection_id:    str = ""
    timestamp:        str = ""
    predicted_class:  str = ""
    confidence:       float = 0.0
    decision:         str = ""           # model decision: ACCEPT / REJECT / REVIEW
    display_decision: str = ""           # may be downgraded to REVIEW by assessment
    is_uncertain:     bool = False
    probabilities:    dict[str, float] = field(default_factory=dict)
    top3:             list[tuple[str, float]] = field(default_factory=list)
    gradcam_overlay:  Any = None         # numpy (224,224,3) or None
    gradcam_error:    str = ""

    # ── Assessment ──────────────────────────────────────────────────────────
    prediction_margin: float = 0.0      # top1 - top2 probability gap
    low_margin:        bool = False
    domain_notes:      list[str] = field(default_factory=list)
    review_reasons:    list[str] = field(default_factory=list)

    # ── Observation / knowledge ─────────────────────────────────────────────
    observation:      str = ""
    factors:          list[str] = field(default_factory=list)
    immediate_steps:  list[str] = field(default_factory=list)
    inspect_steps:    list[str] = field(default_factory=list)
    correct_steps:    list[str] = field(default_factory=list)
    verify_steps:     list[str] = field(default_factory=list)
    monitor_steps:    list[str] = field(default_factory=list)
    summary:          str = ""

    # ── Synthetic process context ───────────────────────────────────────────
    # Fixed for the session via deterministic seed from filename
    station_id:         str = ""
    batch_id:           str = ""
    shift:              str = ""
    product_variant:    str = ""
    cycle_time_s:       float = 0.0
    downtime_min:       float = 0.0
    wip:                int = 0
    changeover_min:     float = 0.0
    temperature_c:      float = 0.0
    pressure_bar:       float = 0.0
    machine_speed_rpm:  int = 0
    vibration_mms:      float = 0.0
    humidity_pct:       float = 0.0

    # ── Bottleneck ──────────────────────────────────────────────────────────
    bn_station:          str = ""
    bn_cycle_s:          float = 0.0
    bn_defect_pct:       float = 0.0
    bn_downtime_min:     float = 0.0
    bn_changeover_min:   float = 0.0
    bn_utilisation_pct:  float = 0.0
    bn_avg_other_s:      float = 0.0
    station_metrics_df:  Any = None      # pd.DataFrame

    # ── Economics ───────────────────────────────────────────────────────────
    currency:            str = "USD"
    defect_rate_pct:     float = 0.0
    total_scrap:         float = 0.0
    total_rework:        float = 0.0
    total_downtime_loss: float = 0.0
    total_quality_loss:  float = 0.0
    avg_contribution:    float = 0.0
    unit_scrap_cost:     float = 0.0
    unit_rework_cost:    float = 0.0
    pct_of_revenue:      float = 0.0

    # ── Recommendation ──────────────────────────────────────────────────────
    final_recommendation: str = ""
    evidence_points:      list[str] = field(default_factory=list)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_inspection_context(
    predicted_class: str,
    confidence: float,
    decision: str,
    is_uncertain: bool,
    probabilities: dict[str, float],
    gradcam_overlay: Any,
    gradcam_error: str,
    tensor: Any,
    prod_df: Any,
    station_metrics: Any,
    econ_summary: dict,
    filename: str,
) -> InspectionContext:
    """
    Build the complete InspectionContext from model outputs and synthetic data.

    Parameters are intentionally typed loosely (Any) to avoid hard torch/pandas
    imports in this module — callers pass real objects.
    """
    ctx = InspectionContext()

    # ── Deterministic session seed from filename ──────────────────────────────
    seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2 ** 31)

    # ── IDs & timestamp ───────────────────────────────────────────────────────
    short = hashlib.md5(filename.encode()).hexdigest()[:6].upper()
    ctx.inspection_id = f"MS-{datetime.now().strftime('%Y%m%d')}-{short}"
    ctx.timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Vision ────────────────────────────────────────────────────────────────
    ctx.predicted_class  = predicted_class
    ctx.confidence       = confidence
    ctx.decision         = decision
    ctx.is_uncertain     = is_uncertain
    ctx.probabilities    = probabilities
    ctx.gradcam_overlay  = gradcam_overlay
    ctx.gradcam_error    = gradcam_error

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    ctx.top3 = sorted_probs[:3]

    top1 = sorted_probs[0][1] if sorted_probs else 0.0
    top2 = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    ctx.prediction_margin = top1 - top2
    ctx.low_margin = ctx.prediction_margin < 0.30

    # ── Assessment & display decision ─────────────────────────────────────────
    reasons = _assess(predicted_class, confidence, decision, ctx.prediction_margin,
                      ctx.low_margin, tensor)
    ctx.review_reasons   = reasons["reasons"]
    ctx.domain_notes     = reasons["domain_notes"]
    ctx.display_decision = reasons["display_decision"]

    # ── Knowledge base ────────────────────────────────────────────────────────
    kb = DEFECT_KNOWLEDGE.get(predicted_class, DEFECT_KNOWLEDGE["normal"])
    ctx.observation     = kb["observation"]
    ctx.factors         = kb["factors"]
    ctx.immediate_steps = kb["immediate"]
    ctx.inspect_steps   = kb["inspect"]
    ctx.correct_steps   = kb["correct"]
    ctx.verify_steps    = kb["verify"]
    ctx.monitor_steps   = kb["monitor"]
    ctx.summary         = kb["summary"]

    # For REVIEW cases, prepend explicit human-verification steps
    if ctx.display_decision == "REVIEW" and predicted_class != "normal":
        ctx.immediate_steps = [
            "Do NOT automatically reject — human confirmation is required",
            "Route this unit to the manual inspection station",
            "Re-inspect under controlled lighting against a reference image",
        ] + ctx.immediate_steps

    # ── Synthetic process context (deterministic for this filename) ───────────
    if prod_df is not None:
        import numpy as np
        rng = np.random.default_rng(seed)

        # Pick a row matching the predicted defect class for demo relevance
        pool = prod_df[prod_df["true_defect_label"] == predicted_class]
        if len(pool) == 0:
            pool = prod_df
        row = pool.iloc[int(rng.integers(0, len(pool)))]

        ctx.station_id        = str(row["station_id"])
        ctx.batch_id          = str(row["batch_id"])
        ctx.shift             = str(row["shift"])
        ctx.product_variant   = str(row["product_variant"])
        ctx.cycle_time_s      = float(row["cycle_time_s"])
        ctx.downtime_min      = float(row["downtime_min"])
        ctx.wip               = int(row["wip"])
        ctx.changeover_min    = float(row["changeover_min"])
        ctx.temperature_c     = float(row["temperature_c"])
        ctx.pressure_bar      = float(row["pressure_bar"])
        ctx.machine_speed_rpm = int(row["machine_speed_rpm"])
        ctx.vibration_mms     = float(row["vibration_mms"])
        ctx.humidity_pct      = float(row["humidity_pct"])

    # ── Bottleneck ────────────────────────────────────────────────────────────
    if station_metrics is not None and len(station_metrics) > 0:
        ctx.station_metrics_df = station_metrics
        bn = station_metrics.iloc[0]
        ctx.bn_station         = str(bn["station_id"])
        ctx.bn_cycle_s         = float(bn["avg_cycle_time_s"])
        ctx.bn_defect_pct      = float(bn["defect_rate_pct"])
        ctx.bn_downtime_min    = float(bn["total_downtime_min"])
        ctx.bn_changeover_min  = float(bn["total_changeover_min"])
        ctx.bn_utilisation_pct = float(bn["utilisation_pct"])
        if len(station_metrics) > 1:
            ctx.bn_avg_other_s = float(station_metrics.iloc[1:]["avg_cycle_time_s"].mean())
        else:
            ctx.bn_avg_other_s = ctx.bn_cycle_s

    # ── Economics ─────────────────────────────────────────────────────────────
    if econ_summary:
        cur = econ_summary.get("currency", "USD")
        ctx.currency            = cur
        ctx.defect_rate_pct     = float(econ_summary.get("defect_rate_pct", 0))
        ctx.total_scrap         = float(econ_summary.get("total_scrap_cost", 0))
        ctx.total_rework        = float(econ_summary.get("total_rework_cost", 0))
        ctx.total_downtime_loss = float(econ_summary.get("total_downtime_loss", 0))
        ctx.total_quality_loss  = ctx.total_scrap + ctx.total_rework + ctx.total_downtime_loss
        ctx.avg_contribution    = float(econ_summary.get("avg_contribution_unit", 0))
        total_units             = int(econ_summary.get("total_units", 1))
        total_rev               = total_units * 45.0   # synthetic selling price
        ctx.pct_of_revenue      = (ctx.total_quality_loss / total_rev * 100) if total_rev else 0

        # Per-unit cost for this prediction
        ctx.unit_scrap_cost  = 25.0  # synthetic assumption
        ctx.unit_rework_cost = 10.0  # synthetic assumption

    # ── Final recommendation (dynamic) ───────────────────────────────────────
    ctx.evidence_points, ctx.final_recommendation = _build_recommendation(ctx)

    return ctx


def _assess(predicted_class, confidence, decision, margin, low_margin, tensor) -> dict:
    """Compute prediction reliability and whether display_decision should be downgraded."""
    reasons      = []
    domain_notes = []
    display_dec  = decision
    from app.config import settings

    if confidence < settings.confidence_threshold:
        reasons.append(
            f"Confidence {confidence:.1%} is below the automatic threshold "
            f"({settings.confidence_threshold:.0%})"
        )
    if low_margin:
        reasons.append(
            f"Narrow prediction margin ({margin:.1%}) — model is not clearly "
            "favouring one class over another"
        )
    if predicted_class == "hole" and confidence < 0.90:
        reasons.append(
            "Intentional circular geometry (designed holes, slots) in a manufactured "
            "component can resemble the learned hole-defect pattern — "
            "verify against the approved product drawing"
        )
    # Domain / input distribution check
    if tensor is not None:
        try:
            arr = tensor.squeeze(0).cpu().numpy()
            pmean = float(arr.mean()); pstd = float(arr.std())
            if pmean > 0.6 or pmean < -1.5:
                domain_notes.append(
                    f"Image brightness (normalised mean {pmean:.2f}) is outside the "
                    "typical range of training images — possible domain mismatch"
                )
                reasons.append(domain_notes[-1])
            elif pstd < 0.05:
                domain_notes.append("Very low image contrast — may not match training distribution")
                reasons.append(domain_notes[-1])
        except Exception:
            pass

    # Downgrade REJECT → REVIEW if unreliable
    not_reliable = (confidence < settings.confidence_threshold or low_margin
                    or len(domain_notes) > 0)
    if not_reliable and decision == "REJECT":
        display_dec = "REVIEW"

    return {"reasons": reasons, "domain_notes": domain_notes, "display_decision": display_dec}


def _build_recommendation(ctx: InspectionContext) -> tuple[list[str], str]:
    """Generate evidence points and final recommendation sentence."""
    evidence = []
    dec = ctx.display_decision
    cls = ctx.predicted_class

    if dec == "ACCEPT":
        return [], "No defect detected. Continue production and maintain standard monitoring."

    if dec == "REVIEW":
        evidence.append(
            f"Model confidence {ctx.confidence:.1%} is below the automatic threshold"
        )
        if ctx.low_margin:
            evidence.append(f"Prediction margin is narrow ({ctx.prediction_margin:.1%})")
        return evidence, (
            "Manual verification is required before any disposition decision. "
            "Route this unit to the inspection station and compare against a reference image."
        )

    # REJECT — build data-driven evidence
    evidence.append(f"Model confidence: {ctx.confidence:.1%}")

    if ctx.bn_station:
        if ctx.bn_defect_pct > ctx.defect_rate_pct:
            evidence.append(
                f"Station {ctx.bn_station} shows {ctx.bn_defect_pct:.1f}% defect rate vs "
                f"{ctx.defect_rate_pct:.1f}% overall (synthetic data)"
            )
        evidence.append(
            f"Station {ctx.bn_station} has highest cycle time ({ctx.bn_cycle_s:.1f}s) "
            f"— possible process constraint"
        )

    if ctx.vibration_mms and cls in ("crack", "scratch"):
        evidence.append(
            f"Observed vibration {ctx.vibration_mms:.3f} mm/s "
            f"(associated with {cls} defects in synthetic data)"
        )
    if ctx.pressure_bar and cls == "hole":
        evidence.append(
            f"Observed pressure {ctx.pressure_bar:.2f} bar "
            "(associated with hole defects in synthetic data)"
        )
    if ctx.humidity_pct and cls == "rust":
        evidence.append(
            f"Observed humidity {ctx.humidity_pct:.1f}% "
            "(associated with rust defects in synthetic data)"
        )

    kb = DEFECT_KNOWLEDGE.get(cls, {})
    rec = (
        f"Quarantine this unit. {kb.get('summary', '')} "
        f"Verify correction with a batch run and monitor {cls} rate by station."
    )
    return evidence, rec
