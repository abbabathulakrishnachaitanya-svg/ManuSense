"""
app/recommendations/engine.py
───────────────────────────────
Evidence-based process improvement recommendations.

All recommendations are advisory/simulated — no machine control,
no PLC integration, no live actuator commands.

Placeholder — recommendation logic must be grounded in real root-cause
and bottleneck findings derived from the organizer dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Priority(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


@dataclass
class Recommendation:
    """
    A single advisory recommendation surfaced by the engine.

    Attributes:
        title:       Short human-readable title.
        description: Detailed explanation with supporting evidence.
        priority:    HIGH / MEDIUM / LOW.
        category:    E.g. "process", "maintenance", "quality", "throughput".
        evidence:    Quantitative evidence backing the recommendation
                     (e.g. correlation score, defect rate, cost estimate).
        actions:     Ordered list of concrete advisory steps (strings).
    """
    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    category: str = "general"
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)


class RecommendationEngine:
    """
    Aggregates findings from vision, analytics, and economics modules
    and produces a ranked list of advisory recommendations.

    Usage (once implemented):
        engine = RecommendationEngine()
        recs = engine.generate(root_cause_data, bottleneck_data, profitability_data)
    """

    def generate(
        self,
        root_cause_data: Any,
        bottleneck_data: Any,
        profitability_data: Any,
    ) -> list[Recommendation]:
        """
        Produce a ranked list of recommendations from analysis outputs.

        TODO:
            - Define rule-based or scoring logic driven by:
                * Top root-cause factors (from analytics.root_cause).
                * Bottleneck stations (from analytics.bottleneck).
                * Cost impact (from economics.profitability).
            - Decide whether recommendations are purely rule-based or
              use a lightweight ranking model.

        Args:
            root_cause_data:    Output of root_cause.rank_root_causes().
            bottleneck_data:    Output of bottleneck.identify_bottlenecks().
            profitability_data: Output of profitability.run_profitability_simulation().

        Returns:
            List of :class:`Recommendation` objects sorted by priority
            (HIGH first).

        Raises:
            NotImplementedError: Until analysis modules are implemented.
        """
        raise NotImplementedError("generate — pending analytics module implementation")

    def to_dataframe(self, recommendations: list[Recommendation]):
        """
        Convert the recommendation list to a display-ready DataFrame.

        Args:
            recommendations: Output of :meth:`generate`.

        Returns:
            pandas DataFrame with columns: priority, category, title,
            description, actions.
        """
        import pandas as pd

        return pd.DataFrame([
            {
                "priority":    r.priority.value,
                "category":    r.category,
                "title":       r.title,
                "description": r.description,
                "actions":     " | ".join(r.actions),
            }
            for r in recommendations
        ])
