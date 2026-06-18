"""
Feedback configuration.

All thresholds for rule-based feedback generation: warning triggers
and praise triggers for each scoring component.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackConfig:
    """Configuration for feedback generation thresholds."""

    joint_warn_threshold: float = 60.0
    direction_warn_threshold: float = 60.0
    mask_warn_threshold: float = 60.0
    praise_threshold: float = 75.0

    min_marker_gap: float = 1.0

    energy_low_threshold: float = 0.6
    energy_high_threshold: float = 1.4

    timing_warn_threshold: float = 0.67
    timing_praise_threshold: float = 0.60


DEFAULT_FEEDBACK_CONFIG = FeedbackConfig()
