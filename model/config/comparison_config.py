"""
Comparison configuration.

All constants for dance comparison scoring: weights, decay parameters,
joint definitions, mask settings, and trajectory settings.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for dance comparison scoring."""

    weight_skeleton: float = 0.55
    weight_trajectory: float = 0.15
    weight_mask: float = 0.30

    weight_angles: float = 0.80
    weight_cog: float = 0.20

    angle_sigma: float = 35.0
    cog_sigma: float = 0.08

    dtw_joints: tuple = (
        11, 12,
        13, 14,
        15, 16,
        23, 24,
        25, 26,
        27, 28,
    )

    joint_tolerances: dict = field(default_factory=lambda: {
        'hips': 5.0,
        'knees': 15.0,
        'elbows': 20.0,
        'wrists': 25.0,
        'shoulders': 10.0,
        'ankles': 15.0,
    })

    joint_angles: tuple = (
        (11, 13, 15),
        (12, 14, 16),
        (23, 25, 27),
        (24, 26, 28),
        (13, 11, 23),
        (14, 12, 24),
    )

    cog_weights: dict = field(default_factory=lambda: {
        0: 0.08,
        11: 0.06,
        12: 0.06,
        13: 0.03,
        14: 0.03,
        15: 0.02,
        16: 0.02,
        23: 0.15,
        24: 0.15,
        25: 0.06,
        26: 0.06,
        27: 0.02,
        28: 0.02,
    })

    mask_binary_threshold: int = 128

    efd_harmonics: int = 6
    efd_contour_points: int = 200

    dtm_sigma: float = 25.0

    flow_winsize: int = 15

    weight_shape: float = 0.70
    weight_energy: float = 0.30

    trajectory_weight_direction: float = 0.75
    trajectory_weight_speed: float = 0.25


DEFAULT_COMPARISON_CONFIG = ComparisonConfig()
