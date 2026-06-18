"""
Preprocessor configuration.

All constants for preprocessing: audio synchronization and motion trimming.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration for preprocessing (audio sync + motion trimming)."""

    audio_sample_rate: int = 22050

    motion_threshold_ratio: float = 0.15
    min_active_duration: int = 10
    active_window_ratio: float = 0.7


DEFAULT_PREPROCESSOR_CONFIG = PreprocessorConfig()
