"""
Unit tests for the config dataclasses.

Verifies that every dataclass is frozen, exposes the expected defaults,
and (for ExtractionConfig) can build a MediaPipe options object.
"""

import logging
import os
import sys

import pytest  # type: ignore[import-untyped]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.config import (
    DEFAULT_COMPARISON_CONFIG,
    DEFAULT_EXTRACTION_CONFIG,
    DEFAULT_FEEDBACK_CONFIG,
    DEFAULT_PREPROCESSOR_CONFIG,
    ComparisonConfig,
    ExtractionConfig,
    FeedbackConfig,
    PreprocessorConfig,
)

logger = logging.getLogger(__name__)


def test_comparison_config_defaults_and_frozen():
    """ComparisonConfig defaults should match expected values and be immutable."""
    cfg = ComparisonConfig()

    logger.info("=== ComparisonConfig ===")
    logger.info("  weight_skeleton: %s", cfg.weight_skeleton)
    assert abs(cfg.weight_skeleton + cfg.weight_trajectory + cfg.weight_mask - 1.0) < 1e-6
    assert abs(cfg.weight_angles + cfg.weight_cog - 1.0) < 1e-6
    assert "left_elbow" in cfg.joint_tolerances
    assert len(cfg.joint_angles) == len(cfg.joint_tolerances)
    with pytest.raises(Exception):
        cfg.weight_skeleton = 0.1
    logger.info("  PASSED\n")


def test_comparison_config_singleton_is_equal_instance():
    """DEFAULT_COMPARISON_CONFIG should equal a fresh ComparisonConfig()."""
    logger.info("=== ComparisonConfig Singleton ===")
    assert DEFAULT_COMPARISON_CONFIG == ComparisonConfig()
    logger.info("  PASSED\n")


def test_extraction_config_defaults_and_options():
    """ExtractionConfig should build valid MediaPipe options."""
    cfg = ExtractionConfig()
    options = cfg.create_landmarker_options()

    logger.info("=== ExtractionConfig ===")
    logger.info("  target_fps: %s, options: %s", cfg.target_fps, type(options).__name__)
    assert cfg.target_mask_size == (256, 256)
    assert cfg.norm_center == (128, 128)
    assert options is not None
    assert cfg == DEFAULT_EXTRACTION_CONFIG
    with pytest.raises(Exception):
        cfg.target_fps = 120.0
    logger.info("  PASSED\n")


def test_feedback_config_thresholds():
    """FeedbackConfig default thresholds should be ordered correctly."""
    cfg = FeedbackConfig()

    logger.info("=== FeedbackConfig ===")
    logger.info("  warn: %s, praise: %s", cfg.joint_warn_threshold, cfg.praise_threshold)
    assert cfg.joint_warn_threshold < cfg.praise_threshold
    assert cfg.energy_low_threshold < 1.0 < cfg.energy_high_threshold
    assert cfg == DEFAULT_FEEDBACK_CONFIG
    with pytest.raises(Exception):
        cfg.praise_threshold = 50.0
    logger.info("  PASSED\n")


def test_preprocessor_config_defaults():
    """PreprocessorConfig should expose sensible audio/motion defaults."""
    cfg = PreprocessorConfig()

    logger.info("=== PreprocessorConfig ===")
    logger.info("  sample_rate: %s, min_active: %s", cfg.audio_sample_rate, cfg.min_active_duration)
    assert cfg.audio_sample_rate > 0
    assert cfg.min_active_duration >= 1
    assert 0.0 < cfg.motion_threshold_ratio < 1.0
    assert cfg == DEFAULT_PREPROCESSOR_CONFIG
    with pytest.raises(Exception):
        cfg.audio_sample_rate = 16000
    logger.info("  PASSED\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_comparison_config_defaults_and_frozen()
    test_comparison_config_singleton_is_equal_instance()
    test_extraction_config_defaults_and_options()
    test_feedback_config_thresholds()
    test_preprocessor_config_defaults()
    logger.info("All tests passed!")
