"""
Unit tests for the config dataclasses.

Verifies that every dataclass is frozen, exposes the expected defaults,
and (for ExtractionConfig) can build a MediaPipe options object.
"""

import os
import sys

import pytest

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


def test_comparison_config_defaults_and_frozen():
    """ComparisonConfig defaults should match expected values and be immutable."""
    cfg = ComparisonConfig()

    print("=== ComparisonConfig ===")
    print(f"  weight_skeleton: {cfg.weight_skeleton}")
    assert abs(cfg.weight_skeleton + cfg.weight_trajectory + cfg.weight_mask - 1.0) < 1e-6
    assert abs(cfg.weight_angles + cfg.weight_cog - 1.0) < 1e-6
    assert "left_elbow" in cfg.joint_tolerances
    assert len(cfg.joint_angles) == len(cfg.joint_tolerances)
    with pytest.raises(Exception):
        cfg.weight_skeleton = 0.1
    print("  PASSED\n")


def test_comparison_config_singleton_is_equal_instance():
    """DEFAULT_COMPARISON_CONFIG should equal a fresh ComparisonConfig()."""
    print("=== ComparisonConfig Singleton ===")
    assert DEFAULT_COMPARISON_CONFIG == ComparisonConfig()
    print("  PASSED\n")


def test_extraction_config_defaults_and_options():
    """ExtractionConfig should build valid MediaPipe options."""
    cfg = ExtractionConfig()
    options = cfg.create_landmarker_options()

    print("=== ExtractionConfig ===")
    print(f"  target_fps: {cfg.target_fps}, options: {type(options).__name__}")
    assert cfg.target_mask_size == (256, 256)
    assert cfg.norm_center == (128, 128)
    assert options is not None
    with pytest.raises(Exception):
        cfg.target_fps = 120.0
    print("  PASSED\n")


def test_feedback_config_thresholds():
    """FeedbackConfig default thresholds should be ordered correctly."""
    cfg = FeedbackConfig()

    print("=== FeedbackConfig ===")
    print(f"  warn: {cfg.joint_warn_threshold}, praise: {cfg.praise_threshold}")
    assert cfg.joint_warn_threshold < cfg.praise_threshold
    assert cfg.energy_low_threshold < 1.0 < cfg.energy_high_threshold
    assert cfg == DEFAULT_FEEDBACK_CONFIG
    with pytest.raises(Exception):
        cfg.praise_threshold = 50.0
    print("  PASSED\n")


def test_preprocessor_config_defaults():
    """PreprocessorConfig should expose sensible audio/motion defaults."""
    cfg = PreprocessorConfig()

    print("=== PreprocessorConfig ===")
    print(f"  sample_rate: {cfg.audio_sample_rate}, min_active: {cfg.min_active_duration}")
    assert cfg.audio_sample_rate > 0
    assert cfg.min_active_duration >= 1
    assert 0.0 < cfg.motion_threshold_ratio < 1.0
    assert cfg == DEFAULT_PREPROCESSOR_CONFIG
    with pytest.raises(Exception):
        cfg.audio_sample_rate = 16000
    print("  PASSED\n")


if __name__ == "__main__":
    test_comparison_config_defaults_and_frozen()
    test_comparison_config_singleton_is_equal_instance()
    test_extraction_config_defaults_and_options()
    test_feedback_config_thresholds()
    test_preprocessor_config_defaults()
    print("All tests passed!")
