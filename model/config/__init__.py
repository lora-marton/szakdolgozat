"""
Configuration subpackage for the dance comparison pipeline.

Re-exports all config classes and default instances so that
existing imports like `from model.config import DEFAULT_EXTRACTION_CONFIG` still work.
"""

from model.config.comparison_config import DEFAULT_COMPARISON_CONFIG, ComparisonConfig
from model.config.extraction_config import DEFAULT_EXTRACTION_CONFIG, ExtractionConfig
from model.config.feedback_config import DEFAULT_FEEDBACK_CONFIG, FeedbackConfig
from model.config.preprocessor_config import (
    DEFAULT_PREPROCESSOR_CONFIG,
    PreprocessorConfig,
)

__all__ = [
    "ExtractionConfig",
    "ComparisonConfig",
    "FeedbackConfig",
    "PreprocessorConfig",
    "DEFAULT_EXTRACTION_CONFIG",
    "DEFAULT_COMPARISON_CONFIG",
    "DEFAULT_FEEDBACK_CONFIG",
    "DEFAULT_PREPROCESSOR_CONFIG",
]
