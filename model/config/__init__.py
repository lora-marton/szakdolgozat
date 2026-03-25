"""
Configuration subpackage for the dance comparison pipeline.

Re-exports all config classes and default instances so that
existing imports like `from model.config import DEFAULT_CONFIG` still work.
"""
from model.config.extraction_config import ExtractionConfig, DEFAULT_CONFIG
from model.config.comparison_config import ComparisonConfig, DEFAULT_COMPARISON_CONFIG
from model.config.preprocessor_config import PreprocessorConfig, DEFAULT_PREPROCESSOR_CONFIG

__all__ = [
    'ExtractionConfig',
    'ComparisonConfig',
    'PreprocessorConfig',
    'DEFAULT_CONFIG',
    'DEFAULT_COMPARISON_CONFIG',
    'DEFAULT_PREPROCESSOR_CONFIG',
]
