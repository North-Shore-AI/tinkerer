"""
Thinker orchestrates TDD-first training pipelines for CNS support models.
"""

from __future__ import annotations

from .config import PipelineConfig, load_pipeline_config
from .pipeline import ThinkerPipeline

__all__ = ["PipelineConfig", "ThinkerPipeline", "load_pipeline_config"]

__version__ = "0.1.0"
