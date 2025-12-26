"""
CMS Block-Sparse Experiment Configurations

This module provides predefined experiment configurations for the
CMS block-sparse benchmark suite.

Configurations cover the full difficulty gradient:
1. OOD_MATH_NLP: Math -> NLP (easy, different vocabularies)
2. ID_SEMANTIC_MODULAR: Standard -> Modular arithmetic (hard, same tokens)
3. ID_SYNTACTIC_GRAMMAR: SVO -> OVS patterns (medium, position-dependent)
4. ID_CONTEXT_MODE: MODE:STD -> MODE:MOD7 (hardest, single token routing)
"""

from .experiment_configs import (
    ExperimentConfig,
    OOD_MATH_NLP,
    ID_SEMANTIC_MODULAR,
    ID_SYNTACTIC_GRAMMAR,
    ID_CONTEXT_MODE,
    get_config,
    list_configs,
)

__all__ = [
    "ExperimentConfig",
    "OOD_MATH_NLP",
    "ID_SEMANTIC_MODULAR",
    "ID_SYNTACTIC_GRAMMAR",
    "ID_CONTEXT_MODE",
    "get_config",
    "list_configs",
]
