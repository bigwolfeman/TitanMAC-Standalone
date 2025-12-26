"""
CMS Block-Sparse Experiment Datasets

This module provides datasets for benchmarking the CMS Dynamic Block Sparse
implementation across various forgetting scenarios.

Datasets:
- MathDataset: Standard arithmetic problems
- ModularMathDataset: Modular arithmetic for ID-semantic forgetting tests
- SyntheticNLPDataset: Generated linguistic patterns for cross-domain tests
"""

from .math_datasets import (
    MathDataset,
    ModularMathDataset,
    ContextSwitchedMathDataset,
    MathTokenizer,
    MATH_VOCAB,
    EXTENDED_MATH_VOCAB,
    ID_TO_TOKEN,
)

from .nlp_datasets import (
    SyntheticNLPDataset,
    SVODataset,
    OVSDataset,
    NLPTokenizer,
    NLP_VOCAB,
    COMBINED_VOCAB,
)

__all__ = [
    # Math datasets
    "MathDataset",
    "ModularMathDataset",
    "ContextSwitchedMathDataset",
    "MathTokenizer",
    "MATH_VOCAB",
    "EXTENDED_MATH_VOCAB",
    "ID_TO_TOKEN",
    # NLP datasets
    "SyntheticNLPDataset",
    "SVODataset",
    "OVSDataset",
    "NLPTokenizer",
    "NLP_VOCAB",
    "COMBINED_VOCAB",
]
