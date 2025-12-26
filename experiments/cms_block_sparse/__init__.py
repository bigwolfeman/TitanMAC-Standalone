"""
CMS Block-Sparse Experiments

This package contains the complete benchmark suite for validating the
CMS Dynamic Block Sparse Linear Layer implementation.

Modules:
- datasets: Math and NLP datasets for training/evaluation
- benchmarks: Forgetting and performance measurement utilities
- configs: Predefined experiment configurations

Scripts:
- run_baselines.py: Phase 0 dense baselines
- run_forgetting.py: Phase 4-5 forgetting experiments

Usage:
    # Run dense baselines first
    python experiments/001-cms-block-sparse/run_baselines.py --experiment ood_math_nlp

    # Then run forgetting experiments
    python experiments/001-cms-block-sparse/run_forgetting.py --experiment ood_math_nlp

Experiment Difficulty Gradient:
    1. OOD (Math -> NLP): Easy - different vocabularies
    2. ID-Syntactic (SVO -> OVS): Medium - position-dependent
    3. ID-Semantic (Standard -> Modular): Hard - same inputs, different outputs
    4. ID-Context (MODE:STD -> MODE:MOD7): Hardest - single token routing
"""

__version__ = "0.1.0"
