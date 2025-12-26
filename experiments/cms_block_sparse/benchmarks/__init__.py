"""
CMS Block-Sparse Benchmarks

This module provides benchmark utilities for validating the CMS Dynamic Block
Sparse implementation across correctness, performance, and forgetting metrics.

Benchmark Classes:
- ForgettingBenchmark: Base class for OOD forgetting evaluation
- IDForgettingBenchmark: In-distribution semantic forgetting tests
- Performance functions: benchmark_forward, benchmark_backward, benchmark_step, benchmark_memory
"""

from .forgetting import (
    ForgettingBenchmark,
    IDForgettingBenchmark,
    compute_pathway_overlap,
    compute_topology_divergence,
)

from .performance import (
    benchmark_forward,
    benchmark_backward,
    benchmark_step,
    benchmark_memory,
    PerformanceResults,
)

__all__ = [
    # Forgetting benchmarks
    "ForgettingBenchmark",
    "IDForgettingBenchmark",
    "compute_pathway_overlap",
    "compute_topology_divergence",
    # Performance benchmarks
    "benchmark_forward",
    "benchmark_backward",
    "benchmark_step",
    "benchmark_memory",
    "PerformanceResults",
]
