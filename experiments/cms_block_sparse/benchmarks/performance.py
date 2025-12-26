"""
Performance Benchmarks for CMS Block-Sparse Layers

This module provides utilities for measuring computational performance
of the block-sparse implementation vs dense baselines:

1. Forward pass latency
2. Backward pass latency
3. Full training step time
4. Memory footprint

Target speedups (from benchmarks.md):
- Forward: >= 1.3x at 50% density, >= 1.5x at 25% density
- End-to-end: >= 1.2x tokens/sec at 50% density
- Memory: ~50% weight reduction at 50% density
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


@dataclass
class PerformanceResults:
    """Container for performance benchmark results."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    trials: int

    # Optional metadata
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    density: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.mean_ms:.3f}ms (+/- {self.std_ms:.3f}ms)"

    def speedup_vs(self, baseline: "PerformanceResults") -> float:
        """Compute speedup ratio vs baseline."""
        if self.mean_ms == 0:
            return float('inf')
        return baseline.mean_ms / self.mean_ms


def benchmark_forward(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 10,
    trials: int = 100,
) -> PerformanceResults:
    """
    Benchmark forward pass latency.

    Uses CUDA events for accurate GPU timing.

    Args:
        layer: Layer or model to benchmark
        input_tensor: Input tensor to pass through layer
        warmup: Number of warmup iterations
        trials: Number of timed iterations

    Returns:
        PerformanceResults with timing statistics

    Example:
        >>> layer = BlockSparseLinear(256, 256, density=0.5)
        >>> x = torch.randn(16, 128, 256, device='cuda')
        >>> result = benchmark_forward(layer, x)
        >>> print(f"Forward: {result}")
    """
    layer.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(input_tensor)

    torch.cuda.synchronize()

    # Timed trials using CUDA events
    times = []

    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = layer(input_tensor)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)

    return PerformanceResults(
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        trials=trials,
        batch_size=input_tensor.shape[0],
        seq_length=input_tensor.shape[1] if input_tensor.dim() > 2 else None,
    )


def benchmark_backward(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 10,
    trials: int = 100,
) -> PerformanceResults:
    """
    Benchmark backward pass latency.

    Measures time for loss.backward() including gradient computation.

    Args:
        layer: Layer or model to benchmark
        input_tensor: Input tensor (requires_grad=True)
        warmup: Number of warmup iterations
        trials: Number of timed iterations

    Returns:
        PerformanceResults with timing statistics

    Note:
        Backward may show 10-30% overhead due to atomic contention
        in scatter-add operations (expected per benchmarks.md).
    """
    layer.train()

    # Ensure input requires grad
    input_tensor = input_tensor.requires_grad_(True)

    # Warmup
    for _ in range(warmup):
        output = layer(input_tensor)
        if isinstance(output, dict):
            loss = output.get("logits", output.get("loss", list(output.values())[0]))
            if loss.dim() > 0:
                loss = loss.sum()
        else:
            loss = output.sum()
        loss.backward()
        layer.zero_grad()

    torch.cuda.synchronize()

    # Timed trials
    times = []

    for _ in range(trials):
        output = layer(input_tensor)
        if isinstance(output, dict):
            loss = output.get("logits", output.get("loss", list(output.values())[0]))
            if loss.dim() > 0:
                loss = loss.sum()
        else:
            loss = output.sum()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        layer.zero_grad()

    times = np.array(times)

    return PerformanceResults(
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        trials=trials,
        batch_size=input_tensor.shape[0],
        seq_length=input_tensor.shape[1] if input_tensor.dim() > 2 else None,
    )


def benchmark_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    warmup: int = 5,
    trials: int = 20,
    optimizer_type: str = "adam",
) -> PerformanceResults:
    """
    Benchmark full training step including optimizer update.

    Measures complete training loop: forward, backward, optimizer.step().

    Args:
        model: Model to train
        optimizer: Optimizer (Adam, DeepNestedOptimizer, etc.)
        batch: Training batch with "input_ids" and "labels"
        warmup: Number of warmup iterations
        trials: Number of timed iterations
        optimizer_type: "adam", "deep_nested", or "continuum"

    Returns:
        PerformanceResults with timing statistics

    Example:
        >>> from titans_core.opt import DeepNestedOptimizer
        >>> optimizer = DeepNestedOptimizer(model, base_lr=1e-4)
        >>> batch = {"input_ids": ids, "labels": labels}
        >>> result = benchmark_step(model, optimizer, batch, optimizer_type="deep_nested")
        >>> print(f"Step: {result}")
    """
    model.train()
    is_nested = optimizer_type in ["deep_nested", "continuum"]

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if is_nested:
            optimizer.step(loss.item())
        else:
            optimizer.step()

    torch.cuda.synchronize()

    # Timed trials
    times = []

    for _ in range(trials):
        start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if is_nested:
            optimizer.step(loss.item())
        else:
            optimizer.step()

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms

    times = np.array(times)

    return PerformanceResults(
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        trials=trials,
        batch_size=input_ids.shape[0],
        seq_length=input_ids.shape[1],
    )


@dataclass
class MemoryResults:
    """Container for memory benchmark results."""

    peak_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float

    # Model-specific
    weight_storage_mb: float
    index_overhead_mb: float
    score_storage_mb: float

    density: Optional[float] = None

    def __str__(self) -> str:
        s = f"Peak: {self.peak_memory_gb:.3f} GB"
        s += f"\n  Weights: {self.weight_storage_mb:.1f} MB"
        s += f"\n  Indices: {self.index_overhead_mb:.1f} MB"
        s += f"\n  Scores: {self.score_storage_mb:.1f} MB"
        return s

    def reduction_vs(self, baseline: "MemoryResults") -> float:
        """Compute memory reduction ratio vs baseline."""
        if baseline.weight_storage_mb == 0:
            return 0.0
        return 1.0 - (self.weight_storage_mb / baseline.weight_storage_mb)


def benchmark_memory(
    model: nn.Module,
    batch: Optional[Dict[str, torch.Tensor]] = None,
    run_forward: bool = True,
) -> MemoryResults:
    """
    Measure memory footprint of model.

    Records peak memory during forward pass and estimates storage
    requirements for weights, indices, and scores.

    Args:
        model: Model to measure
        batch: Optional batch for forward pass measurement
        run_forward: Whether to run forward pass for peak measurement

    Returns:
        MemoryResults with memory statistics

    Memory Expectations (from benchmarks.md at 50% density):
        - Weight storage: ~50% reduction
        - Index overhead: ~6% of weight storage
        - Score storage: ~1% of weight storage

    Example:
        >>> result = benchmark_memory(model, batch)
        >>> print(f"Memory: {result}")
        >>> print(f"Weight reduction: {result.reduction_vs(baseline):.1%}")
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Compute weight storage
    weight_bytes = 0
    index_bytes = 0
    score_bytes = 0

    for name, module in model.named_modules():
        # Check for block-sparse layers
        if hasattr(module, 'values'):
            # Block-ELL format: values tensor holds active weights
            weight_bytes += module.values.numel() * 4  # float32

        if hasattr(module, 'col_indices'):
            # Column indices for each block
            index_bytes += module.col_indices.numel() * 4  # int32

        if hasattr(module, 'block_scores'):
            # Score EMAs for each block
            score_bytes += module.block_scores.numel() * 4  # float32

        # Regular layers
        if hasattr(module, 'weight') and module.weight is not None:
            if not hasattr(module, 'values'):  # Not already counted as sparse
                weight_bytes += module.weight.numel() * module.weight.element_size()

    # Run forward pass to measure peak
    if run_forward and batch is not None:
        model.train()
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated()
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()

    return MemoryResults(
        peak_memory_gb=peak_memory / 1e9,
        allocated_memory_gb=allocated_memory / 1e9,
        reserved_memory_gb=reserved_memory / 1e9,
        weight_storage_mb=weight_bytes / 1e6,
        index_overhead_mb=index_bytes / 1e6,
        score_storage_mb=score_bytes / 1e6,
    )


def benchmark_throughput(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    max_steps: int = 100,
    optimizer_type: str = "adam",
) -> float:
    """
    Measure training throughput in tokens per second.

    Args:
        model: Model to train
        dataloader: DataLoader providing batches
        optimizer: Optimizer for training
        max_steps: Number of steps to measure
        optimizer_type: "adam", "deep_nested", or "continuum"

    Returns:
        Tokens per second (float)

    Target (from benchmarks.md at 50% density):
        >= 1.2x tokens/sec improvement over dense
    """
    model.train()
    is_nested = optimizer_type in ["deep_nested", "continuum"]

    total_tokens = 0
    start_time = time.perf_counter()

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if is_nested:
            optimizer.step(loss.item())
        else:
            optimizer.step()

        total_tokens += input_ids.numel()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return total_tokens / elapsed


def run_performance_suite(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    optimizer_type: str = "adam",
) -> Dict[str, Any]:
    """
    Run complete performance benchmark suite.

    Args:
        model: Model to benchmark
        optimizer: Optimizer for training
        batch: Sample batch for benchmarks
        optimizer_type: Type of optimizer

    Returns:
        Dict with all benchmark results
    """
    results = {}

    # Forward benchmark
    print("Benchmarking forward pass...")
    input_tensor = batch["input_ids"]
    if hasattr(model, 'get_input_embeddings'):
        # Use embeddings as input for internal layer benchmarks
        embed = model.get_input_embeddings()
        input_tensor = embed(input_tensor)

    results["forward"] = benchmark_forward(model, batch["input_ids"])
    print(f"  Forward: {results['forward']}")

    # Backward benchmark
    print("Benchmarking backward pass...")
    results["backward"] = benchmark_backward(model, batch["input_ids"])
    print(f"  Backward: {results['backward']}")

    # Step benchmark
    print("Benchmarking training step...")
    results["step"] = benchmark_step(
        model, optimizer, batch,
        optimizer_type=optimizer_type
    )
    print(f"  Step: {results['step']}")

    # Memory benchmark
    print("Benchmarking memory...")
    results["memory"] = benchmark_memory(model, batch)
    print(f"  Memory: {results['memory']}")

    return results
