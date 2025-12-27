#!/usr/bin/env python3
"""Benchmarks for Block-ELL sparse kernels.

T056: Forward/backward timing comparison (dense vs sparse at 25%, 50%, 75% density)
T057: Memory usage benchmark - measure peak GPU memory during forward/backward

Usage:
    python benchmarks/bench_block_ell.py

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import argparse
import gc
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for benchmarks")


def create_sparse_setup(
    batch_size: int,
    in_features: int,
    out_features: int,
    tile_size: int,
    density: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Create sparse layer tensors for benchmarking.

    Returns:
        Tuple of (x, values, col_indices, bias, R, K, B)
    """
    R = out_features // tile_size
    C = in_features // tile_size
    K = max(1, int(C * density))

    # Create random topology
    col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
    for r in range(R):
        perm = torch.randperm(C, device=device)
        col_indices[r] = perm[:K].to(torch.int32)

    # Create tensors
    values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)

    return x, values, col_indices, bias, R, K, tile_size


def create_dense_setup(
    batch_size: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, nn.Linear]:
    """Create dense layer for benchmarking.

    Returns:
        Tuple of (x, dense_layer)
    """
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    layer = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)
    return x, layer


def warmup_cuda(iterations: int = 10):
    """Warm up CUDA for accurate timing."""
    # Simple operation to warm up
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(iterations):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()


def benchmark_forward(
    func,
    *args,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs,
) -> Dict[str, float]:
    """Benchmark forward pass.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
    }


def benchmark_forward_backward(
    forward_func,
    x: torch.Tensor,
    *args,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs,
) -> Dict[str, float]:
    """Benchmark forward + backward pass.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'
    """
    # Warmup
    for _ in range(warmup):
        output = forward_func(x, *args, **kwargs)
        loss = output.sum()
        loss.backward()
        if x.grad is not None:
            x.grad.zero_()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = forward_func(x, *args, **kwargs)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        # Zero gradients for next iteration
        if x.grad is not None:
            x.grad.zero_()

    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
    }


def measure_peak_memory(
    func,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """Measure peak GPU memory usage during function call.

    Returns:
        Dict with 'peak_mb', 'allocated_mb', 'reserved_mb'
    """
    # Clear cache and reset stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run function
    _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    # Get memory stats
    peak_bytes = torch.cuda.max_memory_allocated()
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()

    return {
        "peak_mb": peak_bytes / (1024 * 1024),
        "allocated_mb": allocated_bytes / (1024 * 1024),
        "reserved_mb": reserved_bytes / (1024 * 1024),
    }


def run_benchmark_suite(
    batch_size: int = 128,
    in_features: int = 640,
    out_features: int = 2560,
    tile_size: int = 16,
    densities: List[float] = [0.25, 0.5, 0.75, 1.0],
    iterations: int = 100,
    dtype: torch.dtype = torch.float32,
) -> Dict:
    """Run complete benchmark suite.

    Args:
        batch_size: Batch size for benchmarks
        in_features: Input dimension
        out_features: Output dimension
        tile_size: Block size
        densities: List of density levels to benchmark
        iterations: Number of iterations per benchmark
        dtype: Data type for tensors

    Returns:
        Dict with all benchmark results
    """
    from titans_core.kernels.block_ell_backward import block_ell_autograd, TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        print("Warning: Triton not available, skipping sparse benchmarks")
        return {}

    device = torch.device("cuda")
    results = {}

    print(f"\nBenchmark Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Dimensions: {in_features} -> {out_features}")
    print(f"  Tile size: {tile_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Dtype: {dtype}")
    print()

    # Warmup CUDA
    print("Warming up CUDA...")
    warmup_cuda()

    # Benchmark dense baseline
    print("Benchmarking dense baseline...")
    x_dense, dense_layer = create_dense_setup(batch_size, in_features, out_features, device, dtype)

    results["dense"] = {
        "forward": benchmark_forward(dense_layer, x_dense, iterations=iterations),
        "forward_backward": benchmark_forward_backward(dense_layer, x_dense, iterations=iterations),
        "memory": measure_peak_memory(dense_layer, x_dense),
    }
    print(f"  Dense forward: {results['dense']['forward']['mean_ms']:.3f} ms")
    print(f"  Dense forward+backward: {results['dense']['forward_backward']['mean_ms']:.3f} ms")
    print(f"  Dense peak memory: {results['dense']['memory']['peak_mb']:.2f} MB")

    # Benchmark each density level
    for density in densities:
        if density >= 1.0:
            # Skip full density (same as dense)
            continue

        print(f"\nBenchmarking sparse at {density*100:.0f}% density...")

        x, values, col_indices, bias, R, K, B = create_sparse_setup(
            batch_size, in_features, out_features, tile_size, density, device, dtype
        )

        def sparse_forward(x):
            return block_ell_autograd(x, values, col_indices, bias, in_features, R, K, B, use_triton=True)

        results[f"sparse_{density}"] = {
            "density": density,
            "forward": benchmark_forward(sparse_forward, x, iterations=iterations),
            "forward_backward": benchmark_forward_backward(lambda x: sparse_forward(x), x, iterations=iterations),
            "memory": measure_peak_memory(sparse_forward, x),
        }

        fwd_time = results[f"sparse_{density}"]["forward"]["mean_ms"]
        fwd_bwd_time = results[f"sparse_{density}"]["forward_backward"]["mean_ms"]
        peak_mem = results[f"sparse_{density}"]["memory"]["peak_mb"]

        dense_fwd = results["dense"]["forward"]["mean_ms"]
        dense_fwd_bwd = results["dense"]["forward_backward"]["mean_ms"]

        speedup_fwd = dense_fwd / fwd_time if fwd_time > 0 else 0
        speedup_fwd_bwd = dense_fwd_bwd / fwd_bwd_time if fwd_bwd_time > 0 else 0

        print(f"  Forward: {fwd_time:.3f} ms (speedup: {speedup_fwd:.2f}x)")
        print(f"  Forward+backward: {fwd_bwd_time:.3f} ms (speedup: {speedup_fwd_bwd:.2f}x)")
        print(f"  Peak memory: {peak_mem:.2f} MB")

    return results


def print_summary(results: Dict):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if not results:
        print("No results to display")
        return

    # Header
    print(f"{'Configuration':<20} {'Forward (ms)':<15} {'Fwd+Bwd (ms)':<15} {'Speedup':<12} {'Memory (MB)':<12}")
    print("-" * 70)

    # Dense baseline
    if "dense" in results:
        dense = results["dense"]
        print(f"{'Dense (baseline)':<20} {dense['forward']['mean_ms']:>10.3f}     {dense['forward_backward']['mean_ms']:>10.3f}     {'1.00x':<12} {dense['memory']['peak_mb']:>8.2f}")

    # Sparse configurations
    for key in sorted(results.keys()):
        if key.startswith("sparse_"):
            sparse = results[key]
            density = sparse["density"]

            fwd_speedup = results["dense"]["forward"]["mean_ms"] / sparse["forward"]["mean_ms"]
            fwd_bwd_speedup = results["dense"]["forward_backward"]["mean_ms"] / sparse["forward_backward"]["mean_ms"]

            print(f"{'Sparse ' + f'{density*100:.0f}%':<20} {sparse['forward']['mean_ms']:>10.3f}     {sparse['forward_backward']['mean_ms']:>10.3f}     {fwd_speedup:>4.2f}x       {sparse['memory']['peak_mb']:>8.2f}")

    print("=" * 70)


def main():
    """Main entry point for benchmarks."""
    parser = argparse.ArgumentParser(description="Block-ELL Sparse Kernel Benchmarks")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--in-features", type=int, default=640, help="Input features")
    parser.add_argument("--out-features", type=int, default=2560, help="Output features")
    parser.add_argument("--tile-size", type=int, default=16, help="Block tile size")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Density levels to benchmark")
    args = parser.parse_args()

    print("Block-ELL Sparse Kernel Benchmarks")
    print("=" * 50)

    results = run_benchmark_suite(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        tile_size=args.tile_size,
        densities=args.densities,
        iterations=args.iterations,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
