#!/usr/bin/env python3
"""
Minimal training script demonstrating CMSBlockLinear usage.

This example shows:
- Creating a simple model with CMSBlockLinear layers
- Training loop with gradient accumulation
- Scoring, score_step, and topology_step integration
- Monitoring topology statistics

Usage:
    # Basic training
    python examples/train_block_sparse.py --steps 500

    # With higher density
    python examples/train_block_sparse.py --density 0.75 --steps 500

    # On CUDA with larger model
    python examples/train_block_sparse.py --device cuda --d-model 256

Date: 2025-12-27
Branch: 001-cms-block-sparse
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from titans_core.layers.block_sparse import CMSBlockLinear


@dataclass
class BlockSparseModelConfig:
    """Configuration for the block-sparse demo model."""

    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    tile_size: int = 16
    density: float = 0.5
    num_layers: int = 2


class BlockSparseModel(nn.Module):
    """Simple MLP using CMSBlockLinear layers for demonstration.

    Architecture:
        input -> [BlockSparse -> ReLU] x num_layers -> output

    This model demonstrates how to integrate CMSBlockLinear into a
    standard PyTorch model. The block-sparse layers can be used as
    drop-in replacements for nn.Linear.
    """

    def __init__(self, config: BlockSparseModelConfig):
        super().__init__()
        self.config = config

        # Build layers
        self.layers = nn.ModuleList()

        # First layer: input -> hidden
        self.layers.append(
            CMSBlockLinear(
                in_features=config.input_dim,
                out_features=config.hidden_dim,
                tile_size=config.tile_size,
                density=config.density,
                bias=True,
            )
        )

        # Hidden layers
        for _ in range(config.num_layers - 1):
            self.layers.append(
                CMSBlockLinear(
                    in_features=config.hidden_dim,
                    out_features=config.hidden_dim,
                    tile_size=config.tile_size,
                    density=config.density,
                    bias=True,
                )
            )

        # Output projection
        self.output = nn.Linear(config.hidden_dim, config.output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through block-sparse layers."""
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)

    def get_block_sparse_layers(self) -> List[CMSBlockLinear]:
        """Return all CMSBlockLinear layers for topology management."""
        return [layer for layer in self.layers if isinstance(layer, CMSBlockLinear)]


def create_synthetic_data(
    num_samples: int,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> TensorDataset:
    """Create synthetic regression data for training.

    Generates random input-output pairs where outputs are a linear
    transformation of inputs plus noise. This provides a simple
    regression task for demonstrating block-sparse training.
    """
    torch.manual_seed(42)

    X = torch.randn(num_samples, input_dim, device=device)
    # Create a target transformation
    W_true = torch.randn(input_dim, output_dim, device=device) * 0.1
    noise = torch.randn(num_samples, output_dim, device=device) * 0.01
    y = X @ W_true + noise

    return TensorDataset(X, y)


def accumulate_scores_for_model(model: BlockSparseModel) -> None:
    """Accumulate gradient scores for all block-sparse layers.

    Call this after each backward() pass to update importance scores.
    """
    for layer in model.get_block_sparse_layers():
        layer.accumulate_scores()


def score_step_for_model(model: BlockSparseModel) -> None:
    """Run score_step (Level 1 update) for all block-sparse layers.

    Call this every ~10 training steps to normalize accumulators
    and increment block ages.
    """
    for layer in model.get_block_sparse_layers():
        layer.score_step()


def topology_step_for_model(
    model: BlockSparseModel,
    global_step: int,
) -> Dict[str, float]:
    """Run topology_step (Level 2 update) for all block-sparse layers.

    Call this every ~100 training steps to make topology decisions
    (prune/grow blocks). Returns aggregated statistics.

    Args:
        model: The model containing block-sparse layers
        global_step: Current training step for deterministic RNG

    Returns:
        Dict with aggregated topology statistics
    """
    total_swaps = 0
    total_blocks = 0

    for layer in model.get_block_sparse_layers():
        result = layer.topology_step(global_step=global_step)
        total_swaps += result.num_swaps
        total_blocks += layer.R * layer.K

    return {
        "total_swaps": total_swaps,
        "total_blocks": total_blocks,
        "swap_rate": total_swaps / total_blocks if total_blocks > 0 else 0.0,
    }


def get_model_topology_stats(model: BlockSparseModel) -> Dict[str, float]:
    """Get aggregated topology stats from all block-sparse layers.

    Returns averaged statistics across all CMSBlockLinear layers.
    """
    layers = model.get_block_sparse_layers()
    if not layers:
        return {}

    total_density = 0.0
    total_entropy = 0.0
    total_score = 0.0
    total_age = 0.0
    total_blocks = 0

    for layer in layers:
        stats = layer.get_topology_stats()
        total_density += stats.density
        total_entropy += stats.column_entropy
        total_score += stats.avg_block_score
        total_age += stats.avg_block_age
        total_blocks += stats.num_blocks

    n_layers = len(layers)
    return {
        "avg_density": total_density / n_layers,
        "avg_entropy": total_entropy / n_layers,
        "avg_block_score": total_score / n_layers,
        "avg_block_age": total_age / n_layers,
        "total_blocks": total_blocks,
    }


def train(
    model: BlockSparseModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    log_interval: int = 10,
    score_interval: int = 10,  # Level 1: normalize accumulators
    topology_interval: int = 100,  # Level 2: topology decisions
) -> List[Dict]:
    """Training loop with integrated topology management.

    Demonstrates the CMS update schedule:
    - Every step: accumulate_scores()
    - Every score_interval steps: score_step()
    - Every topology_interval steps: topology_step()

    Args:
        model: BlockSparseModel to train
        dataloader: DataLoader for training data
        optimizer: PyTorch optimizer
        device: Device for training
        max_steps: Maximum training steps
        log_interval: Steps between logging
        score_interval: Steps between score_step calls
        topology_interval: Steps between topology_step calls

    Returns:
        List of per-step metrics dicts
    """
    model.train()
    criterion = nn.MSELoss()

    step = 0
    total_loss = 0.0
    start_time = time.time()
    metrics_history = []

    data_iter = iter(dataloader)

    while step < max_steps:
        # Get batch (cycle through data)
        try:
            X, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, y = next(data_iter)

        X, y = X.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Backward pass
        loss.backward()

        # Accumulate scores after backward (every step)
        accumulate_scores_for_model(model)

        # Optimizer step
        optimizer.step()

        step += 1
        total_loss += loss.item()

        # Score step (Level 1) - every score_interval steps
        if step % score_interval == 0:
            score_step_for_model(model)

        # Topology step (Level 2) - every topology_interval steps
        if step % topology_interval == 0:
            topology_result = topology_step_for_model(model, global_step=step)

            # Log topology update
            print(
                f"\n  [Topology Step {step}] "
                f"Swaps: {topology_result['total_swaps']}/{topology_result['total_blocks']} "
                f"({topology_result['swap_rate']:.2%})"
            )

            # Get current stats
            stats = get_model_topology_stats(model)
            print(
                f"    Entropy: {stats['avg_entropy']:.3f}, "
                f"Avg Score: {stats['avg_block_score']:.4f}, "
                f"Avg Age: {stats['avg_block_age']:.1f}"
            )

        # Logging
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            samples_per_sec = log_interval * dataloader.batch_size / elapsed

            print(f"Step {step:5d} | Loss: {avg_loss:.6f} | " f"Samples/s: {samples_per_sec:.0f}")

            metrics_history.append(
                {
                    "step": step,
                    "loss": avg_loss,
                    "samples_per_sec": samples_per_sec,
                }
            )

            total_loss = 0.0
            start_time = time.time()

    return metrics_history


def main():
    parser = argparse.ArgumentParser(
        description="Block-sparse training example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training args
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    # Model args
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--d-hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of block-sparse layers")

    # Block-sparse args
    parser.add_argument("--tile-size", type=int, default=16, help="Block tile size")
    parser.add_argument("--density", type=float, default=0.5, help="Sparsity density")

    # Topology update schedule
    parser.add_argument(
        "--score-interval", type=int, default=10, help="Steps between score_step (Level 1)"
    )
    parser.add_argument(
        "--topology-interval", type=int, default=100, help="Steps between topology_step (Level 2)"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Block-Sparse Training Example")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Ensure dimensions are divisible by tile_size
    d_model = (args.d_model // args.tile_size) * args.tile_size
    d_hidden = (args.d_hidden // args.tile_size) * args.tile_size
    output_dim = 64  # Fixed for synthetic task

    if d_model != args.d_model:
        print(
            f"Note: Adjusted d_model from {args.d_model} to {d_model} for tile_size={args.tile_size}"
        )
    if d_hidden != args.d_hidden:
        print(
            f"Note: Adjusted d_hidden from {args.d_hidden} to {d_hidden} for tile_size={args.tile_size}"
        )

    # Create model config
    config = BlockSparseModelConfig(
        input_dim=d_model,
        hidden_dim=d_hidden,
        output_dim=output_dim,
        tile_size=args.tile_size,
        density=args.density,
        num_layers=args.n_layers,
    )

    print("\nModel config:")
    print(f"  Input: {config.input_dim}, Hidden: {config.hidden_dim}, Output: {config.output_dim}")
    print(
        f"  Layers: {config.num_layers}, Tile size: {config.tile_size}, Density: {config.density}"
    )

    # Create model
    model = BlockSparseModel(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    sparse_params = sum(layer.values.numel() for layer in model.get_block_sparse_layers())
    print(f"\nParameters: {total_params:,} total, {sparse_params:,} in block-sparse layers")

    # Show initial topology stats
    initial_stats = get_model_topology_stats(model)
    print("\nInitial topology stats:")
    print(f"  Density: {initial_stats['avg_density']:.3f}")
    print(f"  Column entropy: {initial_stats['avg_entropy']:.3f}")
    print(f"  Total blocks: {initial_stats['total_blocks']}")

    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = create_synthetic_data(
        num_samples=1000,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining for {args.steps} steps...")
    print(f"  Score step every {args.score_interval} steps (Level 1)")
    print(f"  Topology step every {args.topology_interval} steps (Level 2)")
    print("-" * 60)

    metrics = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_steps=args.steps,
        log_interval=args.log_interval,
        score_interval=args.score_interval,
        topology_interval=args.topology_interval,
    )

    # Final stats
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    final_stats = get_model_topology_stats(model)
    print("\nFinal topology stats:")
    print(f"  Density: {final_stats['avg_density']:.3f}")
    print(f"  Column entropy: {final_stats['avg_entropy']:.3f}")
    print(f"  Avg block score: {final_stats['avg_block_score']:.6f}")
    print(f"  Avg block age: {final_stats['avg_block_age']:.1f} topology steps")

    # Show swap rate history
    layers = model.get_block_sparse_layers()
    if layers:
        avg_swap_rate = sum(layer.get_avg_swap_rate() for layer in layers) / len(layers)
        print(f"  Avg swap rate: {avg_swap_rate:.2%}")

    # Save checkpoint
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "block_sparse_model.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "final_loss": metrics[-1]["loss"] if metrics else None,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()


# =============================================================================
# BENCHMARK INSTRUCTIONS
# =============================================================================
#
# T097: Forgetting Benchmark
# --------------------------
# To measure catastrophic forgetting with CMS block-sparse:
#
# 1. Train on math data:
#    python examples/train_math.py --steps 5000 --use-neural-memory
#    # Record final math accuracy
#
# 2. Continue training on NLP data:
#    # (Would need an NLP dataset - e.g., wikitext)
#    # Train for additional 5000 steps
#
# 3. Re-evaluate on math:
#    # Measure accuracy degradation
#
# Expected: CMS should reduce forgetting by preserving important sparse blocks
# during domain shift.
#
#
# T098: Speedup Benchmark
# -----------------------
# To measure throughput improvement sparse vs dense:
#
# 1. Create matched dense model (replace CMSBlockLinear with nn.Linear)
#
# 2. Warm-up and benchmark:
#    for _ in range(100):  # warmup
#        model(x)
#    torch.cuda.synchronize()
#    start = time.perf_counter()
#    for _ in range(1000):
#        model(x)
#    torch.cuda.synchronize()
#    samples_per_sec = 1000 * batch_size / (time.perf_counter() - start)
#
# 3. Compare samples/sec between sparse and dense
#
# Expected speedup depends on density and hardware:
# - density=0.5: ~1.5-2x speedup
# - density=0.25: ~2-4x speedup
# - Triton kernels required for best performance
#
#
# T099: Stability Validation
# --------------------------
# To validate training stability over 10K steps:
#
# 1. Run extended training:
#    python examples/train_block_sparse.py --steps 10000 --log-interval 100
#
# 2. Monitor for loss spikes:
#    - Track max(loss_t / loss_{t-1}) ratio
#    - Flag if ratio > 2.0 (loss spike)
#
# 3. Check topology stability:
#    - Swap rate should stabilize around 5-15%
#    - Column entropy should remain > 0.5
#
# Expected: No loss spikes > 2x with default hyperparameters.
#
# =============================================================================
