#!/usr/bin/env python3
"""
Phase 0: Dense Baselines for CMS Block-Sparse Validation

This script runs dense baseline experiments required for meaningful
comparison with block-sparse results. Must be run BEFORE sparse experiments.

Baselines from benchmarks.md:
1. Dense Task A Only - Train on math until convergence
2. Dense Task A->B - Sequential training without sparsity
3. Dense with EWC - State-of-art continual learning comparison
4. Static Sparse 50% - Random fixed topology (lower bound)

Expected Results (from literature):
- Dense model: 40-60% forgetting
- EWC: 25-40% forgetting
- Static sparse: ~same as dense or worse
- Dynamic block-sparse target: <30% forgetting

Usage:
    # Run all baselines
    python run_baselines.py --experiment ood_math_nlp

    # Run specific baseline
    python run_baselines.py --experiment ood_math_nlp --baseline dense_a_only

    # Quick test (fewer steps)
    python run_baselines.py --experiment ood_math_nlp --quick
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Install with: pip install wandb")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC

# Local imports
from datasets import (
    MathDataset, ModularMathDataset, MathTokenizer,
    SyntheticNLPDataset, NLPTokenizer,
    SVODataset, OVSDataset, ContextSwitchedMathDataset,
)
from configs import get_config, ExperimentConfig
from benchmarks import ForgettingBenchmark, IDForgettingBenchmark, benchmark_step, benchmark_memory


def create_model(config: ExperimentConfig, device: str) -> TitanMAC:
    """Create TitanMAC model from experiment config."""
    # Use d_ff from config if available, otherwise 4x d_model
    d_ff = getattr(config, 'd_ff', config.d_model * 4)

    model_config = TitanMACConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=d_ff,
        max_seq_len=config.seq_length,
        window_size=64,
        n_persistent=8,
        dropout=0.1,
    )

    model = TitanMAC(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  d_model={config.d_model}, d_ff={d_ff}, n_layers={config.n_layers}, n_heads={config.n_heads}")

    return model.to(device)


def create_dataset(
    task_type: str,
    tokenizer,
    config: ExperimentConfig,
    seed: int,
    task_params: Optional[Dict[str, Any]] = None,
):
    """Create dataset based on task type."""
    # Use provided params or empty dict
    params = task_params if task_params is not None else {}

    if task_type == "math":
        return MathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            **params,
        )
    elif task_type == "modular":
        return ModularMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            **params,
        )
    elif task_type == "nlp":
        return SyntheticNLPDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            **params,
        )
    elif task_type == "svo":
        return SVODataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
        )
    elif task_type == "ovs":
        return OVSDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
        )
    elif task_type == "mode_std":
        # Filter out 'mode' from params since we set it explicitly
        filtered_params = {k: v for k, v in params.items() if k != 'mode'}
        return ContextSwitchedMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            mode="STD",
            **filtered_params,
        )
    elif task_type == "mode_mod7":
        # Filter out 'mode' from params since we set it explicitly
        filtered_params = {k: v for k, v in params.items() if k != 'mode'}
        return ContextSwitchedMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            mode="MOD7",
            **filtered_params,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_task(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    log_interval: int = 100,
    task_name: str = "Task",
    global_step_offset: int = 0,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Train model on a single task.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        max_steps: Maximum training steps
        log_interval: Steps between logging
        task_name: Name for logging
        global_step_offset: Offset for global step (for sequential tasks)
        use_wandb: Whether to log to wandb

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0.0
    step = 0
    start_time = time.time()

    losses = []
    step_times = []
    grad_norms = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        step_start = time.time()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        global_step = global_step_offset + step
        step_time = time.time() - step_start
        step_times.append(step_time)
        grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        total_loss += loss.item()
        losses.append(loss.item())

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_grad_norm = sum(grad_norms[-log_interval:]) / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (log_interval * input_ids.numel()) / elapsed

            print(f"[{task_name}] Step {step:5d} | Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.3f} | Tok/s: {tokens_per_sec:.0f}")

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"train/{task_name.lower()}_loss": avg_loss,
                    f"train/{task_name.lower()}_grad_norm": avg_grad_norm,
                    f"perf/tokens_per_sec": tokens_per_sec,
                    f"perf/step_time_ms": (step_time * 1000),
                }, step=global_step)

            total_loss = 0.0
            start_time = time.time()

        if step >= max_steps:
            break

    # Log memory stats at end
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"perf/{task_name.lower()}_peak_memory_gb": peak_memory_gb}, step=global_step_offset + step)
        print(f"[{task_name}] Peak GPU memory: {peak_memory_gb:.2f} GB")

    return {
        "final_loss": losses[-1] if losses else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "steps": step,
        "avg_step_time": sum(step_times) / len(step_times) if step_times else 0.0,
        "avg_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
    }


def run_dense_task_a_only(
    config: ExperimentConfig,
    device: str,
    seed: int,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Baseline 1: Train on Task A only until convergence.

    Establishes accuracy_A_baseline for comparison.
    """
    print("\n" + "=" * 60)
    print("Baseline: Dense Task A Only")
    print("=" * 60)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model
    model = create_model(config, device)

    # Create tokenizer and dataset
    if config.task_a_type in ["mode_std", "mode_mod7"]:
        tokenizer = MathTokenizer(extended=True)
    elif config.task_a_type in ["math", "modular"]:
        tokenizer = MathTokenizer()
    elif config.task_a_type in ["svo", "ovs"]:
        tokenizer = NLPTokenizer()
    else:
        tokenizer = NLPTokenizer()

    dataset = create_dataset(config.task_a_type, tokenizer, config, seed, config.task_a_params)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
    )

    # Train
    train_metrics = train_task(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device(device),
        max_steps=config.steps_a,
        log_interval=config.log_interval,
        task_name="Task A",
        use_wandb=use_wandb,
    )

    # Evaluate
    benchmark = ForgettingBenchmark(
        tokenizer=tokenizer,
        n_eval=config.n_eval,
        seed=config.eval_seed,
        device=device,
        task_type=config.task_a_type,
    )
    accuracy = benchmark.evaluate(model)

    print(f"\n[Result] Task A Accuracy: {accuracy:.4f}")

    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "forgetting/accuracy_a": accuracy,
        }, step=config.steps_a)

    return {
        "baseline": "dense_task_a_only",
        "accuracy_a": accuracy,
        "train_metrics": train_metrics,
    }


def run_dense_sequential(
    config: ExperimentConfig,
    device: str,
    seed: int,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Baseline 2: Dense Task A -> Task B sequential training.

    Measures catastrophic forgetting without any mitigation.
    """
    print("\n" + "=" * 60)
    print("Baseline: Dense Task A -> B (Sequential)")
    print("=" * 60)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model
    model = create_model(config, device)

    # Create tokenizers
    if config.task_a_type in ["mode_std", "mode_mod7"]:
        tokenizer_a = MathTokenizer(extended=True)
    elif config.task_a_type in ["math", "modular"]:
        tokenizer_a = MathTokenizer()
    elif config.task_a_type in ["svo", "ovs"]:
        tokenizer_a = NLPTokenizer()
    else:
        tokenizer_a = NLPTokenizer()

    if config.task_b_type in ["mode_std", "mode_mod7"]:
        tokenizer_b = MathTokenizer(extended=True)
    elif config.task_b_type in ["nlp", "svo", "ovs"]:
        tokenizer_b = NLPTokenizer()
    else:
        tokenizer_b = MathTokenizer()

    # Task A dataset
    dataset_a = create_dataset(config.task_a_type, tokenizer_a, config, seed, config.task_a_params)
    dataloader_a = DataLoader(dataset_a, batch_size=config.batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
    )

    # Train Task A
    print("\n--- Training Task A ---")
    train_a_metrics = train_task(
        model=model,
        dataloader=dataloader_a,
        optimizer=optimizer,
        device=torch.device(device),
        max_steps=config.steps_a,
        log_interval=config.log_interval,
        task_name="Task A",
        global_step_offset=0,
        use_wandb=use_wandb,
    )

    # Evaluate Task A before Task B
    benchmark_a = ForgettingBenchmark(
        tokenizer=tokenizer_a,
        n_eval=config.n_eval,
        seed=config.eval_seed,
        device=device,
        task_type=config.task_a_type,
    )
    accuracy_a_before = benchmark_a.evaluate(model)
    print(f"\n[After Task A] Accuracy A: {accuracy_a_before:.4f}")

    # For ID experiments, also track standard vs modular accuracy
    id_benchmark = None
    standard_acc_before = None
    modular_acc_before = None
    if config.experiment_type.startswith("ID"):
        # Determine appropriate modulus based on experiment type
        modulus = config.task_b_params.get("modulus", 7)
        id_benchmark = IDForgettingBenchmark(
            tokenizer=tokenizer_a,
            n_eval=config.n_eval,
            seed=config.eval_seed,
            modulus=modulus,
            device=device,
            experiment_type=config.experiment_type,
        )
        standard_acc_before = id_benchmark.eval_standard(model)
        modular_acc_before = id_benchmark.eval_modular(model)
        print(f"[After Task A] Standard Accuracy: {standard_acc_before:.4f}")
        print(f"[After Task A] Modular Accuracy: {modular_acc_before:.4f}")

    # Log accuracy to wandb
    if use_wandb and WANDB_AVAILABLE:
        log_dict = {"forgetting/accuracy_a_after_task_a": accuracy_a_before}
        if standard_acc_before is not None:
            log_dict["forgetting/standard_accuracy_before"] = standard_acc_before
            log_dict["forgetting/modular_accuracy_before"] = modular_acc_before
        wandb.log(log_dict, step=config.steps_a)

    # Task B dataset
    dataset_b = create_dataset(config.task_b_type, tokenizer_b, config, seed + 1, config.task_b_params)
    dataloader_b = DataLoader(dataset_b, batch_size=config.batch_size, num_workers=0)

    # Train Task B
    print("\n--- Training Task B ---")
    train_b_metrics = train_task(
        model=model,
        dataloader=dataloader_b,
        optimizer=optimizer,
        device=torch.device(device),
        max_steps=config.steps_b,
        log_interval=config.log_interval,
        task_name="Task B",
        global_step_offset=config.steps_a,
        use_wandb=use_wandb,
    )

    # Evaluate both tasks after Task B
    accuracy_a_after = benchmark_a.evaluate(model)

    # Compute forgetting
    if accuracy_a_before > 0:
        forgetting_pct = 100 * (accuracy_a_before - accuracy_a_after) / accuracy_a_before
        retention_pct = 100 * accuracy_a_after / accuracy_a_before
    else:
        forgetting_pct = 0.0
        retention_pct = 100.0

    print(f"\n[After Task B] Accuracy A: {accuracy_a_after:.4f}")
    print(f"[Result] Forgetting: {forgetting_pct:.1f}% | Retention: {retention_pct:.1f}%")

    # For ID experiments, evaluate standard vs modular accuracy after Task B
    standard_acc_after = None
    modular_acc_after = None
    id_forgetting_pct = None
    if id_benchmark is not None:
        standard_acc_after = id_benchmark.eval_standard(model)
        modular_acc_after = id_benchmark.eval_modular(model)
        print(f"[After Task B] Standard Accuracy: {standard_acc_after:.4f}")
        print(f"[After Task B] Modular Accuracy: {modular_acc_after:.4f}")

        # Compute ID-specific forgetting (standard accuracy degradation)
        if standard_acc_before > 0:
            id_forgetting_pct = 100 * (standard_acc_before - standard_acc_after) / standard_acc_before
            print(f"[Result] ID Forgetting (Standard): {id_forgetting_pct:.1f}%")

    # Log forgetting metrics to wandb
    if use_wandb and WANDB_AVAILABLE:
        log_dict = {
            "forgetting/accuracy_a_after_task_b": accuracy_a_after,
            "forgetting/forgetting_pct": forgetting_pct,
            "forgetting/retention_pct": retention_pct,
        }
        if standard_acc_after is not None:
            log_dict["forgetting/standard_accuracy_after"] = standard_acc_after
            log_dict["forgetting/modular_accuracy_after"] = modular_acc_after
        if id_forgetting_pct is not None:
            log_dict["forgetting/id_forgetting_pct"] = id_forgetting_pct
        wandb.log(log_dict, step=config.steps_a + config.steps_b)

    # Build results dict
    results = {
        "baseline": "dense_sequential",
        "accuracy_a_before": accuracy_a_before,
        "accuracy_a_after": accuracy_a_after,
        "forgetting_pct": forgetting_pct,
        "retention_pct": retention_pct,
        "train_a_metrics": train_a_metrics,
        "train_b_metrics": train_b_metrics,
    }

    # Add ID-specific metrics if available
    if standard_acc_before is not None:
        results["standard_accuracy_before"] = standard_acc_before
        results["standard_accuracy_after"] = standard_acc_after
        results["modular_accuracy_before"] = modular_acc_before
        results["modular_accuracy_after"] = modular_acc_after
        results["id_forgetting_pct"] = id_forgetting_pct

    return results


def run_static_sparse(
    config: ExperimentConfig,
    device: str,
    seed: int,
    density: float = 0.5,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Baseline 4: Static sparse with random fixed topology.

    Lower bound for sparse methods - no topology adaptation.
    """
    print("\n" + "=" * 60)
    print(f"Baseline: Static Sparse {int(density*100)}%")
    print("=" * 60)

    # Note: This requires BlockSparseLinear implementation
    # For now, simulate with masked dense layers

    print("[Warning] Static sparse not yet implemented - using dense as placeholder")

    # Fall back to dense sequential for now
    return run_dense_sequential(config, device, seed, use_wandb=use_wandb)


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str,
    baseline_name: str,
):
    """Save baseline results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{baseline_name}_{timestamp}.json"

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0 dense baselines for CMS block-sparse validation"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="ood_math_nlp",
        choices=["ood_math_nlp", "id_semantic_modular", "id_syntactic_grammar", "id_context_mode"],
        help="Experiment configuration to run baselines for",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="all",
        choices=["all", "dense_a_only", "dense_sequential", "static_sparse"],
        help="Which baseline to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer steps (for testing)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override steps for both Task A and B (overrides --quick)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baselines",
        help="Output directory for results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cms-block-sparse",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="wandb run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--swap-tasks",
        action="store_true",
        help="Swap Task A and Task B (train on B first, then A)",
    )

    args = parser.parse_args()

    # Load config
    config = get_config(args.experiment)

    # Reduce steps for quick mode
    if args.quick:
        config.steps_a = 500
        config.steps_b = 500
        config.log_interval = 50
        config.n_eval = 100

    # Override steps if specified
    if args.steps is not None:
        config.steps_a = args.steps
        config.steps_b = args.steps
        config.log_interval = max(50, args.steps // 10)

    # Swap tasks if requested (train on B first, then A)
    if args.swap_tasks:
        config.task_a_type, config.task_b_type = config.task_b_type, config.task_a_type
        config.task_a_params, config.task_b_params = config.task_b_params, config.task_a_params
        print("[SWAPPED] Task A is now NLP, Task B is now Math")

    print("=" * 60)
    print(f"CMS Block-Sparse Baselines - {config.name}")
    print("=" * 60)
    print(f"Config: {config.description}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {config.steps_a} (Task A) + {config.steps_b} (Task B)")
    print(f"Task A: {config.task_a_type}, Task B: {config.task_b_type}")
    print(f"Model: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.experiment}_{args.baseline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "experiment": args.experiment,
                "baseline": args.baseline,
                "seed": args.seed,
                "steps_a": config.steps_a,
                "steps_b": config.steps_b,
                "task_a_type": config.task_a_type,
                "task_b_type": config.task_b_type,
                "swapped_tasks": args.swap_tasks,
                "batch_size": config.batch_size,
                "d_model": config.d_model,
                "d_ff": getattr(config, 'd_ff', config.d_model * 4),
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "vocab_size": config.vocab_size,
                "base_lr": config.base_lr,
                "seq_length": config.seq_length,
            },
        )
        print(f"wandb initialized: {wandb.run.url}")
    elif args.wandb:
        print("[Warning] --wandb specified but wandb not available. Install with: pip install wandb")

    output_dir = Path(args.output_dir)
    all_results = {}

    # Run selected baselines
    if args.baseline in ["all", "dense_a_only"]:
        results = run_dense_task_a_only(config, args.device, args.seed, use_wandb=use_wandb)
        all_results["dense_a_only"] = results
        save_results(results, output_dir, args.experiment, "dense_a_only")

    if args.baseline in ["all", "dense_sequential"]:
        results = run_dense_sequential(config, args.device, args.seed, use_wandb=use_wandb)
        all_results["dense_sequential"] = results
        save_results(results, output_dir, args.experiment, "dense_sequential")

    if args.baseline in ["all", "static_sparse"]:
        results = run_static_sparse(config, args.device, args.seed, use_wandb=use_wandb)
        all_results["static_sparse"] = results
        save_results(results, output_dir, args.experiment, "static_sparse")

    # Print summary
    print("\n" + "=" * 60)
    print("Baseline Summary")
    print("=" * 60)

    for baseline, results in all_results.items():
        if "forgetting_pct" in results:
            print(f"{baseline}: Forgetting = {results['forgetting_pct']:.1f}%")
        elif "accuracy_a" in results:
            print(f"{baseline}: Accuracy A = {results['accuracy_a']:.4f}")

    # Finish wandb run
    if use_wandb:
        # Log final summary
        summary = {}
        for baseline, results in all_results.items():
            if "forgetting_pct" in results:
                summary[f"{baseline}/forgetting_pct"] = results["forgetting_pct"]
                summary[f"{baseline}/retention_pct"] = results["retention_pct"]
            if "accuracy_a" in results:
                summary[f"{baseline}/accuracy_a"] = results["accuracy_a"]
            if "accuracy_a_before" in results:
                summary[f"{baseline}/accuracy_a_before"] = results["accuracy_a_before"]
                summary[f"{baseline}/accuracy_a_after"] = results["accuracy_a_after"]
        wandb.log(summary)
        wandb.finish()
        print("\nwandb run finished.")


if __name__ == "__main__":
    main()
