#!/usr/bin/env python3
"""
Phase 4-5: Forgetting Experiments for CMS Block-Sparse Validation

This script runs forgetting experiments across the full difficulty gradient:
1. OOD (Out-of-Distribution): Different vocabularies (easy)
2. ID-Semantic: Same tokens, different meanings (hard)
3. ID-Syntactic: Same tokens, different positions (medium)
4. ID-Context: Mode-switched by prefix (hardest)

Success Criteria (from benchmarks.md):
- Phase 4: >= 30% relative improvement vs dense baseline
- Phase 5 OOD: < 15% forgetting
- Phase 5 ID-Semantic: < 30% forgetting (Silver tier)
- Phase 5 ID-Context: < 50% forgetting (stretch goal)

Usage:
    # Run OOD forgetting experiment
    python run_forgetting.py --experiment ood_math_nlp

    # Run ID-semantic with modular arithmetic
    python run_forgetting.py --experiment id_semantic_modular

    # Run with specific optimizer
    python run_forgetting.py --experiment ood_math_nlp --optimizer deep_nested

    # Quick test
    python run_forgetting.py --experiment ood_math_nlp --quick
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC

# Local imports
from datasets import (
    MathDataset, ModularMathDataset, MathTokenizer,
    SyntheticNLPDataset, NLPTokenizer,
)
from datasets.nlp_datasets import SVODataset, OVSDataset
from datasets.math_datasets import ContextSwitchedMathDataset
from configs import get_config, ExperimentConfig
from benchmarks import (
    ForgettingBenchmark,
    IDForgettingBenchmark,
    compute_pathway_overlap,
    compute_topology_divergence,
    benchmark_step,
    benchmark_memory,
)
from benchmarks.forgetting import snapshot_topology


def create_model(config: ExperimentConfig, device: str) -> TitanMAC:
    """Create TitanMAC model from experiment config."""
    model_config = TitanMACConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_model * 4,
        max_seq_len=config.seq_length,
        window_size=64,
        n_persistent=8,
        dropout=0.1,
    )

    model = TitanMAC(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    return model.to(device)


def create_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
):
    """Create optimizer based on config."""
    if config.optimizer == "deep_nested":
        from titans_core.opt import DeepNestedOptimizer
        return DeepNestedOptimizer(
            model=model,
            base_lr=config.base_lr,
            meta_lr=config.meta_lr,
            cms_frequencies=[1, 10, 100],
            mode='simple',
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
        )
    elif config.optimizer == "continuum":
        from titans_core.opt import ContinuumOptimizer
        return ContinuumOptimizer(
            model=model,
            base_lr=config.base_lr,
            controller_lr=config.meta_lr,
            base_optim_kwargs={"weight_decay": config.weight_decay},
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )


def create_tokenizer(task_type: str, config: ExperimentConfig):
    """Create appropriate tokenizer for task type."""
    if task_type in ["mode_std", "mode_mod7"]:
        return MathTokenizer(extended=True)
    elif task_type in ["math", "modular"]:
        return MathTokenizer()
    else:  # nlp, svo, ovs
        return NLPTokenizer()


def create_dataset(
    task_type: str,
    tokenizer,
    config: ExperimentConfig,
    seed: int,
    is_task_a: bool = True,
):
    """Create dataset based on task type."""
    params = config.task_a_params if is_task_a else config.task_b_params

    if task_type == "math":
        return MathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            max_num=params.get("max_num", 99),
            include_large=params.get("include_large", True),
        )
    elif task_type == "modular":
        return ModularMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            modulus=params.get("modulus", 7),
            max_num=params.get("max_num", 99),
        )
    elif task_type == "nlp":
        return SyntheticNLPDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            pattern_types=params.get("pattern_types", None),
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
        return ContextSwitchedMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            mode="STD",
            modulus=params.get("modulus", 7),
            max_num=params.get("max_num", 99),
        )
    elif task_type == "mode_mod7":
        return ContextSwitchedMathDataset(
            tokenizer=tokenizer,
            seq_length=config.seq_length,
            seed=seed,
            mode="MOD7",
            modulus=params.get("modulus", 7),
            max_num=params.get("max_num", 99),
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_task(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    max_steps: int,
    log_interval: int = 100,
    task_name: str = "Task",
    optimizer_type: str = "adam",
) -> Dict[str, Any]:
    """
    Train model on a single task.

    Returns:
        Dict with training metrics
    """
    model.train()
    is_nested = optimizer_type in ["deep_nested", "continuum"]

    total_loss = 0.0
    step = 0
    start_time = time.time()

    losses = []
    step_times = []
    lr_multipliers = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        step_start = time.time()

        if is_nested:
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if is_nested:
            result = optimizer.step(loss.item())
            lr_mults = result.get('lr_multipliers', None)
            if lr_mults is not None and len(lr_mults) >= 1:
                lr_mult = lr_mults[0].item() if isinstance(lr_mults[0], torch.Tensor) else lr_mults[0]
                lr_multipliers.append(lr_mult)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        step_time = time.time() - step_start
        step_times.append(step_time)

        total_loss += loss.item()
        losses.append(loss.item())

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (log_interval * input_ids.numel()) / elapsed

            log_str = f"[{task_name}] Step {step:5d} | Loss: {avg_loss:.4f} | Tok/s: {tokens_per_sec:.0f}"
            if lr_multipliers:
                log_str += f" | LR_mult: {lr_multipliers[-1]:.4f}"
            print(log_str)

            total_loss = 0.0
            start_time = time.time()

        if step >= max_steps:
            break

    return {
        "final_loss": losses[-1] if losses else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "steps": step,
        "avg_step_time": sum(step_times) / len(step_times) if step_times else 0.0,
        "avg_lr_mult": sum(lr_multipliers) / len(lr_multipliers) if lr_multipliers else 1.0,
    }


def run_ood_forgetting(
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Run OOD (out-of-distribution) forgetting experiment.

    Task A and Task B have different vocabularies, making this the
    easiest forgetting test. Topology should trivially separate.

    Target: < 15% forgetting
    """
    print("\n" + "=" * 60)
    print(f"OOD Forgetting: {config.task_a_type} -> {config.task_b_type}")
    print("=" * 60)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model
    model = create_model(config, device)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create tokenizers (may be different for each task)
    tokenizer_a = create_tokenizer(config.task_a_type, config)
    tokenizer_b = create_tokenizer(config.task_b_type, config)

    # Task A dataset
    dataset_a = create_dataset(config.task_a_type, tokenizer_a, config, seed, is_task_a=True)
    dataloader_a = DataLoader(dataset_a, batch_size=config.batch_size, num_workers=0)

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
        optimizer_type=config.optimizer,
    )

    # Evaluate Task A and capture topology
    benchmark_a = ForgettingBenchmark(
        tokenizer=tokenizer_a,
        n_eval=config.n_eval,
        seed=config.eval_seed,
        device=device,
    )
    accuracy_a_before = benchmark_a.evaluate(model)
    topology_a = snapshot_topology(model)

    print(f"\n[After Task A] Accuracy A: {accuracy_a_before:.4f}")

    # Task B dataset
    dataset_b = create_dataset(config.task_b_type, tokenizer_b, config, seed + 1, is_task_a=False)
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
        optimizer_type=config.optimizer,
    )

    # Evaluate both tasks after Task B
    accuracy_a_after = benchmark_a.evaluate(model)
    topology_b = snapshot_topology(model)

    # Compute forgetting
    if accuracy_a_before > 0:
        forgetting_pct = 100 * (accuracy_a_before - accuracy_a_after) / accuracy_a_before
        retention_pct = 100 * accuracy_a_after / accuracy_a_before
    else:
        forgetting_pct = 0.0
        retention_pct = 100.0

    # Compute topology divergence (if sparse layers present)
    topology_div = None
    if topology_a and topology_b:
        # Average divergence across layers
        divergences = []
        for layer_name in topology_a:
            if layer_name in topology_b:
                div = compute_topology_divergence(topology_a[layer_name], topology_b[layer_name])
                divergences.append(div)
        if divergences:
            topology_div = sum(divergences) / len(divergences)

    print(f"\n[After Task B] Accuracy A: {accuracy_a_after:.4f}")
    print(f"[Result] Forgetting: {forgetting_pct:.1f}% | Retention: {retention_pct:.1f}%")
    if topology_div is not None:
        print(f"[Result] Topology Divergence: {topology_div:.3f}")

    # Check success criteria
    success = forgetting_pct <= config.target_forgetting
    tier = "PASS" if success else "FAIL"
    if forgetting_pct < 10:
        tier = "EXCELLENT"
    elif forgetting_pct < 15:
        tier = "GOOD"
    elif forgetting_pct < 20:
        tier = "ACCEPTABLE"

    print(f"[Result] Status: {tier} (target: <{config.target_forgetting}%)")

    return {
        "experiment": config.name,
        "experiment_type": config.experiment_type,
        "accuracy_a_before": accuracy_a_before,
        "accuracy_a_after": accuracy_a_after,
        "forgetting_pct": forgetting_pct,
        "retention_pct": retention_pct,
        "topology_divergence": topology_div,
        "success": success,
        "tier": tier,
        "train_a_metrics": train_a_metrics,
        "train_b_metrics": train_b_metrics,
        "config": config.to_dict(),
    }


def run_id_semantic_forgetting(
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Run ID-semantic forgetting experiment.

    Same inputs, different outputs. This is the hard test for
    block-sparse - can topology learn to separate identical inputs?

    Success Tiers:
    - < 15%: Gold
    - 15-30%: Silver
    - 30-50%: Bronze
    - > 50%: Fail
    """
    print("\n" + "=" * 60)
    print(f"ID-Semantic Forgetting: {config.task_a_type} -> {config.task_b_type}")
    print("=" * 60)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create model
    model = create_model(config, device)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # For ID-semantic, both tasks use the same tokenizer
    tokenizer = MathTokenizer()

    # Create ID forgetting benchmark (with shared inputs)
    id_benchmark = IDForgettingBenchmark(
        tokenizer=tokenizer,
        n_eval=config.n_eval,
        seed=config.eval_seed,
        modulus=config.task_b_params.get("modulus", 7),
        device=device,
    )

    # Task A dataset (standard arithmetic)
    dataset_a = create_dataset(config.task_a_type, tokenizer, config, seed, is_task_a=True)
    dataloader_a = DataLoader(dataset_a, batch_size=config.batch_size, num_workers=0)

    # Train Task A
    print("\n--- Training Task A (Standard Arithmetic) ---")
    train_a_metrics = train_task(
        model=model,
        dataloader=dataloader_a,
        optimizer=optimizer,
        device=torch.device(device),
        max_steps=config.steps_a,
        log_interval=config.log_interval,
        task_name="Task A",
        optimizer_type=config.optimizer,
    )

    # Evaluate standard arithmetic and capture topology
    standard_accuracy_before = id_benchmark.eval_standard(model)
    topology_a = snapshot_topology(model)

    print(f"\n[After Task A] Standard Accuracy: {standard_accuracy_before:.4f}")

    # Task B dataset (modular arithmetic)
    dataset_b = create_dataset(config.task_b_type, tokenizer, config, seed + 1, is_task_a=False)
    dataloader_b = DataLoader(dataset_b, batch_size=config.batch_size, num_workers=0)

    # Train Task B
    print("\n--- Training Task B (Modular Arithmetic) ---")
    train_b_metrics = train_task(
        model=model,
        dataloader=dataloader_b,
        optimizer=optimizer,
        device=torch.device(device),
        max_steps=config.steps_b,
        log_interval=config.log_interval,
        task_name="Task B",
        optimizer_type=config.optimizer,
    )

    # Evaluate both after Task B
    standard_accuracy_after = id_benchmark.eval_standard(model)
    modular_accuracy = id_benchmark.eval_modular(model)
    topology_b = snapshot_topology(model)

    # Compute forgetting
    if standard_accuracy_before > 0:
        forgetting_pct = 100 * (standard_accuracy_before - standard_accuracy_after) / standard_accuracy_before
        retention_pct = 100 * standard_accuracy_after / standard_accuracy_before
    else:
        forgetting_pct = 0.0
        retention_pct = 100.0

    # Compute topology divergence
    topology_div = None
    if topology_a and topology_b:
        divergences = []
        for layer_name in topology_a:
            if layer_name in topology_b:
                div = compute_topology_divergence(topology_a[layer_name], topology_b[layer_name])
                divergences.append(div)
        if divergences:
            topology_div = sum(divergences) / len(divergences)

    print(f"\n[After Task B] Standard Accuracy: {standard_accuracy_after:.4f}")
    print(f"[After Task B] Modular Accuracy: {modular_accuracy:.4f}")
    print(f"[Result] Forgetting: {forgetting_pct:.1f}% | Retention: {retention_pct:.1f}%")
    if topology_div is not None:
        print(f"[Result] Topology Divergence: {topology_div:.3f}")

    # Determine tier
    if forgetting_pct < 15:
        tier = "GOLD"
    elif forgetting_pct < 30:
        tier = "SILVER"
    elif forgetting_pct < 50:
        tier = "BRONZE"
    else:
        tier = "FAIL"

    success = forgetting_pct <= config.target_forgetting
    print(f"[Result] Tier: {tier} (target: <{config.target_forgetting}%)")

    return {
        "experiment": config.name,
        "experiment_type": config.experiment_type,
        "standard_accuracy_before": standard_accuracy_before,
        "standard_accuracy_after": standard_accuracy_after,
        "modular_accuracy": modular_accuracy,
        "forgetting_pct": forgetting_pct,
        "retention_pct": retention_pct,
        "topology_divergence": topology_div,
        "success": success,
        "tier": tier,
        "train_a_metrics": train_a_metrics,
        "train_b_metrics": train_b_metrics,
        "config": config.to_dict(),
    }


def run_experiment(
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Route to appropriate experiment based on type."""
    if config.experiment_type == "OOD":
        return run_ood_forgetting(config, device, seed)
    elif config.experiment_type == "ID-Semantic":
        return run_id_semantic_forgetting(config, device, seed)
    elif config.experiment_type == "ID-Syntactic":
        # ID-Syntactic uses similar structure to OOD but with grammar datasets
        return run_ood_forgetting(config, device, seed)
    elif config.experiment_type == "ID-Context":
        # ID-Context uses ID-semantic structure but with mode-switched datasets
        return run_id_semantic_forgetting(config, device, seed)
    else:
        raise ValueError(f"Unknown experiment type: {config.experiment_type}")


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str,
):
    """Save experiment results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4-5 forgetting experiments for CMS block-sparse"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="ood_math_nlp",
        choices=["ood_math_nlp", "id_semantic_modular", "id_syntactic_grammar", "id_context_mode"],
        help="Experiment to run",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "deep_nested", "continuum"],
        help="Override optimizer from config",
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
        help="Quick run with fewer steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/forgetting",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load config
    config = get_config(args.experiment)

    # Override optimizer if specified
    if args.optimizer:
        config.optimizer = args.optimizer

    # Reduce steps for quick mode
    if args.quick:
        config.steps_a = 500
        config.steps_b = 500
        config.log_interval = 50
        config.n_eval = 100

    print("=" * 60)
    print(f"CMS Block-Sparse Forgetting Experiment")
    print("=" * 60)
    print(f"Experiment: {config.name}")
    print(f"Type: {config.experiment_type}")
    print(f"Description: {config.description}")
    print(f"Device: {args.device}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Seed: {args.seed}")

    # Run experiment
    results = run_experiment(config, args.device, args.seed)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, args.experiment)

    # Print final summary
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    print(f"Forgetting: {results['forgetting_pct']:.1f}%")
    print(f"Tier: {results['tier']}")
    print(f"Success: {results['success']}")


if __name__ == "__main__":
    main()
