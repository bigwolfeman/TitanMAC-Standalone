"""
Training script for TitanMAC model with DeepNestedOptimizer.

This is Experiment 3 of the A/B comparison:
- Experiment 1: TitanMAC with Muon+AdamW (architecture test)
- Experiment 2: MoE with DeepNestedOptimizer (optimizer test)
- Experiment 3 (this file): TitanMAC with DeepNestedOptimizer (both)

Hypothesis: The nested optimizer's learned momentum and LR scheduling may work
particularly well with TitanMAC's neural memory updates, since both involve
learned optimization dynamics.

Usage:
    python train_titanmac_nested.py --experiment_name titanmac_nested_exp1

Key differences from other experiments:
- TitanMAC architecture (sliding window attention + neural memory)
- DeepNestedOptimizer instead of Muon+AdamW
- AMP disabled (required for neural memory gradient updates)
- Uses SimplifiedMetaTrainer by default (memory efficient)
"""

import argparse
import time
import os
import math
import json
import torch
import torch.nn.functional as F
import logging
from collections import deque
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.titanmac_config import TitanMACModelConfig, TitanMACGPU24GBConfig, TitanMAC168MConfig, DebugTitanMACConfig
from configs.dataset_config import DataConfig
from models.titanmac_wrapper import TitanMACWrapper, create_titanmac_model

# Add TitanMAC path for nested optimizer imports
import sys
_titan_path = os.path.join(os.path.dirname(__file__), "111TitanMAC-Standalone")
if _titan_path not in sys.path:
    sys.path.insert(0, _titan_path)
from titans_core.opt import DeepNestedOptimizer, ContinuumOptimizer, group_titans_params
from training.evaluation import evaluate_model
from utils.helpers import set_seed
from utils.logger import setup_logging


logger = logging.getLogger(__name__)


def get_config(config_name: str) -> TitanMACModelConfig:
    """Get config by name."""
    configs = {
        "default": TitanMAC168MConfig,
        "24gb": TitanMACGPU24GBConfig,
        "168m": TitanMAC168MConfig,
        "debug": DebugTitanMACConfig,
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(configs.keys())}")
    return configs[config_name]()


def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    """Prepare train and validation datasets with caching."""
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    # Define what config parameters invalidate the cache
    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    # Try to load valid cache
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")

    # Rebuild cache
    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    # Load and split
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )

    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val

    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")

    # Tokenize and save
    print("Tokenizing train set...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    train_ds.save_to_disk(train_cache)

    print("Tokenizing validation set...")
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)
    val_ds.save_to_disk(val_cache)

    # Save cache info
    with open(info_path, "w") as f:
        json.dump(config_state, f, indent=2)
    print("Saved dataset cache info.")

    return train_ds, val_ds


def create_loss_fn(config):
    """
    Create a loss function compatible with meta_update interface.

    Returns a function that takes (model, batch) and returns total loss.
    """
    def loss_fn(model, batch):
        device = next(model.parameters()).device

        # Handle different batch formats
        if isinstance(batch, dict):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        logits, aux_loss = model(x, return_aux_loss=True)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )

        total_loss = ce_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        return total_loss

    return loss_fn


def train_titanmac_nested(
    config: TitanMACModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    nested_config: Optional[Dict[str, Any]] = None,
    resume_from: Optional[str] = None,
):
    """
    Train TitanMAC model with DeepNestedOptimizer.

    Args:
        config: TitanMAC model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Optional directory to save outputs
        experiment_name: Optional experiment name for logging
        nested_config: Optional config dict for nested optimizer
        resume_from: Optional path to checkpoint to resume from

    Returns:
        model, final_metrics, metrics_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default nested optimizer config
    if nested_config is None:
        nested_config = {
            'base_lr': 3e-4,
            'meta_lr': 1e-4,
            'k_unroll': 5,
            'use_unrolled': False,  # Use SimplifiedMetaTrainer by default
            'use_cms_updates': False,  # AdamW + learned LR multipliers (proven to work)
            'momentum_hidden_dim': 64,
            'controller_hidden_dim': 32,
            'mode': 'explicit',
            'meta_update_freq': config.eval_every,
            'weight_decay': config.weight_decay,
            'max_grad_norm': config.grad_clip,
        }

    print(f"\n[Nested Optimizer] Training TitanMAC model with DeepNestedOptimizer")
    print(f"  Base LR: {nested_config['base_lr']}")
    print(f"  Meta LR: {nested_config['meta_lr']}")
    print(f"  K-unroll: {nested_config['k_unroll']}")
    print(f"  Mode: {nested_config['mode']}")
    meta_mode = "UnrolledMetaTrainer" if nested_config.get('use_unrolled', False) else "SimplifiedMetaTrainer"
    print(f"  Meta-learning: {meta_mode}")

    # Enable TF32 for faster matmul on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  TF32 matmul: enabled")

    # Initialize model
    set_seed(42)
    model = create_titanmac_model(config)
    model = model.to(device)

    # Enable gradient checkpointing if configured
    if getattr(config, 'use_gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print("  Gradient checkpointing: enabled")

    # Apply torch.compile if speedy mode enabled
    # NOTE: With Triton kernel integration, torch.autograd.grad() is replaced
    # with manual backprop, so torch.compile should work now.
    use_torch_compile = nested_config.get('use_torch_compile', False)
    if use_torch_compile:
        print("  Applying torch.compile to model...")
        from torch._functorch import config as functorch_config
        functorch_config.donated_buffer = False
        torch.set_float32_matmul_precision('high')
        # Use mode='default' instead of 'reduce-overhead' to avoid CUDA graph
        # memory allocation issues with the Triton kernel's in-place updates
        model = torch.compile(model, mode='default', dynamic=False)
        print("  torch.compile: enabled (mode=default, TF32=high)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    core_params, embed_params = group_titans_params(model)
    core_numel = sum(p.numel() for p in core_params)
    embed_numel = sum(p.numel() for p in embed_params)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Core parameters: {core_numel:,}")
    print(f"  Embed parameters: {embed_numel:,}")

    # Create optimizer based on mode
    use_cms = nested_config.get('use_cms_updates', False)
    if use_cms:
        # Use ContinuumOptimizer with CMS enabled
        optimizer = ContinuumOptimizer(
            model=model,
            base_lr=nested_config['base_lr'],
            update_freq=50,  # Controller update frequency
            controller_lr=nested_config['meta_lr'],
            use_cms=True,
            base_optim_kwargs={'weight_decay': nested_config['weight_decay']},
        )
        print(f"  Optimizer: ContinuumOptimizer (CMS enabled)")
    else:
        # Use DeepNestedOptimizer (default)
        optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=nested_config['base_lr'],
            meta_lr=nested_config['meta_lr'],
            k_unroll=nested_config['k_unroll'],
            momentum_hidden_dim=nested_config['momentum_hidden_dim'],
            controller_hidden_dim=nested_config['controller_hidden_dim'],
            mode=nested_config['mode'],
            meta_update_freq=nested_config['meta_update_freq'],
            weight_decay=nested_config['weight_decay'],
            max_grad_norm=nested_config['max_grad_norm'],
        )
        print(f"  Optimizer: DeepNestedOptimizer")

    # Resume from checkpoint if provided
    start_step = 0
    resumed_metrics_history = None
    if resume_from and os.path.exists(resume_from):
        print(f"\n[Resume] Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Optimizer state restored")
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state: {e}")
        if 'step' in checkpoint:
            start_step = checkpoint['step']
            print(f"  Resuming from step {start_step}")
        if 'metrics_history' in checkpoint:
            resumed_metrics_history = checkpoint['metrics_history']
            print(f"  Metrics history restored ({len(resumed_metrics_history.get('steps', []))} entries)")
        print(f"  Model state restored")

    # Create loss function for meta-updates
    loss_fn = create_loss_fn(config)

    # Buffer for recent training batches (for k-step unrolling)
    k_unroll = nested_config['k_unroll']
    train_batch_buffer = deque(maxlen=k_unroll + 2)

    # AMP setup - enabled with Triton kernel since it doesn't use torch.autograd.grad
    use_amp = nested_config.get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  AMP (mixed precision): enabled")

    # Wandb initialization
    use_wandb = nested_config.get('use_wandb', False)
    log_every = nested_config.get('log_every', 10)
    if use_wandb:
        wandb_config = {
            # Model config
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'max_seq_len': config.max_seq_len,
            'vocab_size': config.vocab_size,
            'batch_size': config.batch_size,
            'max_steps': config.max_steps,
            'window_size': getattr(config, 'window_size', 512),
            'titans_variant': getattr(config, 'titans_variant', 'MAG'),
            'use_neural_memory': getattr(config, 'use_neural_memory', True),
            # Nested optimizer config
            **{f'nested_{k}': v for k, v in nested_config.items() if not k.startswith('wandb')},
            # Total params
            'total_params': total_params,
            'core_params': core_numel,
            'embed_params': embed_numel,
        }
        wandb.init(
            project=nested_config.get('wandb_project', 'titanmac-nested'),
            entity=nested_config.get('wandb_entity'),
            name=experiment_name,
            config=wandb_config,
            resume='allow' if resume_from else None,
        )
        print(f"  Wandb: enabled (project={nested_config.get('wandb_project')}, log_every={log_every})")

    # Reset peak memory stats for accurate tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training metrics tracking
    train_start_time = time.time()
    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_aux_losses': [],
        'val_accuracies': [],
        'val_perplexities': [],
        'elapsed_times': [],
        'learning_rates': [],
        'lr_multipliers_core': [],
        'lr_multipliers_embed': [],
        'meta_losses': [],
        # Memory saturation metrics
        'memory_alpha_t': [],  # Forget gate (0=retain, 1=forget)
        'memory_eta_t': [],    # Decay gate (momentum decay)
        'memory_grad_norm': [],  # Gradient norm before clipping
        'memory_param_norm': [],  # Memory MLP weight norm
        'momentum_norm': [],  # Momentum buffer norm
    }
    # Restore previous metrics history if resuming
    if resumed_metrics_history:
        for key in metrics_history:
            if key in resumed_metrics_history:
                metrics_history[key] = resumed_metrics_history[key].copy()

    # Training loop
    model.train()
    step = start_step
    desc = f"Training {experiment_name}" if experiment_name else "Training (TitanMAC+Nested)"
    pbar = tqdm(total=config.max_steps, desc=desc, initial=start_step)

    # Get a validation batch iterator for meta-updates
    val_iter = iter(val_loader)

    def get_val_batch():
        nonlocal val_iter
        try:
            return next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            return next(val_iter)

    while step < config.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x, y = x.to(device), y.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Store batch in buffer for meta-updates
            train_batch_buffer.append({'input_ids': x, 'labels': y})

            # Forward pass with optional AMP
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    loss = total_loss / config.gradient_accumulation_steps

                # Backward with scaler
                scaler.scale(loss).backward()

                # Optimizer step with scaler
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer.base_optimizer)
                    optimizer.step(loss_value=total_loss.item())
                    scaler.update()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step(loss_value=total_loss.item())

            # Per-step wandb logging (high frequency for baseline capture)
            if use_wandb and step % log_every == 0:
                with torch.no_grad():
                    current_loss = ce_loss.item()
                    mem_loss = aux_loss.item() if aux_loss is not None else 0.0
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    perplexity = math.exp(min(current_loss, 20))
                    lr_mults = optimizer.get_lr_multipliers()

                    # Get memory stats for surprise/gate tracking
                    memory_stats = model.get_memory_stats()

                    # Core training metrics
                    wandb_log = {
                        'train/loss': current_loss,
                        'train/memory_loss': mem_loss,
                        'train/accuracy': accuracy,
                        'train/perplexity': perplexity,
                        'train/total_loss': current_loss + mem_loss,
                        'optimizer/lr_mult_core': lr_mults[0].item(),
                        'optimizer/lr_mult_embed': lr_mults[1].item(),
                        'step': step,
                    }

                    # Memory/surprise metrics (critical for Professor Forcing baseline)
                    if memory_stats:
                        wandb_log.update({
                            'memory/alpha_t': memory_stats.get('alpha_t', 0.0),
                            'memory/eta_t': memory_stats.get('eta_t', 0.0),
                            'memory/grad_norm': memory_stats.get('grad_norm', 0.0),
                            'memory/surprise': memory_stats.get('surprise', 0.0),
                            'memory/param_norm': memory_stats.get('memory_param_norm', 0.0),
                            'memory/momentum_norm': memory_stats.get('momentum_norm', 0.0),
                            'memory/param_mean': memory_stats.get('memory_param_mean', 0.0),
                            'memory/param_std': memory_stats.get('memory_param_std', 0.0),
                        })

                    # GPU memory tracking
                    if torch.cuda.is_available():
                        wandb_log['system/gpu_memory_gb'] = torch.cuda.memory_allocated() / 1e9
                        wandb_log['system/gpu_memory_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9

                    wandb.log(wandb_log, step=step)

            # Console logging (less frequent)
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                    lr_mults = optimizer.get_lr_multipliers()

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'mem': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr_c': f'{lr_mults[0].item():.3f}',
                    'lr_e': f'{lr_mults[1].item():.3f}',
                })

            # Evaluation and meta-update
            if step % config.eval_every == 0 and step > 0:
                # Evaluation
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                lr_mults = optimizer.get_lr_multipliers()
                effective_lrs = optimizer.get_effective_lrs()

                # Meta-update using buffered train batches and a val batch
                # Skip for CMS mode - ContinuumOptimizer has its own internal controller update
                if not use_cms and nested_config['mode'] == 'explicit' and len(train_batch_buffer) >= k_unroll:
                    val_batch = get_val_batch()
                    if isinstance(val_batch, dict):
                        val_batch_dict = {
                            'input_ids': val_batch['input_ids'].to(device),
                            'labels': val_batch['labels'].to(device),
                        }
                    else:
                        vx, vy = val_batch[0], val_batch[-1]
                        val_batch_dict = {
                            'input_ids': vx.to(device),
                            'labels': vy.to(device),
                        }

                    train_batches = list(train_batch_buffer)[:k_unroll]

                    # Perform meta-update
                    # Note: use_unrolled=True is VERY memory intensive
                    # For TitanMAC, we use SimplifiedMetaTrainer by default
                    optimizer.meta_update(
                        val_batch=val_batch_dict,
                        train_batches=train_batches,
                        loss_fn=loss_fn,
                        use_unrolled=nested_config.get('use_unrolled', False),
                    )

                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_aux_losses'].append(eval_metrics['val_aux_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(effective_lrs[0])
                metrics_history['lr_multipliers_core'].append(lr_mults[0].item())
                metrics_history['lr_multipliers_embed'].append(lr_mults[1].item())
                meta_loss = optimizer.last_meta_loss
                if meta_loss is not None:
                    meta_loss = meta_loss.item() if hasattr(meta_loss, 'item') else float(meta_loss)
                else:
                    meta_loss = 0.0
                metrics_history['meta_losses'].append(meta_loss)

                # Memory saturation metrics
                memory_stats = model.get_memory_stats()
                if memory_stats:
                    metrics_history['memory_alpha_t'].append(memory_stats.get('alpha_t', 0.0))
                    metrics_history['memory_eta_t'].append(memory_stats.get('eta_t', 0.0))
                    metrics_history['memory_grad_norm'].append(memory_stats.get('grad_norm', 0.0))
                    metrics_history['memory_param_norm'].append(memory_stats.get('memory_param_norm', 0.0))
                    metrics_history['momentum_norm'].append(memory_stats.get('momentum_norm', 0.0))
                else:
                    # No neural memory - fill with zeros
                    metrics_history['memory_alpha_t'].append(0.0)
                    metrics_history['memory_eta_t'].append(0.0)
                    metrics_history['memory_grad_norm'].append(0.0)
                    metrics_history['memory_param_norm'].append(0.0)
                    metrics_history['momentum_norm'].append(0.0)

                # Build memory stats string for logging
                mem_stats_str = ""
                if memory_stats and 'alpha_t' in memory_stats:
                    mem_stats_str = f", α={memory_stats['alpha_t']:.3f}, η={memory_stats['eta_t']:.3f}"

                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Mem Loss: {eval_metrics['val_aux_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR mult [core: {lr_mults[0].item():.3f}, embed: {lr_mults[1].item():.3f}]"
                      f"{mem_stats_str}")

                # Wandb eval logging
                if use_wandb:
                    eval_wandb_log = {
                        'eval/val_loss': eval_metrics['val_loss'],
                        'eval/val_memory_loss': eval_metrics['val_aux_loss'],
                        'eval/val_accuracy': eval_metrics['val_accuracy'],
                        'eval/val_perplexity': eval_metrics['val_perplexity'],
                        'eval/elapsed_minutes': elapsed_time,
                        'optimizer/meta_loss': meta_loss,
                        'optimizer/effective_lr_core': effective_lrs[0],
                        'optimizer/effective_lr_embed': effective_lrs[1] if len(effective_lrs) > 1 else effective_lrs[0],
                    }
                    if memory_stats:
                        eval_wandb_log.update({
                            'eval/memory_alpha_t': memory_stats.get('alpha_t', 0.0),
                            'eval/memory_eta_t': memory_stats.get('eta_t', 0.0),
                            'eval/memory_grad_norm': memory_stats.get('grad_norm', 0.0),
                            'eval/memory_surprise': memory_stats.get('surprise', 0.0),
                        })
                    wandb.log(eval_wandb_log, step=step)

                model.train()

            step += 1
            if step % 20 == 0:
                pbar.update(20)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    elapsed_time = (time.time() - train_start_time) / 60
    lr_mults = optimizer.get_lr_multipliers()
    effective_lrs = optimizer.get_effective_lrs()

    metrics_history['steps'].append(step)
    metrics_history['val_losses'].append(final_eval['val_loss'])
    metrics_history['val_aux_losses'].append(final_eval['val_aux_loss'])
    metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
    metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
    metrics_history['elapsed_times'].append(elapsed_time)
    metrics_history['learning_rates'].append(effective_lrs[0])
    metrics_history['lr_multipliers_core'].append(lr_mults[0].item())
    metrics_history['lr_multipliers_embed'].append(lr_mults[1].item())
    meta_loss = optimizer.last_meta_loss
    if meta_loss is not None:
        meta_loss = meta_loss.item() if hasattr(meta_loss, 'item') else float(meta_loss)
    else:
        meta_loss = 0.0
    metrics_history['meta_losses'].append(meta_loss)

    # Final memory saturation metrics
    final_memory_stats = model.get_memory_stats()
    if final_memory_stats:
        metrics_history['memory_alpha_t'].append(final_memory_stats.get('alpha_t', 0.0))
        metrics_history['memory_eta_t'].append(final_memory_stats.get('eta_t', 0.0))
        metrics_history['memory_grad_norm'].append(final_memory_stats.get('grad_norm', 0.0))
        metrics_history['memory_param_norm'].append(final_memory_stats.get('memory_param_norm', 0.0))
        metrics_history['momentum_norm'].append(final_memory_stats.get('momentum_norm', 0.0))
    else:
        metrics_history['memory_alpha_t'].append(0.0)
        metrics_history['memory_eta_t'].append(0.0)
        metrics_history['memory_grad_norm'].append(0.0)
        metrics_history['memory_param_norm'].append(0.0)
        metrics_history['momentum_norm'].append(0.0)

    total_time = (time.time() - train_start_time) / 60

    # Track peak VRAM usage
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_vram_gb = 0.0

    # Add to final metrics
    final_eval['peak_vram_gb'] = peak_vram_gb

    print(f"\n[TitanMAC + Nested] Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Memory Loss: {final_eval['val_aux_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"   Final LR multipliers: [core: {lr_mults[0].item():.3f}, embed: {lr_mults[1].item():.3f}]")
    print(f"   Total time: {total_time:.1f} min")

    # Save checkpoint
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(config),
            'nested_config': nested_config,
            'metrics': final_eval,
            'metrics_history': metrics_history,
            'step': step,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save metrics as JSON (including peak VRAM)
        metrics_path = os.path.join(output_dir, "metrics.json")
        metrics_data = {
            'final_metrics': final_eval,
            'total_time_minutes': total_time,
            'peak_vram_gb': peak_vram_gb,
            'actual_steps': step,
            'history': metrics_history,
            'nested_config': nested_config,
            'model_type': 'TitanMAC',
            'optimizer_type': 'DeepNestedOptimizer',
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        # Plot training curves
        plot_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(metrics_history, plot_path)
        print(f"Saved training curves to {plot_path}")

    # Final wandb logging and cleanup
    if use_wandb:
        # Log final summary metrics
        wandb.summary.update({
            'final/val_loss': final_eval['val_loss'],
            'final/val_memory_loss': final_eval['val_aux_loss'],
            'final/val_accuracy': final_eval['val_accuracy'],
            'final/val_perplexity': final_eval['val_perplexity'],
            'final/peak_vram_gb': peak_vram_gb,
            'final/total_time_minutes': total_time,
            'final/total_steps': step,
        })
        # Log final memory stats
        final_memory_stats = model.get_memory_stats()
        if final_memory_stats:
            wandb.summary.update({
                'final/memory_alpha_t': final_memory_stats.get('alpha_t', 0.0),
                'final/memory_eta_t': final_memory_stats.get('eta_t', 0.0),
                'final/memory_param_norm': final_memory_stats.get('memory_param_norm', 0.0),
            })
        wandb.finish()
        print("Wandb run finished")

    return model, final_eval, metrics_history


def plot_training_curves(metrics_history, save_path):
    """Plot and save training curves including memory saturation metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    # Row 1: Core training metrics
    # Val Loss
    axes[0, 0].plot(metrics_history['steps'], metrics_history['val_losses'], 'b-', label='Val Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Val Accuracy
    axes[0, 1].plot(metrics_history['steps'], metrics_history['val_accuracies'], 'g-', label='Val Acc')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Val Perplexity
    axes[0, 2].plot(metrics_history['steps'], metrics_history['val_perplexities'], 'r-', label='Val PPL')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Perplexity')
    axes[0, 2].set_title('Validation Perplexity')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Row 2: Optimizer metrics
    # Memory Loss (aux loss)
    axes[1, 0].plot(metrics_history['steps'], metrics_history['val_aux_losses'], 'm-', label='Memory Loss')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Memory Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # LR Multipliers
    axes[1, 1].plot(metrics_history['steps'], metrics_history['lr_multipliers_core'], 'b-', label='Core LR mult')
    axes[1, 1].plot(metrics_history['steps'], metrics_history['lr_multipliers_embed'], 'r-', label='Embed LR mult')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('LR Multiplier')
    axes[1, 1].set_title('LR Multipliers')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Meta Loss
    axes[1, 2].plot(metrics_history['steps'], metrics_history['meta_losses'], 'c-', label='Meta Loss')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Meta-Learning Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Row 3: Memory saturation metrics
    # Gate values (α_t and η_t)
    if 'memory_alpha_t' in metrics_history and metrics_history['memory_alpha_t']:
        axes[2, 0].plot(metrics_history['steps'], metrics_history['memory_alpha_t'], 'r-', label='α (forget)', linewidth=2)
        axes[2, 0].plot(metrics_history['steps'], metrics_history['memory_eta_t'], 'b-', label='η (decay)', linewidth=2)
        axes[2, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Gate Value')
        axes[2, 0].set_title('Memory Gates (Saturation)')
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Add annotation for saturation interpretation
        final_alpha = metrics_history['memory_alpha_t'][-1] if metrics_history['memory_alpha_t'] else 0
        if final_alpha > 0.7:
            axes[2, 0].annotate('High forget rate\n(possible saturation)',
                               xy=(0.95, 0.95), xycoords='axes fraction',
                               ha='right', va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        axes[2, 0].text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Memory Gates (N/A)')

    # Memory norms
    if 'memory_param_norm' in metrics_history and metrics_history['memory_param_norm']:
        axes[2, 1].plot(metrics_history['steps'], metrics_history['memory_param_norm'], 'g-', label='Param norm', linewidth=2)
        axes[2, 1].plot(metrics_history['steps'], metrics_history['momentum_norm'], 'orange', label='Momentum norm', linewidth=2)
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Norm')
        axes[2, 1].set_title('Memory Weight/Momentum Norms')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    else:
        axes[2, 1].text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Memory Norms (N/A)')

    # Gradient norm
    if 'memory_grad_norm' in metrics_history and metrics_history['memory_grad_norm']:
        axes[2, 2].plot(metrics_history['steps'], metrics_history['memory_grad_norm'], 'purple', label='Grad norm', linewidth=2)
        axes[2, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Clip threshold')
        axes[2, 2].set_xlabel('Step')
        axes[2, 2].set_ylabel('Gradient Norm')
        axes[2, 2].set_title('Memory Gradient Norm')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
    else:
        axes[2, 2].text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Memory Gradient (N/A)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main entry point for TitanMAC + Nested Optimizer training."""
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting TitanMAC + Nested Optimizer training")

    print_system_info()
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train TitanMAC with DeepNestedOptimizer")
    parser.add_argument("--config", type=str, default="168m", help="Config name: 168m, 24gb, debug")
    parser.add_argument("--base_lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--meta_lr", type=float, default=1e-4, help="Meta learning rate")
    parser.add_argument("--k_unroll", type=int, default=5, help="K-step unrolling for meta-update")
    parser.add_argument("--momentum_hidden_dim", type=int, default=64, help="Hidden dimension for momentum network")
    parser.add_argument("--momentum_num_layers", type=int, default=2, help="Number of layers in momentum network")
    parser.add_argument("--controller_hidden_dim", type=int, default=32, help="Hidden dimension for controller network")
    parser.add_argument("--controller_num_layers", type=int, default=2, help="Number of layers in controller network")
    parser.add_argument("--use_unrolled", action="store_true",
                        help="Use full k-step unrolled meta-learning (WARNING: very memory intensive)")
    parser.add_argument("--use_cms", action="store_true",
                        help="Use ContinuumOptimizer with CMS (Continuum Memory System) instead of "
                             "DeepNestedOptimizer. CMS uses multi-frequency updates for core vs memory params.")
    parser.add_argument("--variant", type=str, default="MAG", choices=["MAC", "MAG", "MAL"],
                        help="TitanMAC variant to use")
    parser.add_argument("--eval_every", type=int, default=2000,
                        help="Evaluate every N steps (default: 2000)")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Number of eval batches per evaluation (default: 50)")
    parser.add_argument("--steps", "--max_steps", type=int, dest="max_steps", help="Override max_steps")
    parser.add_argument("--experiment_name", type=str, default="titanmac_nested", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--speedy", action="store_true",
                        help="Enable performance optimizations: torch.compile (with Triton kernel), "
                             "AMP (mixed precision), and TF32 matmul. Expect 2-3x speedup.")
    # Wandb logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging for comprehensive metric tracking")
    parser.add_argument("--wandb_project", type=str, default="titanmac-nested",
                        help="W&B project name (default: titanmac-nested)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name (optional)")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log metrics to wandb every N steps (default: 10)")
    args = parser.parse_args()

    # Get config by name
    config = get_config(args.config)
    print(f"Using {args.config} configuration")

    # Override config with args
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    if args.variant:
        config.titans_variant = args.variant

    # Override eval settings
    config.eval_every = args.eval_every
    config.eval_steps = args.eval_steps

    experiment_name = args.experiment_name
    output_dir = os.path.join(args.output_dir, experiment_name)

    # Nested optimizer config
    nested_config = {
        'base_lr': args.base_lr,
        'meta_lr': args.meta_lr,
        'k_unroll': args.k_unroll,
        'use_unrolled': args.use_unrolled,  # False by default - SimplifiedMetaTrainer is cheaper
        'use_cms_updates': args.use_cms,  # False by default (AdamW + learned LR multipliers)
        'momentum_hidden_dim': args.momentum_hidden_dim,
        'momentum_num_layers': args.momentum_num_layers,
        'controller_hidden_dim': args.controller_hidden_dim,
        'controller_num_layers': args.controller_num_layers,
        'mode': 'explicit',
        'meta_update_freq': config.eval_every,
        'weight_decay': config.weight_decay,
        'max_grad_norm': config.grad_clip,
        # Speedy mode: torch.compile + AMP + TF32
        'use_torch_compile': args.speedy,
        'use_amp': args.speedy,
        # Wandb logging
        'use_wandb': args.wandb and WANDB_AVAILABLE,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'log_every': args.log_every,
    }

    # Warn if wandb requested but not available
    if args.wandb and not WANDB_AVAILABLE:
        print("  WARNING: --wandb requested but wandb not installed. Run: pip install wandb")

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )

    from data.loader import setup_tokenizer

    # Setup tokenizer first to get vocab size
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching automatically)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Check for sufficient data
    total_needed = config.max_steps * config.batch_size
    if len(train_ds) < total_needed:
        msg = (
            f"Insufficient training data! "
            f"Need {total_needed} sequences (max_steps={config.max_steps} * batch_size={config.batch_size}) "
            f"but only have {len(train_ds)} sequences. "
            f"The model will overfit if data repeats. "
            f"To fix: increase num_documents (currently {config.num_documents}) "
            f"or reduce max_steps."
        )
        logger.error(msg)
        raise ValueError(msg)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel configuration (TitanMAC)")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"window_size: {config.window_size}, n_persistent: {config.n_persistent}")
    print(f"variant: {config.titans_variant}, neural_memory: {config.use_neural_memory}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"gradient_accumulation: {config.gradient_accumulation_steps}")
    print(f"vocab size: {config.vocab_size}")
    print(f"\nNested Optimizer configuration")
    print("-" * 70)
    print(f"base_lr: {nested_config['base_lr']}, meta_lr: {nested_config['meta_lr']}")
    print(f"k_unroll: {nested_config['k_unroll']}, mode: {nested_config['mode']}")
    meta_mode = "UnrolledMetaTrainer (memory intensive)" if nested_config['use_unrolled'] else "SimplifiedMetaTrainer (memory efficient)"
    print(f"meta-learning: {meta_mode}")
    print()
    logger.info(f"TitanMAC configuration: {vars(config)}")
    logger.info(f"Nested optimizer configuration: {nested_config}")

    print("Starting training with TitanMAC + DeepNestedOptimizer...")
    print("-" * 70)
    start = time.time()

    model, metrics, history = train_titanmac_nested(
        config, train_loader, val_loader,
        output_dir=output_dir,
        experiment_name=experiment_name,
        nested_config=nested_config,
        resume_from=args.resume,
    )

    elapsed = (time.time() - start) / 60
    print(f"\nTotal training time: {elapsed:.1f} min")

    # Print final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: TitanMAC + DeepNestedOptimizer - COMPLETE")
    print("=" * 70)
    print(f"Final Val Loss: {metrics['val_loss']:.4f}")
    print(f"Final Val Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Final Val Perplexity: {metrics['val_perplexity']:.2f}")
    print(f"Final Memory Loss: {metrics['val_aux_loss']:.4f}")
    print(f"Checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
