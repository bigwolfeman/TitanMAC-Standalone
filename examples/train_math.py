#!/usr/bin/env python3
"""
Example training script for TitanMAC on generated math data.

Uses StarCoder2 tokenizer and generates arithmetic problems for training.
This demonstrates how to use the TitanMAC architecture standalone.

Supports two optimizer modes:
- Adam/AdamW: Standard optimizer (default)
- Nested: DeepNestedOptimizer implementing Nested Learning (NeurIPS 2025)

Usage:
    # Standard training with AdamW
    python examples/train_math.py --steps 1000 --batch-size 4

    # Training with Nested Learning optimizer
    python examples/train_math.py --steps 1000 --optimizer nested --meta-lr 1e-4
"""

import argparse
import random
import time
from typing import Iterator, Tuple

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

# Import from local package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC


class MathDataset(IterableDataset):
    """
    Generates arithmetic problems on-the-fly.

    Examples:
        "123 + 456 = 579"
        "789 - 234 = 555"
        "12 * 34 = 408"
    """

    def __init__(
        self,
        tokenizer,
        seq_length: int = 128,
        max_num: int = 1000,
        operations: Tuple[str, ...] = ("+", "-", "*"),
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_num = max_num
        self.operations = operations

    def _generate_problem(self) -> str:
        """Generate a single math problem with answer."""
        op = random.choice(self.operations)

        if op == "+":
            a = random.randint(0, self.max_num)
            b = random.randint(0, self.max_num)
            result = a + b
        elif op == "-":
            a = random.randint(0, self.max_num)
            b = random.randint(0, a)  # Ensure non-negative result
            result = a - b
        elif op == "*":
            a = random.randint(0, int(self.max_num ** 0.5))
            b = random.randint(0, int(self.max_num ** 0.5))
            result = a * b
        else:
            raise ValueError(f"Unknown operation: {op}")

        return f"{a} {op} {b} = {result}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized math problems indefinitely."""
        while True:
            # Generate multiple problems to fill sequence
            problems = []
            total_tokens = 0

            while total_tokens < self.seq_length:
                problem = self._generate_problem()
                problems.append(problem)
                # Rough estimate: 1 token per 3 chars
                total_tokens += len(problem) // 3 + 2

            text = " | ".join(problems)

            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # Create labels - mask out padding tokens with -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


def create_model(config: TitanMACConfig) -> TitanMAC:
    """Create TitanMAC model from config."""
    model = TitanMAC(config)

    # Print param count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    return model


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 15,
    greedy: bool = True,
) -> str:
    """Generate text from a prompt using greedy decoding."""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]

            # Get next token logits (last position)
            next_logits = logits[:, -1, :]

            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at pipe or space after getting some digits
            decoded_token = tokenizer.decode(next_token.item())
            if decoded_token in ["|", "\n"] or (input_ids.shape[1] > prompt_len + 5 and decoded_token.strip() == ""):
                break

    model.train()
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,  # Can be torch.optim.Optimizer or DeepNestedOptimizer
    device: torch.device,
    tokenizer,
    max_steps: int,
    log_interval: int = 10,
    sample_interval: int = 40,
    grad_accum_steps: int = 1,
    use_nested_optimizer: bool = False,
):
    """Training loop with periodic sample generation."""
    model.train()

    total_loss = 0.0
    step = 0
    accum_step = 0
    start_time = time.time()

    # Track nested optimizer metrics
    nested_metrics = {'lr_mults': [], 'meta_loss': None}

    # Test prompts for generation (no trailing space - model predicts space then digit)
    test_prompts = [
        "5 + 3 =",
        "10 - 4 =",
        "7 * 6 =",
        "123 + 456 =",
        "50 - 25 =",
    ]

    optimizer.zero_grad()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"] / grad_accum_steps

        # Backward pass
        loss.backward()

        accum_step += 1
        total_loss += loss.item() * grad_accum_steps

        # Optimizer step
        if accum_step >= grad_accum_steps:
            # Gradient clipping (nested optimizer does this internally)
            if not use_nested_optimizer:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Step with loss value for nested optimizer
            if use_nested_optimizer:
                result = optimizer.step(total_loss / grad_accum_steps)
                nested_metrics['lr_mults'] = result.get('lr_multipliers', [])
                nested_metrics['meta_loss'] = optimizer.last_meta_loss
            else:
                optimizer.step()

            optimizer.zero_grad()

            step += 1
            accum_step = 0

            # Logging
            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (log_interval * input_ids.numel()) / elapsed

                log_str = (
                    f"Step {step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )

                # Add nested optimizer metrics
                if use_nested_optimizer and len(nested_metrics['lr_mults']) > 0:
                    lr_mults = nested_metrics['lr_mults']
                    lr_str = ", ".join(f"{m:.3f}" for m in lr_mults.tolist())
                    log_str += f" | LR mults: [{lr_str}]"

                print(log_str)

                total_loss = 0.0
                start_time = time.time()

            # Sample generation every sample_interval steps
            if step % sample_interval == 0:
                print("\n" + "=" * 50)
                print(f"Sample generations at step {step}:")
                print("-" * 50)
                for prompt in test_prompts:
                    generated = generate_sample(model, tokenizer, prompt, device)
                    # Show prompt vs generated
                    print(f"  '{prompt}' -> '{generated}'")

                # Debug: show what the model predicts as top-5 next tokens for first prompt
                model.eval()
                debug_prompt = "5 + 3 ="
                debug_ids = tokenizer.encode(debug_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    debug_out = model(input_ids=debug_ids)
                    debug_logits = debug_out["logits"][0, -1, :]  # last position
                    top5 = debug_logits.topk(5)
                    print(f"\n  Debug top-5 next tokens after '{debug_prompt}':")
                    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
                        tok = tokenizer.decode(idx.item())
                        print(f"    {i+1}. '{tok}' (id={idx.item()}, logit={val.item():.2f})")
                model.train()
                print("=" * 50 + "\n")

            if step >= max_steps:
                break

    print(f"\nTraining complete! Final step: {step}")


def main():
    parser = argparse.ArgumentParser(description="Train TitanMAC on math data")

    # Training args
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    # Model args
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--window-size", type=int, default=64, help="Attention window")
    parser.add_argument("--n-persistent", type=int, default=8, help="Persistent tokens")

    # Neural memory
    parser.add_argument("--use-neural-memory", action="store_true", help="Enable neural memory")
    parser.add_argument("--memory-capacity", type=int, default=128, help="Memory slots")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer selection (Nested Learning support)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "nested"],
                        help="Optimizer: 'adam' (AdamW) or 'nested' (DeepNestedOptimizer)")
    parser.add_argument("--meta-lr", type=float, default=1e-4,
                        help="Meta-learning rate for nested optimizer")
    parser.add_argument("--meta-update-freq", type=int, default=50,
                        help="Meta-update frequency for nested optimizer")
    parser.add_argument("--cms-frequencies", type=str, default="1,10,100",
                        help="CMS update frequencies (comma-separated)")

    args = parser.parse_args()

    print("=" * 60)
    print("TitanMAC Training on Generated Math Data")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading StarCoder2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}")

    # Create config
    config = TitanMACConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        window_size=args.window_size,
        n_persistent=args.n_persistent,
        use_neural_memory=args.use_neural_memory,
        memory_capacity=args.memory_capacity if args.use_neural_memory else 512,
        dropout=0.1,
    )
    config.validate()
    print(f"\nConfig: {config}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(args.device)

    # Create dataset
    print("\nCreating math dataset...")
    dataset = MathDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        max_num=10000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Create optimizer
    use_nested_optimizer = args.optimizer == "nested"

    if use_nested_optimizer:
        from titans_core.opt import DeepNestedOptimizer
        cms_freqs = [int(x) for x in args.cms_frequencies.split(",")]
        optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=args.lr,
            meta_lr=args.meta_lr,
            cms_frequencies=cms_freqs,
            mode='simple',
            meta_update_freq=args.meta_update_freq,
            weight_decay=0.01,
            max_grad_norm=1.0,
        )
        print(f"\nUsing DeepNestedOptimizer (Nested Learning)")
        print(f"  Meta LR: {args.meta_lr}")
        print(f"  Meta update freq: {args.meta_update_freq}")
        print(f"  CMS frequencies: {cms_freqs}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )
        print(f"\nUsing AdamW optimizer")

    # Train
    print(f"\nTraining for {args.steps} steps on {args.device}...")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")
    print("-" * 60)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device(args.device),
        tokenizer=tokenizer,
        max_steps=args.steps,
        log_interval=args.log_interval,
        sample_interval=40,
        grad_accum_steps=args.grad_accum,
        use_nested_optimizer=use_nested_optimizer,
    )

    # Save checkpoint
    checkpoint_name = "math_model_nested.pt" if use_nested_optimizer else "math_model.pt"
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / checkpoint_name
    checkpoint_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "step": args.steps,
        "optimizer_type": args.optimizer,
    }

    # Save optimizer state
    if use_nested_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["lr_multipliers"] = optimizer.get_lr_multipliers().tolist()
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)

    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Print final nested optimizer stats
    if use_nested_optimizer:
        lr_mults = optimizer.get_lr_multipliers()
        print(f"Final LR multipliers: {lr_mults.tolist()}")


if __name__ == "__main__":
    main()
