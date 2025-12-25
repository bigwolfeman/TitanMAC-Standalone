#!/usr/bin/env python3
"""
A/B test script for comparing TitanMAC optimizer implementations.

Supports:
- DeepNestedOptimizer (--optimizer deep_nested)
- ContinuumOptimizer (--optimizer continuum)
- AdamW baseline (--optimizer adam)

Usage for A/B tests:
    # Phase 1: DeepNestedOptimizer performance test (500 steps)
    python examples/ab_test_optimizers.py --steps 500 --optimizer deep_nested --log-interval 100

    # Phase 2: ContinuumOptimizer quality test (2000 steps)
    python examples/ab_test_optimizers.py --steps 2000 --optimizer continuum --log-interval 200
"""

import argparse
import random
import time
import json
from typing import Iterator, List, Tuple, Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC


# Restricted vocabulary for math
MATH_VOCAB = {
    '<pad>': 0,
    '<eos>': 1,
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
    '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '*': 14, '=': 15,
    ' ': 16, '|': 17, ',': 18,
}
VOCAB_SIZE = len(MATH_VOCAB)
ID_TO_TOKEN = {v: k for k, v in MATH_VOCAB.items()}


class MathTokenizer:
    """Simple tokenizer for math expressions with restricted vocabulary."""

    def __init__(self):
        self.vocab = MATH_VOCAB
        self.id_to_token = ID_TO_TOKEN
        self.pad_token_id = MATH_VOCAB['<pad>']
        self.eos_token_id = MATH_VOCAB['<eos>']

    def encode(self, text: str, max_length: int = None, return_tensors: str = None) -> torch.Tensor:
        """Encode text to token IDs."""
        ids = []
        for char in text:
            if char in self.vocab:
                ids.append(self.vocab[char])

        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]

        result = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in ['<pad>', '<eos>']:
                    continue
                result.append(token)
        return ''.join(result)

    def __call__(self, text: str, max_length: int = None, truncation: bool = False,
                 padding: str = None, return_tensors: str = None) -> Dict:
        """Tokenize text (HuggingFace-compatible interface)."""
        ids = self.encode(text, max_length=max_length, return_tensors=return_tensors)

        if return_tensors == "pt":
            attention_mask = (ids != self.pad_token_id).long()
            return {"input_ids": ids, "attention_mask": attention_mask}

        return {"input_ids": ids}


def format_number(n: int) -> str:
    """Format number with commas for thousands."""
    return f"{n:,}"


class ImprovedMathDataset(IterableDataset):
    """
    Generates arithmetic problems with:
    - Comma formatting for numbers >= 1000
    - Mix of 2-digit and 3-digit problems
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        seq_length: int = 128,
        seed: int = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)

    def _generate_problem(self) -> str:
        """Generate a single math problem with formatted answer."""
        if self.rng.random() < 0.6:
            max_num = 99
            mult_max = 12
        else:
            max_num = 999
            mult_max = 31

        op = self.rng.choice(['+', '-', '*'])

        if op == '+':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(10, max_num)
            result = a + b
        elif op == '-':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(0, a)
            result = a - b
        else:
            a = self.rng.randint(2, mult_max)
            b = self.rng.randint(2, mult_max)
            result = a * b

        a_str = format_number(a)
        b_str = format_number(b)
        result_str = format_number(result)

        return f"{a_str} {op} {b_str} = {result_str}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized math problems indefinitely."""
        while True:
            problems = []
            total_chars = 0

            while total_chars < self.seq_length * 2:
                problem = self._generate_problem()
                problems.append(problem)
                total_chars += len(problem) + 3

            text = " | ".join(problems)

            encoded = self.tokenizer(
                text,
                max_length=self.seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {"input_ids": input_ids, "labels": labels}


def create_model(config: TitanMACConfig) -> TitanMAC:
    """Create TitanMAC model from config."""
    model = TitanMAC(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    max_steps: int,
    log_interval: int = 100,
    optimizer_type: str = "adam",
    run_name: str = "",
):
    """Training loop with detailed metrics."""
    model.train()

    total_loss = 0.0
    step = 0
    accum_step = 0
    start_time = time.time()

    # Metrics storage
    metrics = {
        "steps": [],
        "loss": [],
        "tok_s": [],
        "lr_mult_core": [],
        "lr_mult_embed": [],
        "step_time": [],
    }

    optimizer.zero_grad()
    is_nested_opt = optimizer_type in ["deep_nested", "continuum"]

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        step_start = time.time()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if not is_nested_opt:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        if is_nested_opt:
            result = optimizer.step(loss.item())
            lr_mults = result.get('lr_multipliers', None)
            if lr_mults is not None and len(lr_mults) >= 2:
                lr_mult_core = lr_mults[0].item() if isinstance(lr_mults[0], torch.Tensor) else lr_mults[0]
                lr_mult_embed = lr_mults[1].item() if isinstance(lr_mults[1], torch.Tensor) else lr_mults[1]
            else:
                lr_mult_core = lr_mult_embed = 1.0
        else:
            optimizer.step()
            lr_mult_core = lr_mult_embed = 1.0

        optimizer.zero_grad()
        step += 1

        step_time = time.time() - step_start
        total_loss += loss.item()

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (log_interval * input_ids.numel()) / elapsed

            # Store metrics
            metrics["steps"].append(step)
            metrics["loss"].append(avg_loss)
            metrics["tok_s"].append(tokens_per_sec)
            metrics["lr_mult_core"].append(lr_mult_core)
            metrics["lr_mult_embed"].append(lr_mult_embed)
            metrics["step_time"].append(step_time)

            log_str = f"[{run_name}] Step {step:5d} | Loss: {avg_loss:.4f} | Tok/s: {tokens_per_sec:.0f}"
            if is_nested_opt:
                log_str += f" | LR_core: {lr_mult_core:.4f} | LR_embed: {lr_mult_embed:.4f}"

            print(log_str)

            total_loss = 0.0
            start_time = time.time()

        if step >= max_steps:
            break

    print(f"\n[{run_name}] Training complete! Final step: {step}")

    # Print final summary
    if metrics["tok_s"]:
        avg_tok_s = sum(metrics["tok_s"]) / len(metrics["tok_s"])
        final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0
        final_lr_core = metrics["lr_mult_core"][-1] if metrics["lr_mult_core"] else 1.0
        final_lr_embed = metrics["lr_mult_embed"][-1] if metrics["lr_mult_embed"] else 1.0

        print(f"\n=== Final Metrics ===")
        print(f"Average tok/s: {avg_tok_s:.2f}")
        print(f"Final loss: {final_loss:.4f}")
        if is_nested_opt:
            print(f"Final LR_mult_core: {final_lr_core:.4f}")
            print(f"Final LR_mult_embed: {final_lr_embed:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="A/B test TitanMAC optimizers")

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)

    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "deep_nested", "continuum"])
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--meta-update-freq", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/ab_tests")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_name = args.run_name or f"{args.optimizer}_seed{args.seed}"

    print("=" * 60)
    print(f"TitanMAC A/B Test - {run_name}")
    print("=" * 60)

    tokenizer = MathTokenizer()
    print(f"Vocab size: {VOCAB_SIZE} (restricted to math tokens)")

    config = TitanMACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        window_size=64,
        n_persistent=8,
        dropout=0.1,
    )

    print(f"\nCreating model...")
    model = create_model(config)
    model = model.to(args.device)

    dataset = ImprovedMathDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    # Create optimizer
    if args.optimizer == "deep_nested":
        from titans_core.opt import DeepNestedOptimizer
        optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=args.lr,
            meta_lr=args.meta_lr,
            cms_frequencies=[1, 10, 100],
            mode='simple',
            meta_update_freq=args.meta_update_freq,
            weight_decay=0.01,
            max_grad_norm=1.0,
        )
        print(f"Using DeepNestedOptimizer (mode=simple)")
    elif args.optimizer == "continuum":
        from titans_core.opt import ContinuumOptimizer
        optimizer = ContinuumOptimizer(
            model=model,
            base_lr=args.lr,
            update_freq=args.meta_update_freq,
            controller_lr=args.meta_lr,
            base_optim_kwargs={"weight_decay": 0.01},
        )
        print(f"Using ContinuumOptimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"Using AdamW")

    print(f"\nTraining for {args.steps} steps...")
    print("-" * 60)

    metrics = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device(args.device),
        max_steps=args.steps,
        log_interval=args.log_interval,
        optimizer_type=args.optimizer,
        run_name=run_name,
    )

    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{run_name}_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {results_file}")


if __name__ == "__main__":
    main()
