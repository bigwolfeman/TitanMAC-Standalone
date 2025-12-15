#!/usr/bin/env python3
"""
Improved math training script with:
- Restricted vocabulary (only digits, operators, formatting)
- Comma formatting for large numbers (1,000 not 1000)
- Difficulty levels (2-digit, 3-digit problems)
- Seed for reproducibility

Usage:
    python examples/train_math_v2.py --steps 5000 --optimizer adam --seed 42
    python examples/train_math_v2.py --steps 5000 --optimizer nested --seed 42
"""

import argparse
import random
import time
from typing import Iterator, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import sys
from pathlib import Path
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
            # Skip unknown characters

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
        # Choose difficulty: 2-digit (60%), 3-digit (40%)
        if self.rng.random() < 0.6:
            # 2-digit problems
            max_num = 99
            mult_max = 12  # Up to 12x12
        else:
            # 3-digit problems
            max_num = 999
            mult_max = 31  # Up to 31x31

        op = self.rng.choice(['+', '-', '*'])

        if op == '+':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(10, max_num)
            result = a + b
        elif op == '-':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(0, a)
            result = a - b
        else:  # *
            a = self.rng.randint(2, mult_max)
            b = self.rng.randint(2, mult_max)
            result = a * b

        # Format with commas
        a_str = format_number(a)
        b_str = format_number(b)
        result_str = format_number(result)

        return f"{a_str} {op} {b_str} = {result_str}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized math problems indefinitely."""
        while True:
            problems = []
            total_chars = 0

            while total_chars < self.seq_length * 2:  # Rough estimate
                problem = self._generate_problem()
                problems.append(problem)
                total_chars += len(problem) + 3  # +3 for " | "

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


def generate_sample(
    model: nn.Module,
    tokenizer: MathTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    """Generate text from a prompt using greedy decoding."""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            decoded = tokenizer.decode(next_token.item())
            if decoded in ['|', '<eos>', '<pad>']:
                break

    model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    tokenizer: MathTokenizer,
    max_steps: int,
    log_interval: int = 100,
    sample_interval: int = 500,
    grad_accum_steps: int = 1,
    use_nested_optimizer: bool = False,
    run_name: str = "",
):
    """Training loop."""
    model.train()

    total_loss = 0.0
    step = 0
    accum_step = 0
    start_time = time.time()

    # Test prompts with comma formatting
    test_prompts = [
        "5 + 3 =",
        "12 + 7 =",
        "45 - 23 =",
        "8 * 9 =",
        "123 + 456 =",
        "99 * 11 =",
    ]

    optimizer.zero_grad()
    loss_history = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"] / grad_accum_steps
        loss.backward()

        accum_step += 1
        total_loss += loss.item() * grad_accum_steps

        if accum_step >= grad_accum_steps:
            if not use_nested_optimizer:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_nested_optimizer:
                result = optimizer.step(total_loss / grad_accum_steps)
                lr_mults = result.get('lr_multipliers', torch.ones(2))
            else:
                optimizer.step()
                lr_mults = None

            optimizer.zero_grad()
            step += 1
            accum_step = 0

            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (log_interval * input_ids.numel()) / elapsed

                log_str = f"[{run_name}] Step {step:5d} | Loss: {avg_loss:.4f} | Tok/s: {tokens_per_sec:.0f}"
                if use_nested_optimizer and lr_mults is not None:
                    lr_str = ", ".join(f"{m:.3f}" for m in lr_mults.tolist())
                    log_str += f" | LR: [{lr_str}]"

                print(log_str)
                loss_history.append((step, avg_loss))

                total_loss = 0.0
                start_time = time.time()

            if step % sample_interval == 0:
                print(f"\n--- Samples at step {step} ---")
                for prompt in test_prompts:
                    gen = generate_sample(model, tokenizer, prompt, device)
                    # Evaluate correctness
                    try:
                        expr = prompt.replace('=', '').strip()
                        expected = eval(expr.replace(',', ''))
                        expected_str = format_number(expected)
                        gen_answer = gen.split('=')[-1].strip().split('|')[0].strip()
                        correct = gen_answer == expected_str
                        mark = "OK" if correct else "X"
                    except:
                        mark = "?"
                        expected_str = "?"
                    print(f"  {prompt} -> {gen.split('|')[0].strip()} (expect {expected_str}) [{mark}]")
                print("---\n")

            if step >= max_steps:
                break

    print(f"\n[{run_name}] Training complete! Final step: {step}")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description="Train TitanMAC on improved math data")

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=500)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "nested"])
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--meta-update-freq", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_name = args.run_name or f"{args.optimizer}_seed{args.seed}"

    print("=" * 60)
    print(f"TitanMAC Math Training - {run_name}")
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

    use_nested = args.optimizer == "nested"

    if use_nested:
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
        print(f"Using DeepNestedOptimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"Using AdamW")

    print(f"\nTraining for {args.steps} steps...")
    print("-" * 60)

    loss_history = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device(args.device),
        tokenizer=tokenizer,
        max_steps=args.steps,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        use_nested_optimizer=use_nested,
        run_name=run_name,
    )

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"{run_name}_loss.txt"
    with open(results_file, 'w') as f:
        for step, loss in loss_history:
            f.write(f"{step},{loss:.6f}\n")

    print(f"\nLoss history saved to: {results_file}")

    if use_nested:
        lr_mults = optimizer.get_lr_multipliers()
        print(f"Final LR multipliers: {lr_mults.tolist()}")


if __name__ == "__main__":
    main()
