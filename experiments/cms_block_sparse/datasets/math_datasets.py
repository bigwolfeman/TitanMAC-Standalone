"""
Math Datasets for CMS Block-Sparse Benchmarks

Contains:
- MathDataset: Standard arithmetic (addition, subtraction, multiplication)
- ModularMathDataset: Modular arithmetic for ID-semantic forgetting tests
- MathTokenizer: Tokenizer with MODE tokens for context-switched experiments

The ModularMathDataset is critical for testing in-distribution semantic forgetting,
where the model sees the SAME inputs (e.g., "5 + 3 =") but must produce DIFFERENT
outputs depending on whether it was trained on standard or modular arithmetic.
"""

import random
from typing import Iterator, Dict, Optional, List

import torch
from torch.utils.data import IterableDataset


# Math vocabulary (tokens 0-18)
MATH_VOCAB = {
    '<pad>': 0,
    '<eos>': 1,
    '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
    '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
    '+': 12, '-': 13, '*': 14, '=': 15,
    ' ': 16, '|': 17, ',': 18,
}

# Extended math vocab with MODE tokens for context-switched experiments
EXTENDED_MATH_VOCAB = {
    **MATH_VOCAB,
    'MODE': 47,
    ':': 44,
    'STD': 48,
    'MOD7': 49,
}

ID_TO_TOKEN = {v: k for k, v in MATH_VOCAB.items()}
EXTENDED_ID_TO_TOKEN = {v: k for k, v in EXTENDED_MATH_VOCAB.items()}


def format_number(n: int) -> str:
    """Format number with commas for thousands."""
    return f"{n:,}"


class MathTokenizer:
    """
    Tokenizer for math expressions with restricted vocabulary.

    Supports extended mode with MODE:STD and MODE:MOD7 tokens for
    context-switched experiments.

    Attributes:
        vocab: Token to ID mapping
        id_to_token: ID to token mapping
        pad_token_id: Padding token ID (0)
        eos_token_id: End-of-sequence token ID (1)
        extended: Whether to use extended vocab with MODE tokens
    """

    def __init__(self, extended: bool = False):
        """
        Initialize MathTokenizer.

        Args:
            extended: If True, includes MODE tokens for context-switching
        """
        self.extended = extended
        if extended:
            self.vocab = EXTENDED_MATH_VOCAB.copy()
            self.id_to_token = EXTENDED_ID_TO_TOKEN.copy()
        else:
            self.vocab = MATH_VOCAB.copy()
            self.id_to_token = ID_TO_TOKEN.copy()

        self.pad_token_id = self.vocab['<pad>']
        self.eos_token_id = self.vocab['<eos>']

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return max(self.vocab.values()) + 1

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            max_length: Maximum sequence length (pads/truncates if set)
            return_tensors: If "pt", returns PyTorch tensor

        Returns:
            Token IDs as list or tensor
        """
        ids = []

        # Handle special multi-char tokens first
        i = 0
        while i < len(text):
            # Check for multi-char tokens (MODE, STD, MOD7)
            matched = False
            for token in ['MODE', 'MOD7', 'STD']:
                if text[i:i+len(token)] == token and token in self.vocab:
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break

            if not matched:
                char = text[i]
                if char in self.vocab:
                    ids.append(self.vocab[char])
                i += 1

        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: Token IDs (list, tensor, or int)
            skip_special_tokens: Skip <pad> and <eos> tokens

        Returns:
            Decoded text string
        """
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

    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Optional[str] = None,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """
        Tokenize text (HuggingFace-compatible interface).

        Args:
            text: Input text
            max_length: Maximum sequence length
            truncation: Whether to truncate (unused, always truncates if max_length set)
            padding: Padding strategy (unused, always pads to max_length)
            return_tensors: If "pt", returns PyTorch tensors

        Returns:
            Dict with input_ids and attention_mask
        """
        ids = self.encode(text, max_length=max_length, return_tensors=return_tensors)

        if return_tensors == "pt":
            attention_mask = (ids != self.pad_token_id).long()
            return {"input_ids": ids, "attention_mask": attention_mask}

        return {"input_ids": ids}


class MathDataset(IterableDataset):
    """
    Standard arithmetic dataset for baseline training and evaluation.

    Generates problems like:
    - "45 + 32 = 77"
    - "123 - 45 = 78"
    - "12 * 7 = 84"

    Used as Task A in OOD forgetting experiments.

    Attributes:
        tokenizer: MathTokenizer instance
        seq_length: Maximum sequence length
        rng: Random number generator (seeded for reproducibility)
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
        max_num: int = 99,
        include_large: bool = True,
    ):
        """
        Initialize MathDataset.

        Args:
            tokenizer: MathTokenizer for encoding
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            max_num: Maximum number for operands (default 99)
            include_large: If True, 40% of problems use 3-digit numbers
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)
        self.max_num = max_num
        self.include_large = include_large

    def _generate_problem(self) -> str:
        """Generate a single math problem with formatted answer."""
        # Mix 2-digit and 3-digit problems
        if self.include_large and self.rng.random() < 0.4:
            max_num = 999
            mult_max = 31
        else:
            max_num = min(self.max_num, 99)
            mult_max = 12

        op = self.rng.choice(['+', '-', '*'])

        if op == '+':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(10, max_num)
            result = a + b
        elif op == '-':
            a = self.rng.randint(10, max_num)
            b = self.rng.randint(0, a)
            result = a - b
        else:  # multiplication
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

            # Pack multiple problems into one sequence
            while total_chars < self.seq_length * 2:
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


class ModularMathDataset(IterableDataset):
    """
    Modular arithmetic dataset for ID-semantic forgetting tests.

    Generates the SAME input patterns as MathDataset but with results
    computed modulo N. For example with modulus=7:
    - "5 + 3 = 1"   (8 mod 7 = 1, vs standard "5 + 3 = 8")
    - "12 * 4 = 6"  (48 mod 7 = 6, vs standard "12 * 4 = 48")

    This tests whether topology can learn to separate pathways for
    identical inputs that require different outputs.

    Critical for ID-Semantic forgetting benchmarks (Phase 5.2).

    Attributes:
        tokenizer: MathTokenizer instance
        seq_length: Maximum sequence length
        modulus: Modulo value (default 7)
        rng: Random number generator
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
        modulus: int = 7,
        max_num: int = 99,
    ):
        """
        Initialize ModularMathDataset.

        Args:
            tokenizer: MathTokenizer for encoding
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            modulus: Modulo value for arithmetic (default 7)
            max_num: Maximum number for operands
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)
        self.modulus = modulus
        self.max_num = max_num

    def _generate_problem(self) -> str:
        """Generate a single modular math problem."""
        op = self.rng.choice(['+', '-', '*'])

        a = self.rng.randint(0, self.max_num)
        b = self.rng.randint(0, self.max_num)

        if op == '+':
            result = (a + b) % self.modulus
        elif op == '-':
            result = (a - b) % self.modulus
        else:  # multiplication
            result = (a * b) % self.modulus

        # Result is always single digit (0 to modulus-1)
        return f"{a} {op} {b} = {result}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized modular math problems indefinitely."""
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


class ContextSwitchedMathDataset(IterableDataset):
    """
    Mode-switched math dataset for ID-context forgetting tests.

    Prefixes each problem with MODE:STD or MODE:MOD7 to indicate
    which arithmetic system to use:
    - "MODE:STD | 5 + 3 = 8"
    - "MODE:MOD7 | 5 + 3 = 1"

    This is the HARDEST forgetting test - a single prefix token must
    route to entirely different computation pathways.

    Critical for ID-Context forgetting benchmarks (Phase 5.4).

    Attributes:
        tokenizer: Extended MathTokenizer (with MODE tokens)
        seq_length: Maximum sequence length
        mode: Either "STD" (standard) or "MOD7" (modular)
        modulus: Modulo value when mode="MOD7" (default 7)
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
        mode: str = "STD",
        modulus: int = 7,
        max_num: int = 99,
    ):
        """
        Initialize ContextSwitchedMathDataset.

        Args:
            tokenizer: MathTokenizer with extended=True
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            mode: "STD" for standard arithmetic, "MOD7" for modular
            modulus: Modulo value when mode="MOD7"
            max_num: Maximum number for operands
        """
        if not tokenizer.extended:
            raise ValueError("ContextSwitchedMathDataset requires extended tokenizer")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)
        self.mode = mode
        self.modulus = modulus
        self.max_num = max_num

    def _generate_problem(self) -> str:
        """Generate a mode-prefixed math problem."""
        op = self.rng.choice(['+', '-', '*'])

        a = self.rng.randint(0, self.max_num)
        b = self.rng.randint(0, self.max_num)

        if op == '+':
            raw_result = a + b
        elif op == '-':
            raw_result = a - b
        else:
            raw_result = a * b

        if self.mode == "MOD7":
            result = raw_result % self.modulus
        else:
            result = raw_result

        return f"MODE:{self.mode} | {a} {op} {b} = {result}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized context-switched math problems."""
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


def generate_fixed_eval_inputs(
    n: int = 500,
    seed: int = 42,
    max_num: int = 99
) -> List[str]:
    """
    Generate fixed arithmetic input prompts for evaluation.

    These are used to compare model outputs before/after task switching.
    The same inputs are evaluated under both standard and modular modes.

    Args:
        n: Number of inputs to generate
        seed: Random seed for reproducibility
        max_num: Maximum number for operands

    Returns:
        List of input prompts like ["5 + 3 =", "12 * 4 =", ...]
    """
    rng = random.Random(seed)
    inputs = []

    for _ in range(n):
        op = rng.choice(['+', '-', '*'])
        a = rng.randint(0, max_num)
        b = rng.randint(0, max_num)
        inputs.append(f"{a} {op} {b} =")

    return inputs
