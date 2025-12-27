"""
NLP Datasets for CMS Block-Sparse Benchmarks

Contains:
- SyntheticNLPDataset: Generated linguistic patterns (POS tagging, antonyms, etc.)
- NLPTokenizer: Extended tokenizer with ~50 NLP tokens
- NLP_VOCAB: Vocabulary dictionary starting at token ID 19

These datasets are designed for cross-domain forgetting experiments (OOD)
and share the base tokenizer with math datasets for unified training.

Pattern Types:
1. pos - Part-of-speech tagging: "The quick fox jumps | quick: ADJ"
2. antonym - Antonym completion: "hot is to cold as big is to: small"
3. reorder - Word reordering: "jumped fox the brown | the brown fox jumped"
4. cloze - Fill-in-the-blank: "The ___ barks loudly | dog"
5. sequence - Pattern completion: "A B A B A: B"
"""

import random
from typing import Iterator, Dict, Optional, List, Tuple

import torch
from torch.utils.data import IterableDataset

from .math_datasets import MATH_VOCAB, format_number


# NLP vocabulary starts at token ID 19 (after math vocab ends at 18)
NLP_VOCAB = {
    # Core function words
    'the': 19, 'a': 20, 'is': 21, 'to': 22, 'of': 23,
    'and': 24, 'in': 25, 'on': 26, 'at': 27, 'as': 28,
    'was': 89, 'by': 90,  # For OVS passive construction

    # Transitive verbs (past tense / past participle forms)
    'chased': 91, 'followed': 92, 'watched': 93, 'helped': 94,
    'found': 95, 'caught': 96, 'saw': 97, 'seen': 98, 'heard': 99,

    # Animals (nouns)
    'dog': 29, 'cat': 30, 'fox': 31, 'bird': 32, 'mouse': 33,
    'bear': 34, 'fish': 35, 'frog': 36, 'wolf': 37, 'deer': 38,

    # Adjectives
    'big': 39, 'small': 40, 'hot': 41, 'cold': 42, 'quick': 43,
    'slow': 44, 'loud': 45, 'soft': 46, 'old': 47, 'young': 48,
    'brown': 49, 'white': 50, 'black': 51, 'red': 52, 'blue': 53,

    # Verbs
    'run': 54, 'jump': 55, 'sit': 56, 'bark': 57, 'chase': 58,
    'eat': 59, 'sleep': 60, 'swim': 61, 'fly': 62, 'walk': 63,
    'runs': 64, 'jumps': 65, 'sits': 66, 'barks': 67, 'chases': 68,

    # Adverbs
    'loudly': 69, 'softly': 70, 'quickly': 71, 'slowly': 72, 'always': 73,

    # POS tags
    'NOUN': 74, 'VERB': 75, 'ADJ': 76, 'ADV': 77, 'SUBJ': 78,
    'OBJ': 79, 'DET': 80, 'PREP': 81,

    # Special tokens for NLP patterns
    '_': 82,  # Blank for cloze

    # Sequence pattern tokens
    'A': 83, 'B': 84, 'C': 85, 'X': 86, 'Y': 87, 'Z': 88,
}

# Combined vocabulary (math + NLP)
COMBINED_VOCAB = {**MATH_VOCAB, **NLP_VOCAB}
COMBINED_ID_TO_TOKEN = {v: k for k, v in COMBINED_VOCAB.items()}

# Word banks for generation
NOUNS = ['dog', 'cat', 'fox', 'bird', 'mouse', 'bear', 'fish', 'frog', 'wolf', 'deer']
ADJECTIVES = ['big', 'small', 'quick', 'slow', 'loud', 'soft', 'old', 'young', 'brown', 'white']
VERBS_BASE = ['run', 'jump', 'sit', 'bark', 'chase', 'eat', 'sleep', 'swim', 'fly', 'walk']
VERBS_THIRD = ['runs', 'jumps', 'sits', 'barks', 'chases']  # Third person singular
ADVERBS = ['loudly', 'softly', 'quickly', 'slowly', 'always']

# Transitive verbs for SVO/OVS experiments (same verbs in both datasets)
# Format: (past_tense, past_participle) - for regular verbs these are the same
TRANSITIVE_VERBS = [
    ('chased', 'chased'),      # chase
    ('followed', 'followed'),  # follow
    ('watched', 'watched'),    # watch
    ('helped', 'helped'),      # help
    ('found', 'found'),        # find (irregular but same form)
    ('caught', 'caught'),      # catch (irregular but same form)
    ('saw', 'seen'),           # see (irregular)
    ('heard', 'heard'),        # hear
]

# Antonym pairs
ANTONYM_PAIRS = [
    ('hot', 'cold'), ('cold', 'hot'),
    ('big', 'small'), ('small', 'big'),
    ('quick', 'slow'), ('slow', 'quick'),
    ('loud', 'soft'), ('soft', 'loud'),
    ('old', 'young'), ('young', 'old'),
]


class NLPTokenizer:
    """
    Tokenizer for NLP patterns with extended vocabulary.

    Extends the math tokenizer with ~50 NLP tokens for cross-domain
    experiments. Supports the same interface as MathTokenizer.

    Attributes:
        vocab: Combined token to ID mapping
        id_to_token: ID to token mapping
        pad_token_id: Padding token ID (0)
        eos_token_id: End-of-sequence token ID (1)
    """

    def __init__(self):
        """Initialize NLPTokenizer with combined vocabulary."""
        self.vocab = COMBINED_VOCAB.copy()
        self.id_to_token = COMBINED_ID_TO_TOKEN.copy()
        self.pad_token_id = self.vocab['<pad>']
        self.eos_token_id = self.vocab['<eos>']

        # Build list of multi-char tokens sorted by length (longest first)
        self._multi_char_tokens = sorted(
            [k for k in self.vocab.keys() if len(k) > 1 and k not in ['<pad>', '<eos>']],
            key=lambda x: -len(x)
        )

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

        Uses greedy longest-match tokenization for multi-char tokens.

        Args:
            text: Input text string
            max_length: Maximum sequence length
            return_tensors: If "pt", returns PyTorch tensor

        Returns:
            Token IDs as list or tensor
        """
        ids = []
        i = 0

        while i < len(text):
            matched = False

            # Try multi-char tokens (longest first)
            for token in self._multi_char_tokens:
                if text[i:i+len(token)] == token:
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
        return ' '.join(result)

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
            truncation: Whether to truncate
            padding: Padding strategy
            return_tensors: If "pt", returns PyTorch tensors

        Returns:
            Dict with input_ids and attention_mask
        """
        ids = self.encode(text, max_length=max_length, return_tensors=return_tensors)

        if return_tensors == "pt":
            attention_mask = (ids != self.pad_token_id).long()
            return {"input_ids": ids, "attention_mask": attention_mask}

        return {"input_ids": ids}


class SyntheticNLPDataset(IterableDataset):
    """
    Synthetic NLP pattern dataset for cross-domain forgetting tests.

    Generates 5 types of linguistic patterns:
    1. pos - Part-of-speech tagging
    2. antonym - Antonym completion
    3. reorder - Word reordering
    4. cloze - Fill-in-the-blank
    5. sequence - Abstract pattern completion

    Used as Task B in OOD forgetting experiments to test whether
    the model can learn NLP patterns without forgetting math.

    Attributes:
        tokenizer: NLPTokenizer instance
        seq_length: Maximum sequence length
        pattern_types: List of pattern types to generate
        rng: Random number generator
    """

    PATTERN_TYPES = ['pos', 'antonym', 'reorder', 'cloze', 'sequence']

    def __init__(
        self,
        tokenizer: NLPTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
        pattern_types: Optional[List[str]] = None,
    ):
        """
        Initialize SyntheticNLPDataset.

        Args:
            tokenizer: NLPTokenizer for encoding
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            pattern_types: List of pattern types to include (default: all)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)
        self.pattern_types = pattern_types or self.PATTERN_TYPES

    def _generate_pos_example(self) -> str:
        """
        Generate a POS tagging example.

        Format: "The {adj} {noun} {verb} | {word}: {POS}"
        Example: "the quick fox jumps | quick: ADJ"
        """
        adj = self.rng.choice(ADJECTIVES)
        noun = self.rng.choice(NOUNS)
        verb = self.rng.choice(VERBS_THIRD)

        # Randomly choose which word to tag
        word_type = self.rng.choice(['adj', 'noun', 'verb'])

        if word_type == 'adj':
            target_word, tag = adj, 'ADJ'
        elif word_type == 'noun':
            target_word, tag = noun, 'NOUN'
        else:
            target_word, tag = verb, 'VERB'

        return f"the {adj} {noun} {verb} | {target_word}: {tag}"

    def _generate_antonym_example(self) -> str:
        """
        Generate an antonym completion example.

        Format: "{word1} is to {ant1} as {word2} is to: {ant2}"
        Example: "hot is to cold as big is to: small"
        """
        pair1 = self.rng.choice(ANTONYM_PAIRS)
        # Find a different pair
        pair2 = self.rng.choice([p for p in ANTONYM_PAIRS if p[0] != pair1[0]])

        return f"{pair1[0]} is to {pair1[1]} as {pair2[0]} is to: {pair2[1]}"

    def _generate_reorder_example(self) -> str:
        """
        Generate a word reordering example.

        Format: "{scrambled} | {correct}"
        Example: "jumps fox the quick | the quick fox jumps"
        """
        adj = self.rng.choice(ADJECTIVES)
        noun = self.rng.choice(NOUNS)
        verb = self.rng.choice(VERBS_THIRD)

        correct = f"the {adj} {noun} {verb}"
        words = [adj, noun, verb, 'the']
        self.rng.shuffle(words)
        scrambled = ' '.join(words)

        return f"{scrambled} | {correct}"

    def _generate_cloze_example(self) -> str:
        """
        Generate a cloze (fill-in-blank) example.

        Format: "the ___ {verb} {adv} | {noun}"
        Example: "the ___ barks loudly | dog"
        """
        noun = self.rng.choice(NOUNS)
        verb = self.rng.choice(VERBS_THIRD)
        adv = self.rng.choice(ADVERBS)

        return f"the ___ {verb} {adv} | {noun}"

    def _generate_sequence_example(self) -> str:
        """
        Generate an abstract sequence pattern example.

        Format: "{pattern}: {next}"
        Example: "A B A B A: B"
        """
        patterns = [
            ('A B A B A', 'B'),  # Alternating
            ('A A B A A B A A', 'B'),  # AAB repeat
            ('A B C A B C A', 'B'),  # ABC repeat
            ('A B B A B B A', 'B'),  # ABB repeat
            ('X Y X Y X', 'Y'),  # Alternating with different tokens
            ('A A A B B B A A A', 'B'),  # Three repeat
        ]
        pattern, answer = self.rng.choice(patterns)
        return f"{pattern}: {answer}"

    def _generate_example(self) -> str:
        """Generate a random example from enabled pattern types."""
        pattern_type = self.rng.choice(self.pattern_types)

        if pattern_type == 'pos':
            return self._generate_pos_example()
        elif pattern_type == 'antonym':
            return self._generate_antonym_example()
        elif pattern_type == 'reorder':
            return self._generate_reorder_example()
        elif pattern_type == 'cloze':
            return self._generate_cloze_example()
        elif pattern_type == 'sequence':
            return self._generate_sequence_example()
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized NLP patterns indefinitely."""
        while True:
            examples = []
            total_chars = 0

            # Pack multiple examples into one sequence
            while total_chars < self.seq_length * 2:
                example = self._generate_example()
                examples.append(example)
                total_chars += len(example) + 3

            text = " | ".join(examples)

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


class SVODataset(IterableDataset):
    """
    Subject-Verb-Object (SVO) pattern dataset for ID-syntactic forgetting.

    Generates SVO patterns where position determines semantic role:
    - "the cat chased the mouse | SUBJ: cat"

    Used with OVSDataset to test whether topology can distinguish
    the same words in different syntactic positions.

    Attributes:
        tokenizer: NLPTokenizer instance
        seq_length: Maximum sequence length
        rng: Random number generator
    """

    def __init__(
        self,
        tokenizer: NLPTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
    ):
        """
        Initialize SVODataset.

        Args:
            tokenizer: NLPTokenizer for encoding
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)

    def _generate_example(self) -> str:
        """Generate SVO pattern with subject identification.

        Uses past tense transitive verbs for consistency with OVS passive voice.
        Example: "the cat chased the mouse | SUBJ: cat"
        """
        # Ensure subject and object are different
        nouns = self.rng.sample(NOUNS, 2)
        subject, obj = nouns[0], nouns[1]
        # Use past tense form (index 0 of tuple)
        verb_past, _ = self.rng.choice(TRANSITIVE_VERBS)

        return f"the {subject} {verb_past} the {obj} | SUBJ: {subject}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized SVO patterns indefinitely."""
        while True:
            examples = []
            total_chars = 0

            while total_chars < self.seq_length * 2:
                example = self._generate_example()
                examples.append(example)
                total_chars += len(example) + 3

            text = " | ".join(examples)

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


class OVSDataset(IterableDataset):
    """
    Object-Verb-Subject (OVS) pattern dataset for ID-syntactic forgetting.

    TRUE OVS word order: Object-Verb-Subject
    In OVS languages (like Hixkaryana, Klingon), the sentence structure is:
    - Object (what receives the action) comes FIRST
    - Verb (the action) comes SECOND
    - Subject (who does the action) comes LAST

    Example comparison:
    - SVO (English): "The cat chased the mouse" -> cat=subject (chaser), mouse=object (chased)
    - OVS (true):    "The mouse chased the cat" -> cat=subject (chaser), mouse=object (chased)
                     In OVS, the LAST noun is the doer, FIRST noun is receiver

    This dataset uses a BY-marker construction to make OVS unambiguous:
    - "the mouse was chased by the cat | SUBJ: cat"

    The BY-marker clearly indicates the cat is the agent (subject/doer) even though
    it appears last, making this true OVS semantically valid.

    For cross-domain forgetting tests:
    - SVODataset: First noun = subject (doer)
    - OVSDataset: Last noun = subject (doer)

    The model must learn that position determines role differently in each dataset.

    Attributes:
        tokenizer: NLPTokenizer instance
        seq_length: Maximum sequence length
        rng: Random number generator
    """

    def __init__(
        self,
        tokenizer: NLPTokenizer,
        seq_length: int = 128,
        seed: Optional[int] = None,
    ):
        """
        Initialize OVSDataset.

        Args:
            tokenizer: NLPTokenizer for encoding
            seq_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.rng = random.Random(seed)

    def _generate_example(self) -> str:
        """
        Generate true OVS pattern with subject identification.

        Uses passive voice construction "X was VERBed by Y" which naturally
        puts the object first and subject last, making it semantically
        unambiguous OVS.

        Returns:
            OVS sentence with subject label, e.g.:
            "the mouse was chased by the cat | SUBJ: cat"
            (The cat is doing the chasing, the mouse is being chased)
        """
        nouns = self.rng.sample(NOUNS, 2)
        subject, obj = nouns[0], nouns[1]
        # Use past participle form (index 1 of tuple) for passive voice
        _, verb_participle = self.rng.choice(TRANSITIVE_VERBS)

        # True OVS: Object first, verb, subject last
        # "the mouse was chased by the cat" = cat is the chaser (subject)
        # Position: [Object] [Verb] [Subject]
        return f"the {obj} was {verb_participle} by the {subject} | SUBJ: {subject}"

    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized OVS patterns indefinitely."""
        while True:
            examples = []
            total_chars = 0

            while total_chars < self.seq_length * 2:
                example = self._generate_example()
                examples.append(example)
                total_chars += len(example) + 3

            text = " | ".join(examples)

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


def generate_fixed_nlp_eval_set(
    n: int = 500,
    seed: int = 42,
    pattern_types: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """
    Generate fixed NLP evaluation examples.

    Args:
        n: Number of examples to generate
        seed: Random seed for reproducibility
        pattern_types: Which pattern types to include

    Returns:
        List of (input_prompt, expected_answer) tuples
    """
    rng = random.Random(seed)
    pattern_types = pattern_types or SyntheticNLPDataset.PATTERN_TYPES
    examples = []

    for _ in range(n):
        pattern_type = rng.choice(pattern_types)

        if pattern_type == 'pos':
            adj = rng.choice(ADJECTIVES)
            noun = rng.choice(NOUNS)
            verb = rng.choice(VERBS_THIRD)
            word_type = rng.choice(['adj', 'noun', 'verb'])

            if word_type == 'adj':
                target, answer = adj, 'ADJ'
            elif word_type == 'noun':
                target, answer = noun, 'NOUN'
            else:
                target, answer = verb, 'VERB'

            prompt = f"the {adj} {noun} {verb} | {target}:"
            examples.append((prompt, answer))

        elif pattern_type == 'antonym':
            pair1 = rng.choice(ANTONYM_PAIRS)
            pair2 = rng.choice([p for p in ANTONYM_PAIRS if p[0] != pair1[0]])
            prompt = f"{pair1[0]} is to {pair1[1]} as {pair2[0]} is to:"
            examples.append((prompt, pair2[1]))

        elif pattern_type == 'cloze':
            noun = rng.choice(NOUNS)
            verb = rng.choice(VERBS_THIRD)
            adv = rng.choice(ADVERBS)
            prompt = f"the ___ {verb} {adv} |"
            examples.append((prompt, noun))

        elif pattern_type == 'sequence':
            patterns = [
                ('A B A B A:', 'B'),
                ('A A B A A B A A:', 'B'),
                ('X Y X Y X:', 'Y'),
            ]
            prompt, answer = rng.choice(patterns)
            examples.append((prompt, answer))

    return examples
