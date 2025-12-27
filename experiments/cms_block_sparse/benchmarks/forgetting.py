"""
Forgetting Benchmarks for CMS Block-Sparse Validation

This module provides tools to measure catastrophic forgetting in continual
learning scenarios, with specific support for:

1. OOD Forgetting: Different vocabularies between tasks (easy baseline)
2. ID-Semantic Forgetting: Same tokens, different meanings (hard test)
3. ID-Syntactic Forgetting: Same tokens, different positions (medium test)
4. ID-Context Forgetting: Mode-switched by prefix token (hardest test)

Key Functions:
- compute_pathway_overlap(): Measures block overlap between tasks
- compute_topology_divergence(): Measures how much topology changed

Key Classes:
- ForgettingBenchmark: OOD forgetting with fixed eval sets
- IDForgettingBenchmark: ID forgetting with shared inputs, different targets
"""

import random
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ForgettingResult:
    """Container for forgetting measurement results."""

    accuracy_before: float
    accuracy_after: float
    forgetting_pct: float  # (before - after) / before * 100
    retention_pct: float  # after / before * 100

    # Optional topology metrics
    pathway_overlap: Optional[float] = None
    topology_divergence: Optional[float] = None

    def __str__(self) -> str:
        s = f"Forgetting: {self.forgetting_pct:.1f}% | Retention: {self.retention_pct:.1f}%"
        s += f"\n  Accuracy: {self.accuracy_before:.3f} -> {self.accuracy_after:.3f}"
        if self.pathway_overlap is not None:
            s += f"\n  Pathway overlap: {self.pathway_overlap:.3f}"
        if self.topology_divergence is not None:
            s += f"\n  Topology divergence: {self.topology_divergence:.3f}"
        return s


def compute_pathway_overlap(
    blocks_active_A: Set[Tuple[int, int]],
    blocks_active_B: Set[Tuple[int, int]],
) -> float:
    """
    Compute Jaccard overlap between active block sets from two tasks.

    Used to measure whether Task A and Task B use distinct pathways.
    Lower overlap suggests better pathway separation (less forgetting risk).

    Args:
        blocks_active_A: Set of (layer, block_idx) tuples active for Task A
        blocks_active_B: Set of (layer, block_idx) tuples active for Task B

    Returns:
        Overlap ratio in [0, 1]. 0 = no overlap, 1 = identical pathways

    Example:
        >>> blocks_A = {(0, 1), (0, 2), (1, 5)}
        >>> blocks_B = {(0, 2), (0, 3), (1, 6)}
        >>> overlap = compute_pathway_overlap(blocks_A, blocks_B)
        >>> print(f"Overlap: {overlap:.2f}")  # ~0.2 (1 shared out of 5 unique)

    Target Values (from benchmarks.md):
        - < 70%: Distinct pathways for distinct tasks (good)
        - > 90%: Tasks using same pathways (forgetting likely)
    """
    if not blocks_active_A and not blocks_active_B:
        return 0.0

    intersection = blocks_active_A & blocks_active_B
    union = blocks_active_A | blocks_active_B

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compute_topology_divergence(
    topology_A: torch.Tensor,
    topology_B: torch.Tensor,
) -> float:
    """
    Compute how different the topology is after Task B vs Task A.

    For ID forgetting tests, high divergence on SAME inputs indicates
    the model learned to use separate pathways (desirable).

    Args:
        topology_A: Column indices tensor after Task A [R, K] or [L, R, K]
        topology_B: Column indices tensor after Task B [R, K] or [L, R, K]

    Returns:
        Divergence ratio in [0, 1]. 0 = identical, 1 = completely different

    Example:
        >>> topo_A = torch.tensor([[0, 5, 12], [1, 8, 15]])
        >>> topo_B = torch.tensor([[0, 6, 12], [2, 8, 16]])
        >>> div = compute_topology_divergence(topo_A, topo_B)
        >>> print(f"Divergence: {div:.2f}")  # ~0.4 (some blocks changed)

    Target Values:
        - > 0.3: At least 30% of blocks differ (healthy separation)
        - < 0.1: Topology didn't adapt to new task (stagnation)
    """
    # Flatten to sets for comparison
    set_A = set(topology_A.flatten().tolist())
    set_B = set(topology_B.flatten().tolist())

    if not set_A and not set_B:
        return 0.0

    intersection = set_A & set_B
    union = set_A | set_B

    if not union:
        return 0.0

    overlap = len(intersection) / len(union)
    return 1.0 - overlap


class ForgettingBenchmark:
    """
    Base class for measuring catastrophic forgetting.

    Provides evaluation protocol for OOD (out-of-distribution) forgetting:
    - Train on Task A, evaluate, record accuracy_A_before
    - Train on Task B
    - Re-evaluate on Task A, record accuracy_A_after
    - Compute forgetting = (before - after) / before

    Uses fixed evaluation sets (seed=42) for reproducibility.

    Attributes:
        tokenizer: Tokenizer for encoding eval inputs
        eval_inputs: Fixed list of input prompts
        eval_targets: Expected outputs for each prompt
        device: Torch device for evaluation
    """

    def __init__(
        self,
        tokenizer,
        n_eval: int = 500,
        seed: int = 42,
        device: str = "cuda",
        task_type: str = "math",
    ):
        """
        Initialize ForgettingBenchmark.

        Args:
            tokenizer: Tokenizer for encoding evaluation inputs
            n_eval: Number of evaluation examples
            seed: Random seed for reproducibility (default 42)
            device: Device for evaluation ("cuda" or "cpu")
            task_type: Type of task ("math" or "nlp")
        """
        self.tokenizer = tokenizer
        self.device = device
        self.n_eval = n_eval
        self.seed = seed
        self.task_type = task_type

        # Generate fixed eval set based on task type
        self.eval_inputs, self.eval_targets = self._generate_eval_set(n_eval, seed)

    # Word lists for SVO evaluation (must match nlp_datasets.py)
    SVO_NOUNS = ['dog', 'cat', 'fox', 'bird', 'mouse', 'bear', 'fish', 'frog', 'wolf', 'deer']
    SVO_VERBS_PAST = ['chased', 'followed', 'watched', 'helped', 'found', 'caught', 'saw', 'heard']

    def _generate_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed evaluation inputs and targets.

        Override in subclasses for specific task types.

        Args:
            n: Number of examples
            seed: Random seed

        Returns:
            Tuple of (input_prompts, expected_outputs)
        """
        if self.task_type == "nlp":
            return self._generate_nlp_eval_set(n, seed)
        elif self.task_type == "svo":
            return self._generate_svo_eval_set(n, seed)
        elif self.task_type == "ovs":
            return self._generate_ovs_eval_set(n, seed)
        elif self.task_type == "mode_std":
            return self._generate_mode_std_eval_set(n, seed)
        elif self.task_type == "mode_mod7":
            return self._generate_mode_mod7_eval_set(n, seed)
        else:
            return self._generate_math_eval_set(n, seed)

    def _generate_svo_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed SVO evaluation set.

        Format: "the {subject} {verb_past} the {obj} | SUBJ:" -> "{subject}"
        """
        rng = random.Random(seed)
        inputs = []
        targets = []

        for _ in range(n):
            subject = rng.choice(self.SVO_NOUNS)
            obj = rng.choice([noun for noun in self.SVO_NOUNS if noun != subject])
            verb = rng.choice(self.SVO_VERBS_PAST)

            inputs.append(f"the {subject} {verb} the {obj} | SUBJ:")
            targets.append(subject)

        return inputs, targets

    def _generate_ovs_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed OVS (passive voice) evaluation set.

        Format: "the {obj} was {verb_participle} by the {subject} | SUBJ:" -> "{subject}"
        """
        rng = random.Random(seed)
        inputs = []
        targets = []

        # Past participle forms (most are same as past tense)
        verbs_participle = ['chased', 'followed', 'watched', 'helped', 'found', 'caught', 'seen', 'heard']

        for _ in range(n):
            subject = rng.choice(self.SVO_NOUNS)
            obj = rng.choice([noun for noun in self.SVO_NOUNS if noun != subject])
            verb_participle = rng.choice(verbs_participle)

            inputs.append(f"the {obj} was {verb_participle} by the {subject} | SUBJ:")
            targets.append(subject)

        return inputs, targets

    def _generate_math_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """Generate fixed math evaluation set."""
        rng = random.Random(seed)
        inputs = []
        targets = []

        for _ in range(n):
            op = rng.choice(['+', '-', '*'])
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)

            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:
                result = a * b

            inputs.append(f"{a} {op} {b} =")
            targets.append(str(result))

        return inputs, targets

    def _generate_nlp_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed NLP evaluation set.

        Uses pattern types that can be evaluated deterministically:
        - antonym: "hot is to cold as big is to:" -> "small"
        - sequence: "A B A B A:" -> "B"
        """
        rng = random.Random(seed)
        inputs = []
        targets = []

        # Antonym pairs for evaluation
        antonym_pairs = [
            ('hot', 'cold'), ('cold', 'hot'),
            ('big', 'small'), ('small', 'big'),
            ('quick', 'slow'), ('slow', 'quick'),
            ('loud', 'soft'), ('soft', 'loud'),
            ('old', 'young'), ('young', 'old'),
        ]

        # Sequence patterns for evaluation
        sequence_patterns = [
            ('A B A B A:', 'B'),
            ('A A B A A B A A:', 'B'),
            ('X Y X Y X:', 'Y'),
        ]

        for i in range(n):
            # Alternate between antonyms and sequences
            if i % 2 == 0:
                # Antonym pattern
                pair1 = rng.choice(antonym_pairs)
                pair2 = rng.choice([p for p in antonym_pairs if p[0] != pair1[0]])
                prompt = f"{pair1[0]} is to {pair1[1]} as {pair2[0]} is to:"
                inputs.append(prompt)
                targets.append(pair2[1])
            else:
                # Sequence pattern
                prompt, answer = rng.choice(sequence_patterns)
                inputs.append(prompt)
                targets.append(answer)

        return inputs, targets

    def _generate_mode_std_eval_set(
        self,
        n: int,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed MODE:STD evaluation set.

        Format matches ContextSwitchedMathDataset:
        "MODE:STD | 5 + 3 =" -> "8"
        """
        rng = random.Random(seed)
        inputs = []
        targets = []

        for _ in range(n):
            op = rng.choice(['+', '-', '*'])
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)

            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:
                result = a * b

            inputs.append(f"MODE:STD | {a} {op} {b} =")
            targets.append(str(result))

        return inputs, targets

    def _generate_mode_mod7_eval_set(
        self,
        n: int,
        seed: int,
        modulus: int = 7
    ) -> Tuple[List[str], List[str]]:
        """
        Generate fixed MODE:MOD7 evaluation set.

        Format matches ContextSwitchedMathDataset with mode=MOD7:
        "MODE:MOD7 | 5 + 3 =" -> "1" (8 mod 7)
        """
        rng = random.Random(seed)
        inputs = []
        targets = []

        for _ in range(n):
            op = rng.choice(['+', '-', '*'])
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)

            if op == '+':
                raw_result = a + b
            elif op == '-':
                raw_result = a - b
            else:
                raw_result = a * b

            result = raw_result % modulus

            inputs.append(f"MODE:MOD7 | {a} {op} {b} =")
            targets.append(str(result))

        return inputs, targets

    def evaluate(
        self,
        model: nn.Module,
        max_new_tokens: int = 10,
    ) -> float:
        """
        Evaluate model accuracy on the fixed eval set.

        Args:
            model: Model to evaluate
            max_new_tokens: Maximum tokens to generate per example

        Returns:
            Accuracy as fraction in [0, 1]
        """
        model.eval()
        correct = 0

        with torch.no_grad():
            for input_text, target in zip(self.eval_inputs, self.eval_targets):
                # Encode input WITHOUT padding - critical for generation
                # Padding would cause the model to generate from the wrong position
                input_ids = self.tokenizer.encode(input_text)
                input_ids = torch.tensor([input_ids], device=self.device)
                input_len = input_ids.shape[1]

                # Generate output
                output_ids = self._generate(model, input_ids, max_new_tokens)

                # Decode only the generated part (after input)
                output_text = self.tokenizer.decode(
                    output_ids[0, input_len:].tolist(),
                    skip_special_tokens=True
                ).strip()

                # Check if target is in output (allowing for extra tokens)
                if target in output_text or output_text.startswith(target):
                    correct += 1

        return correct / len(self.eval_inputs)

    def _generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate tokens from model (simple greedy decoding).

        Args:
            model: Model to generate from
            input_ids: Input token IDs [1, T]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Full sequence including generated tokens
        """
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated)
            logits = outputs["logits"]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS or pad
            if next_token.item() in [0, 1]:  # <pad> or <eos>
                break

        return generated

    def measure_forgetting(
        self,
        model: nn.Module,
        accuracy_before: Optional[float] = None,
    ) -> ForgettingResult:
        """
        Measure forgetting by comparing current accuracy to baseline.

        Args:
            model: Model to evaluate
            accuracy_before: Accuracy before Task B training (if known)

        Returns:
            ForgettingResult with all metrics
        """
        accuracy_after = self.evaluate(model)

        if accuracy_before is None:
            # Can't compute forgetting without baseline
            return ForgettingResult(
                accuracy_before=0.0,
                accuracy_after=accuracy_after,
                forgetting_pct=0.0,
                retention_pct=100.0,
            )

        if accuracy_before > 0:
            forgetting_pct = 100 * (accuracy_before - accuracy_after) / accuracy_before
            retention_pct = 100 * accuracy_after / accuracy_before
        else:
            forgetting_pct = 0.0
            retention_pct = 100.0

        return ForgettingResult(
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            forgetting_pct=forgetting_pct,
            retention_pct=retention_pct,
        )


class IDForgettingBenchmark:
    """
    In-Distribution Forgetting Benchmark for semantic, syntactic, and context forgetting tests.

    Unlike OOD forgetting, ID forgetting uses the SAME inputs for both
    tasks but expects DIFFERENT outputs. This is the harder test for
    block-sparse topology - can it learn to route identical inputs
    to different computations based on training context?

    Supports three experiment types:

    ID-Semantic (standard vs modular math):
        Task A (standard): "5 + 3 =" -> "8"
        Task B (mod 7):    "5 + 3 =" -> "1"

    ID-Syntactic (SVO vs OVS word order):
        Task A (SVO): "the cat chased the mouse |" -> "cat" (subject)
        Task B (OVS): "the cat chased the mouse |" -> "mouse" (object as subject)

    ID-Context (MODE-prefixed switching):
        Task A: "MODE:STD | 5 + 3 =" -> "8"
        Task B: "MODE:MOD | 5 + 3 =" -> "1"

    Attributes:
        shared_inputs: Fixed input prompts used for both tasks
        standard_targets: Expected outputs for Task A
        modular_targets: Expected outputs for Task B
        experiment_type: One of "ID-Semantic", "ID-Syntactic", "ID-Context"
    """

    # Word lists for syntactic experiments (must match SVO/OVS datasets)
    NOUNS = ['dog', 'cat', 'fox', 'bird', 'mouse', 'bear', 'fish', 'frog', 'wolf', 'deer']
    # Past tense and past participle forms (matching nlp_datasets.py TRANSITIVE_VERBS)
    VERBS_PAST = ['chased', 'followed', 'watched', 'helped', 'found', 'caught', 'saw', 'heard']
    VERBS_PARTICIPLE = ['chased', 'followed', 'watched', 'helped', 'found', 'caught', 'seen', 'heard']

    def __init__(
        self,
        tokenizer,
        n_eval: int = 500,
        seed: int = 42,
        modulus: int = 7,
        device: str = "cuda",
        experiment_type: str = "ID-Semantic",
    ):
        """
        Initialize IDForgettingBenchmark.

        Args:
            tokenizer: Tokenizer for encoding
            n_eval: Number of evaluation examples
            seed: Random seed for reproducibility
            modulus: Modulo value for modular arithmetic (used in ID-Semantic and ID-Context)
            device: Device for evaluation
            experiment_type: Type of ID forgetting experiment:
                - "ID-Semantic": Same math problem, standard vs modular answers
                - "ID-Syntactic": Same sentence, SVO vs OVS subject extraction
                - "ID-Context": MODE-prefixed math problems
        """
        if experiment_type not in ("ID-Semantic", "ID-Syntactic", "ID-Context"):
            raise ValueError(
                f"experiment_type must be one of 'ID-Semantic', 'ID-Syntactic', 'ID-Context', "
                f"got '{experiment_type}'"
            )

        self.tokenizer = tokenizer
        self.device = device
        self.n_eval = n_eval
        self.seed = seed
        self.modulus = modulus
        self.experiment_type = experiment_type

        # Generate shared inputs with both target types
        self._generate_shared_eval_set(n_eval, seed)

    def _generate_shared_eval_set(self, n: int, seed: int):
        """Generate shared inputs with both standard and modular targets based on experiment type."""
        rng = random.Random(seed)

        self.shared_inputs = []
        self.standard_targets = []
        self.modular_targets = []

        if self.experiment_type == "ID-Semantic":
            self._generate_semantic_eval_set(rng, n)
        elif self.experiment_type == "ID-Syntactic":
            self._generate_syntactic_eval_set(rng, n)
        elif self.experiment_type == "ID-Context":
            self._generate_context_eval_set(rng, n)

    def _generate_semantic_eval_set(self, rng: random.Random, n: int):
        """
        Generate ID-Semantic evaluation set: MODE-prefixed math problems.

        For ID-Semantic tests, we use MODE:STD prefix since:
        - Task A trains on MODE:STD only
        - Task B trains on mixed MODE:STD + MODE:MOD7
        - We measure if MODE:STD accuracy degrades after mixed training

        Example:
            Input: "MODE:STD | 5 + 3 ="
            standard_target: "8" (correct for MODE:STD)
            modular_target: "1" (what MOD7 would produce - for comparison)
        """
        for _ in range(n):
            op = rng.choice(['+', '-', '*'])
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)

            if op == '+':
                std_result = a + b
            elif op == '-':
                std_result = a - b
            else:
                std_result = a * b

            mod_result = std_result % self.modulus

            # Use MODE:STD prefix to match training data format
            self.shared_inputs.append(f"MODE:STD | {a} {op} {b} =")
            self.standard_targets.append(str(std_result))
            self.modular_targets.append(str(mod_result))

    def _generate_syntactic_eval_set(self, rng: random.Random, n: int):
        """
        Generate ID-Syntactic evaluation set: SVO vs OVS subject extraction.

        For SVO (Task A): "the cat chased the mouse | SUBJ:" -> "cat"
        For OVS (Task B): "the mouse was chased by the cat | SUBJ:" -> "cat"

        We evaluate both formats:
        - standard_targets: SVO format subject identification
        - modular_targets: OVS format (passive) subject identification

        Note: In OVS passive voice "X was VERBed by Y", Y is the subject (doer).
        """
        for _ in range(n):
            # Pick two different nouns
            subject = rng.choice(self.NOUNS)
            obj = rng.choice([n for n in self.NOUNS if n != subject])
            # Use past tense for SVO
            verb_past = rng.choice(self.VERBS_PAST)
            # Use past participle for OVS passive (most are the same form)
            verb_idx = self.VERBS_PAST.index(verb_past)
            verb_participle = self.VERBS_PARTICIPLE[verb_idx]

            # SVO format: "the {subject} {verb_past} the {obj} | SUBJ:"
            # Expected answer: subject (the first noun, who did the action)
            self.shared_inputs.append(f"the {subject} {verb_past} the {obj} | SUBJ:")
            self.standard_targets.append(subject)

            # For OVS, we still expect "subject" as the answer since that's who
            # is doing the action. The difference is the format/word order.
            # OVS passive: "the {obj} was {verb_participle} by the {subject} | SUBJ:"
            # The model must learn that in passive voice, the BY-agent is the subject.
            self.modular_targets.append(subject)

    def _generate_context_eval_set(self, rng: random.Random, n: int):
        """
        Generate ID-Context evaluation set: MODE-prefixed math problems.

        Example:
            Input: "MODE:STD | 5 + 3 ="
            standard_target: "8"
            modular_target: "1" (mod 7)

        Note: The input includes the MODE prefix, but targets differ based on
        which task the model was trained on.
        """
        for _ in range(n):
            op = rng.choice(['+', '-', '*'])
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)

            if op == '+':
                std_result = a + b
            elif op == '-':
                std_result = a - b
            else:
                std_result = a * b

            mod_result = std_result % self.modulus

            # Include MODE prefix in input - the model learns to interpret this
            self.shared_inputs.append(f"MODE:STD | {a} {op} {b} =")
            self.standard_targets.append(str(std_result))
            self.modular_targets.append(str(mod_result))

    def eval_standard(self, model: nn.Module) -> float:
        """
        Evaluate accuracy on standard arithmetic.

        Args:
            model: Model to evaluate

        Returns:
            Accuracy on standard arithmetic [0, 1]
        """
        return self._evaluate(model, self.standard_targets)

    def eval_modular(self, model: nn.Module) -> float:
        """
        Evaluate accuracy on modular arithmetic.

        Args:
            model: Model to evaluate

        Returns:
            Accuracy on modular arithmetic [0, 1]
        """
        return self._evaluate(model, self.modular_targets)

    def _evaluate(
        self,
        model: nn.Module,
        targets: List[str],
        max_new_tokens: int = 10,
    ) -> float:
        """Evaluate model on shared inputs against given targets."""
        model.eval()
        correct = 0

        with torch.no_grad():
            for input_text, target in zip(self.shared_inputs, targets):
                # Encode input WITHOUT padding - critical for generation
                # Padding would cause the model to generate from the wrong position
                input_ids = self.tokenizer.encode(input_text)
                input_ids = torch.tensor([input_ids], device=self.device)
                input_len = input_ids.shape[1]

                # Generate output
                output_ids = self._generate(model, input_ids, max_new_tokens)

                # Decode only the generated part (after input)
                output_text = self.tokenizer.decode(
                    output_ids[0, input_len:].tolist(),
                    skip_special_tokens=True
                ).strip()

                # Check if target is in output (allowing for extra tokens)
                if target in output_text or output_text.startswith(target):
                    correct += 1

        return correct / len(self.shared_inputs)

    def _generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Simple greedy generation."""
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated)
            logits = outputs["logits"]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in [0, 1]:
                break

        return generated

    def measure_id_forgetting(
        self,
        model: nn.Module,
        standard_accuracy_before: float,
        topology_snapshot_A: Optional[torch.Tensor] = None,
        topology_snapshot_B: Optional[torch.Tensor] = None,
    ) -> ForgettingResult:
        """
        Measure ID-semantic forgetting after modular training.

        Args:
            model: Model to evaluate (after Task B training)
            standard_accuracy_before: Accuracy on standard math before Task B
            topology_snapshot_A: Topology state after Task A (optional)
            topology_snapshot_B: Topology state after Task B (optional)

        Returns:
            ForgettingResult with forgetting metrics

        Success Tiers (from benchmarks.md):
            < 15%: Gold - topology successfully separates identical inputs
            15-30%: Silver - partial separation, better than dense
            30-50%: Bronze - limited benefit
            > 50%: Fail - no better than static sparse
        """
        standard_accuracy_after = self.eval_standard(model)

        if standard_accuracy_before > 0:
            forgetting_pct = 100 * (standard_accuracy_before - standard_accuracy_after) / standard_accuracy_before
            retention_pct = 100 * standard_accuracy_after / standard_accuracy_before
        else:
            forgetting_pct = 0.0
            retention_pct = 100.0

        # Compute topology divergence if snapshots provided
        topo_div = None
        if topology_snapshot_A is not None and topology_snapshot_B is not None:
            topo_div = compute_topology_divergence(topology_snapshot_A, topology_snapshot_B)

        return ForgettingResult(
            accuracy_before=standard_accuracy_before,
            accuracy_after=standard_accuracy_after,
            forgetting_pct=forgetting_pct,
            retention_pct=retention_pct,
            topology_divergence=topo_div,
        )


def record_block_activations(
    model: nn.Module,
    inputs: List[str],
    tokenizer,
    device: str = "cuda",
    threshold: float = 0.1,
) -> Set[Tuple[int, int]]:
    """
    Record which blocks are highly active for given inputs.

    Used to compute pathway overlap between tasks.

    Args:
        model: Model with block-sparse layers
        inputs: Input text strings
        tokenizer: Tokenizer for encoding
        device: Device for computation
        threshold: Activation threshold for "active" classification

    Returns:
        Set of (layer_idx, block_idx) tuples for active blocks
    """
    active_blocks = set()

    # This is a placeholder - actual implementation depends on
    # how BlockSparseLinear exposes activation information
    # The real implementation would:
    # 1. Register forward hooks on sparse layers
    # 2. Accumulate block activation magnitudes
    # 3. Mark blocks above threshold as active

    # TODO: Implement when BlockSparseLinear is available
    # For now, return empty set

    return active_blocks


def snapshot_topology(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Capture current topology state from all sparse layers.

    Args:
        model: Model with block-sparse layers

    Returns:
        Dict mapping layer names to col_indices tensors
    """
    topology = {}

    for name, module in model.named_modules():
        # Check if this is a block-sparse layer
        if hasattr(module, 'col_indices'):
            topology[name] = module.col_indices.clone()

    return topology
