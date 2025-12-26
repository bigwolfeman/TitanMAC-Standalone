"""
Experiment Configuration Definitions

Provides ExperimentConfig dataclass and predefined configurations for
the CMS block-sparse benchmark suite.

Configuration Hierarchy (from benchmarks.md):
1. Phase 0: Dense baselines (required for comparison)
2. Phase 1-3: Correctness, performance, topology dynamics
3. Phase 4-5: Forgetting experiments (OOD and ID)
4. Phase 6: NLP benchmarks

Difficulty Gradient:
| Rank | Experiment | Type | Token Overlap | Expected Forgetting |
|------|------------|------|---------------|---------------------|
| 1 | Math -> NLP-Synth | OOD | 0% | < 15% |
| 2 | SVO -> OVS | ID-Syntactic | 100% | 15-35% |
| 3 | Standard -> Modular | ID-Semantic | 100% | 20-40% |
| 4 | MODE:STD -> MODE:MOD7 | ID-Context | 99% | 25-50% |
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentConfig:
    """
    Configuration for a single forgetting experiment.

    Defines the tasks, training parameters, and success criteria for
    continual learning experiments.

    Attributes:
        name: Experiment identifier
        description: Human-readable description
        experiment_type: "OOD", "ID-Semantic", "ID-Syntactic", or "ID-Context"

        # Task definitions
        task_a_type: Dataset type for Task A
        task_b_type: Dataset type for Task B
        task_a_params: Additional params for Task A dataset
        task_b_params: Additional params for Task B dataset

        # Training parameters
        steps_a: Training steps for Task A
        steps_b: Training steps for Task B
        batch_size: Batch size for training
        seq_length: Sequence length

        # Model parameters
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        vocab_size: Vocabulary size

        # Optimizer parameters
        optimizer: "adam", "deep_nested", or "continuum"
        base_lr: Base learning rate
        meta_lr: Meta learning rate (for nested optimizers)

        # Block-sparse parameters
        density: Target density (0.0-1.0)
        block_size: Block size for sparse layers

        # Success criteria
        expected_forgetting_min: Minimum expected forgetting (%)
        expected_forgetting_max: Maximum expected forgetting (%)
        target_forgetting: Target forgetting to achieve (%)
        blocking: Whether this is a blocking test

        # Evaluation
        n_eval: Number of evaluation examples
        eval_seed: Random seed for eval set
    """

    # Identity
    name: str
    description: str
    experiment_type: str  # OOD, ID-Semantic, ID-Syntactic, ID-Context

    # Task definitions
    task_a_type: str = "math"  # math, nlp, svo, mode_std
    task_b_type: str = "nlp"   # nlp, modular, ovs, mode_mod7
    task_a_params: Dict[str, Any] = field(default_factory=dict)
    task_b_params: Dict[str, Any] = field(default_factory=dict)

    # Training parameters
    steps_a: int = 10000
    steps_b: int = 10000
    batch_size: int = 8
    seq_length: int = 256
    log_interval: int = 100

    # Model parameters (defaults are hardware-aligned for tensor cores)
    # d_model and d_ff should be multiples of 16 for WMMA compatibility
    d_model: int = 640   # 40 × 16 - aligned for tensor cores
    n_heads: int = 10    # head_dim = 64
    n_layers: int = 12   # ~60M params with d_model=640
    d_ff: int = 2560     # 4x d_model, 160 × 16
    vocab_size: int = 89  # Combined math + NLP vocab

    # Optimizer parameters
    optimizer: str = "deep_nested"
    base_lr: float = 3e-4
    meta_lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Block-sparse parameters
    use_sparse: bool = True
    density: float = 0.5
    block_size: int = 16

    # Success criteria
    expected_forgetting_min: float = 0.0
    expected_forgetting_max: float = 100.0
    target_forgetting: float = 30.0
    blocking: bool = False

    # Evaluation
    n_eval: int = 500
    eval_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type,
            "task_a_type": self.task_a_type,
            "task_b_type": self.task_b_type,
            "task_a_params": self.task_a_params,
            "task_b_params": self.task_b_params,
            "steps_a": self.steps_a,
            "steps_b": self.steps_b,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "optimizer": self.optimizer,
            "base_lr": self.base_lr,
            "meta_lr": self.meta_lr,
            "use_sparse": self.use_sparse,
            "density": self.density,
            "block_size": self.block_size,
            "target_forgetting": self.target_forgetting,
            "blocking": self.blocking,
        }


# =============================================================================
# Predefined Experiment Configurations
# =============================================================================

OOD_MATH_NLP = ExperimentConfig(
    name="ood_math_nlp",
    description=(
        "Out-of-distribution forgetting: Train on math, then NLP. "
        "Different vocabularies make this the easiest test - topology "
        "can trivially separate by activating different input columns."
    ),
    experiment_type="OOD",

    # Tasks
    task_a_type="math",
    task_b_type="nlp",
    task_a_params={"max_num": 99, "include_large": True},
    task_b_params={"pattern_types": ["pos", "antonym", "cloze", "sequence"]},

    # Training
    steps_a=10000,
    steps_b=10000,
    batch_size=8,
    seq_length=256,

    # Model
    vocab_size=89,  # math (19) + nlp (70)

    # Sparse
    use_sparse=True,
    density=0.5,

    # Success criteria (from benchmarks.md)
    expected_forgetting_min=0.0,
    expected_forgetting_max=20.0,
    target_forgetting=15.0,
    blocking=True,  # If this fails, block-sparse has fundamental issues
)


ID_SEMANTIC_MODULAR = ExperimentConfig(
    name="id_semantic_modular",
    description=(
        "In-distribution semantic forgetting: Standard arithmetic then "
        "modular arithmetic (mod 7). Same inputs (e.g., '5 + 3 =') must "
        "produce different outputs (8 vs 1). Tests whether topology can "
        "learn to separate identical inputs based on training context."
    ),
    experiment_type="ID-Semantic",

    # Tasks
    task_a_type="math",
    task_b_type="modular",
    task_a_params={"max_num": 99},
    task_b_params={"modulus": 7, "max_num": 99},

    # Training
    steps_a=10000,
    steps_b=10000,
    batch_size=8,
    seq_length=256,

    # Model
    vocab_size=19,  # math only

    # Sparse
    use_sparse=True,
    density=0.5,

    # Success tiers (from benchmarks.md):
    # < 15%: Gold, 15-30%: Silver, 30-50%: Bronze, > 50%: Fail
    expected_forgetting_min=15.0,
    expected_forgetting_max=50.0,
    target_forgetting=30.0,  # Silver tier target
    blocking=False,  # Research goal, not blocking
)


ID_SYNTACTIC_GRAMMAR = ExperimentConfig(
    name="id_syntactic_grammar",
    description=(
        "In-distribution syntactic forgetting: SVO (subject-verb-object) "
        "then OVS (object-verb-subject) patterns. Same tokens but different "
        "positions determine semantic roles. Tests position-awareness."
    ),
    experiment_type="ID-Syntactic",

    # Tasks
    task_a_type="svo",
    task_b_type="ovs",
    task_a_params={},
    task_b_params={},

    # Training
    steps_a=10000,
    steps_b=10000,
    batch_size=8,
    seq_length=256,

    # Model
    vocab_size=89,  # Combined vocab for NLP

    # Sparse
    use_sparse=True,
    density=0.5,

    # Success criteria
    expected_forgetting_min=10.0,
    expected_forgetting_max=40.0,
    target_forgetting=30.0,
    blocking=False,
)


ID_CONTEXT_MODE = ExperimentConfig(
    name="id_context_mode",
    description=(
        "In-distribution context-switched forgetting: MODE:STD then MODE:MOD7 "
        "prefixed math problems. A SINGLE prefix token must route to entirely "
        "different computation. This is the HARDEST test - requires topology "
        "to be mode-aware, not just input-aware."
    ),
    experiment_type="ID-Context",

    # Tasks
    task_a_type="mode_std",
    task_b_type="mode_mod7",
    task_a_params={"mode": "STD", "max_num": 99},
    task_b_params={"mode": "MOD7", "modulus": 7, "max_num": 99},

    # Training
    steps_a=10000,
    steps_b=10000,
    batch_size=8,
    seq_length=256,

    # Model - needs extended vocab for MODE tokens
    vocab_size=50,  # Extended math vocab with MODE tokens

    # Sparse
    use_sparse=True,
    density=0.5,

    # Success criteria (hardest - may not be achievable)
    expected_forgetting_min=20.0,
    expected_forgetting_max=60.0,
    target_forgetting=50.0,
    blocking=False,  # Stretch goal
)


# Registry of all configs
_CONFIG_REGISTRY = {
    "ood_math_nlp": OOD_MATH_NLP,
    "id_semantic_modular": ID_SEMANTIC_MODULAR,
    "id_syntactic_grammar": ID_SYNTACTIC_GRAMMAR,
    "id_context_mode": ID_CONTEXT_MODE,
}


def get_config(name: str) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        name: Configuration name (e.g., "ood_math_nlp")

    Returns:
        ExperimentConfig instance

    Raises:
        KeyError: If config name not found
    """
    if name not in _CONFIG_REGISTRY:
        available = ", ".join(_CONFIG_REGISTRY.keys())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return _CONFIG_REGISTRY[name]


def list_configs() -> List[str]:
    """
    List all available configuration names.

    Returns:
        List of config names
    """
    return list(_CONFIG_REGISTRY.keys())


def create_baseline_config(
    name: str,
    use_sparse: bool = False,
    use_ewc: bool = False,
    **overrides
) -> ExperimentConfig:
    """
    Create a baseline variant of an existing config.

    Args:
        name: Base config name
        use_sparse: Whether to use sparse layers
        use_ewc: Whether to use EWC regularization
        **overrides: Additional parameter overrides

    Returns:
        New ExperimentConfig with modifications

    Example:
        >>> dense_config = create_baseline_config("ood_math_nlp", use_sparse=False)
        >>> ewc_config = create_baseline_config("id_semantic_modular", use_ewc=True)
    """
    base = get_config(name)

    # Create new config with overrides
    params = base.to_dict()
    params["use_sparse"] = use_sparse

    if use_ewc:
        params["name"] = f"{name}_ewc"
        params["description"] = f"EWC baseline: {base.description}"

    params.update(overrides)

    return ExperimentConfig(**params)
