"""
Titan-MAC configuration dataclass.

This module provides configuration for the Titan-MAC architecture with
nested optimizer integration. Supports full Titans paper (arxiv 2501.00663)
and Nested Learning (NeurIPS 2025) features.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Valid Titans architecture variants
TITANS_VARIANTS = ("MAC", "MAG", "MAL")


@dataclass
class TitanMACConfig:
    """
    Configuration for Titan-MAC model architecture.

    Architecture:
        - Titan backbone with windowed attention and persistent tokens
        - Memory-Augmented Context (MAC) dataflow
        - Nested optimizer for test-time adaptation
        - Neural Long-Term Memory (Titans paper)
        - Deep Momentum Gradient Descent (Nested Learning)
        - Continuum Memory System (Nested Learning)

    Args:
        vocab_size: Vocabulary size for embeddings
        d_model: Model dimension (must equal n_heads * d_head)
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension (typically 4x d_model)
        max_seq_len: Maximum sequence length
        window_size: Local attention window size
        n_persistent: Number of persistent (global) tokens
        dropout: Dropout probability
        tie_weights: Whether to tie input/output embeddings
        use_memory_bank: Enable MemoryBank for MAC retrieval
        enable_test_time_memory: Enable test-time memory adaptation
        memory_mlp_expansion: MLP expansion factor in NeuralMemory
        memory_lambda_decay: Fast-weight decay rate (λ in W ← λW + ηek^T)
        memory_eta_lr: Fast-weight learning rate (η in W ← λW + ηek^T)
        memory_rank_ratio: Low-rank ratio for SVD (r = d_head * rank_ratio)

        # Titans Neural Memory (arxiv 2501.00663)
        use_neural_memory: Enable gradient-based neural long-term memory
        memory_capacity: Number of memory slots for NeuralMemory
        d_memory: Memory dimension (defaults to d_model if None)
        memory_theta_lr: Learning rate for memory updates (θ_t)
        memory_forget_hidden: Hidden dim for forget gate MLP
        memory_decay_hidden: Hidden dim for decay gate MLP

        # Titans Architecture Variant
        titans_variant: Architecture variant ("MAC", "MAG", "MAL")
        use_parallel_memory: Use parallel memory update for efficiency

        # Nested Learning: DMGD
        use_dmgd: Enable Deep Momentum Gradient Descent
        dmgd_hidden_dim: Hidden dimension for MomentumMLP

        # Nested Learning: CMS
        use_cms: Enable Continuum Memory System
        cms_frequencies: Per-group update frequencies

    Example:
        >>> config = TitanMACConfig(
        ...     vocab_size=128000,
        ...     d_model=640,
        ...     n_heads=10,
        ...     n_layers=16,
        ...     use_neural_memory=True,
        ...     titans_variant="MAC"
        ... )
        >>> config.validate()
    """

    # Model architecture
    vocab_size: int = 128000
    d_model: int = 640
    n_heads: int = 10
    n_layers: int = 16
    d_ff: int = 2560
    max_seq_len: int = 4096

    # Attention configuration
    window_size: int = 512
    n_persistent: int = 16
    use_block_sparse: bool = True  # Use O(T*w) block-sparse attention (paper-faithful)
    block_size: int = 64  # Block size for block-sparse attention

    # Regularization
    dropout: float = 0.0

    # Embeddings
    tie_weights: bool = True

    # Memory configuration
    use_memory_bank: bool = False
    enable_test_time_memory: bool = False

    # NeuralMemory hyperparameters (if enabled)
    memory_mlp_expansion: int = 4
    memory_lambda_decay: float = 0.95
    memory_eta_lr: float = 0.01
    memory_rank_ratio: float = 0.25

    # Normalization
    norm_eps: float = 1e-5

    # Attention dropout
    attention_dropout: float = 0.0

    # Causal masking
    causal: bool = True

    # =========================================================================
    # Titans Neural Memory (arxiv 2501.00663, Section 3.1)
    # PAPER-FAITHFUL: Memory M is a deep MLP, not an embedding table.
    # The MLP weights ARE the memory, updated via GD at test time.
    # =========================================================================
    use_neural_memory: bool = False  # Enable gradient-based neural long-term memory
    n_memory_layers: int = 2  # Number of layers in memory MLP (paper: L_M >= 2)
    d_memory: Optional[int] = None  # Memory dimension (defaults to d_model if None)
    memory_theta_lr: float = 0.01  # Learning rate for memory updates (θ_t in S_t)
    memory_forget_hidden: int = 32  # Hidden dim for forget gate MLP (α_t)
    memory_decay_hidden: int = 32  # Hidden dim for decay gate MLP (η_t)
    # Legacy parameter - kept for compatibility but ignored
    memory_capacity: int = 512  # DEPRECATED: MLP memory doesn't use capacity

    # =========================================================================
    # Titans Architecture Variant (Section 4)
    # =========================================================================
    titans_variant: str = "MAC"  # Architecture variant: "MAC", "MAG", or "MAL"
    use_parallel_memory: bool = True  # Use parallel memory update (Eq. 11-13)

    # =========================================================================
    # MAC Segment Processing (Section 4.1)
    # PAPER-FAITHFUL: Process sequence in segments with fixed N_l memory tokens
    # This ensures memory tokens are reachable within the attention window.
    # =========================================================================
    segment_size: int = 512  # Size of each segment for MAC processing
    n_memory_tokens: int = 32  # Number of memory tokens retrieved per segment (N_l)

    # =========================================================================
    # Nested Learning: Deep Momentum Gradient Descent (DMGD)
    # =========================================================================
    use_dmgd: bool = False  # Enable learned momentum MLP
    dmgd_hidden_dim: int = 64  # Hidden dimension for MomentumMLP
    dmgd_n_groups: int = 2  # Number of parameter groups for DMGD

    # =========================================================================
    # Nested Learning: Continuum Memory System (CMS)
    # =========================================================================
    use_cms: bool = False  # Enable multi-frequency parameter updates
    cms_frequencies: Optional[Dict[int, int]] = None  # Per-group update frequencies

    @property
    def d_head(self) -> int:
        """Compute head dimension from d_model and n_heads."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @classmethod
    def from_cfg(cls, cfg: Any) -> "TitanMACConfig":
        """
        Create TitanMACConfig from legacy config object.

        Args:
            cfg: Legacy config object (dict-like or object with attributes)

        Returns:
            TitanMACConfig instance

        Example:
            >>> cfg = {"vocab_size": 50000, "d_model": 512, "n_heads": 8}
            >>> config = TitanMACConfig.from_cfg(cfg)
        """
        # Handle dict-like configs
        if isinstance(cfg, dict):
            # Filter to only valid fields
            valid_fields = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
            return cls(**valid_fields)

        # Handle object configs (with attributes)
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(cfg, field_name):
                kwargs[field_name] = getattr(cfg, field_name)

        return cls(**kwargs)

    def validate(self) -> None:
        """
        Validate configuration constraints.

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = TitanMACConfig(d_model=640, n_heads=10)
            >>> config.validate()  # OK
            >>> bad_config = TitanMACConfig(d_model=641, n_heads=10)
            >>> bad_config.validate()  # Raises ValueError
        """
        # Check d_model divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Check persistent tokens
        if self.n_persistent >= self.max_seq_len:
            raise ValueError(
                f"n_persistent ({self.n_persistent}) must be < max_seq_len ({self.max_seq_len})"
            )

        # Check window size
        if self.window_size < 0:
            raise ValueError(f"window_size must be non-negative, got {self.window_size}")

        # Check dimensions
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")

        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

        # Check dropout bounds
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")

        # Check memory hyperparameters
        if not 0.0 < self.memory_lambda_decay < 1.0:
            raise ValueError(
                f"memory_lambda_decay must be in (0, 1), got {self.memory_lambda_decay}"
            )

        if self.memory_eta_lr <= 0:
            raise ValueError(f"memory_eta_lr must be positive, got {self.memory_eta_lr}")

        if not 0.0 < self.memory_rank_ratio <= 1.0:
            raise ValueError(
                f"memory_rank_ratio must be in (0, 1], got {self.memory_rank_ratio}"
            )

        # =====================================================================
        # Validate Titans Neural Memory parameters
        # =====================================================================
        if self.use_neural_memory:
            if self.n_memory_layers < 1:
                raise ValueError(
                    f"n_memory_layers must be >= 1, got {self.n_memory_layers}"
                )
            if self.n_memory_layers < 2:
                import warnings
                warnings.warn(
                    "Paper recommends n_memory_layers >= 2 for better expressivity. "
                    f"Got {self.n_memory_layers}."
                )
            if self.d_memory is not None and self.d_memory <= 0:
                raise ValueError(f"d_memory must be positive, got {self.d_memory}")
            if self.memory_theta_lr <= 0:
                raise ValueError(
                    f"memory_theta_lr must be positive, got {self.memory_theta_lr}"
                )
            if self.memory_forget_hidden <= 0:
                raise ValueError(
                    f"memory_forget_hidden must be positive, got {self.memory_forget_hidden}"
                )
            if self.memory_decay_hidden <= 0:
                raise ValueError(
                    f"memory_decay_hidden must be positive, got {self.memory_decay_hidden}"
                )

        # =====================================================================
        # Validate Titans variant
        # =====================================================================
        if self.titans_variant not in TITANS_VARIANTS:
            raise ValueError(
                f"titans_variant must be one of {TITANS_VARIANTS}, got '{self.titans_variant}'"
            )

        # =====================================================================
        # Validate DMGD parameters
        # =====================================================================
        if self.use_dmgd:
            if self.dmgd_hidden_dim <= 0:
                raise ValueError(
                    f"dmgd_hidden_dim must be positive, got {self.dmgd_hidden_dim}"
                )
            if self.dmgd_n_groups <= 0:
                raise ValueError(
                    f"dmgd_n_groups must be positive, got {self.dmgd_n_groups}"
                )

        # =====================================================================
        # Validate CMS parameters
        # =====================================================================
        if self.use_cms:
            if self.cms_frequencies is not None:
                for group_idx, freq in self.cms_frequencies.items():
                    if not isinstance(group_idx, int) or group_idx < 0:
                        raise ValueError(
                            f"CMS group index must be non-negative int, got {group_idx}"
                        )
                    if not isinstance(freq, int) or freq <= 0:
                        raise ValueError(
                            f"CMS frequency must be positive int, got {freq} for group {group_idx}"
                        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            >>> config = TitanMACConfig()
            >>> config_dict = config.to_dict()
            >>> isinstance(config_dict, dict)
            True
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    def __repr__(self) -> str:
        """Custom repr showing key parameters."""
        parts = [
            f"d_model={self.d_model}",
            f"n_heads={self.n_heads}",
            f"n_layers={self.n_layers}",
            f"d_ff={self.d_ff}",
            f"window_size={self.window_size}",
            f"n_persistent={self.n_persistent}",
            f"titans_variant='{self.titans_variant}'",
        ]
        if self.use_neural_memory:
            parts.append(f"n_memory_layers={self.n_memory_layers}")
        if self.use_dmgd:
            parts.append("use_dmgd=True")
        if self.use_cms:
            parts.append("use_cms=True")
        return f"TitanMACConfig({', '.join(parts)})"
