"""
Neural Long-Term Memory module from Titans paper (arxiv 2501.00663).

PAPER-FAITHFUL IMPLEMENTATION (Refactored 2025-12-14)

Key insight from paper: Memory M is a DEEP MLP, not an embedding table.
The MLP weights ARE the memory. Test-time learning updates MLP weights via GD.

Key equations from Section 3.1:
    - M(k): MLP forward pass (Eq. 15: y_t = M*(q_t))
    - Associative Loss: l(M; x) = ||M(k) - v||^2  (Eq. 12)
    - Momentum: S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)  (Eq. 13)
    - Memory Update: M_t = (1 - α_t) * M_{t-1} + S_t  (Eq. 14)

Where M_t refers to ALL MLP WEIGHTS at time t.

Version: 2.0.0 (Paper-Faithful)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


class ForgetGate(nn.Module):
    """
    Data-dependent forgetting gate that learns when to forget vs retain memory.

    Computes α_t ∈ [0, 1] based on input context.
    α_t = 0 means retain all memory, α_t = 1 means forget all.

    Architecture: 2-layer MLP with SiLU activation and sigmoid output.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute forgetting gate value.

        Args:
            x: Input tensor [batch, d_model] or [batch, seq_len, d_model]

        Returns:
            alpha: Forgetting factor [batch] or [batch, seq_len]
        """
        out = self.net(x)
        return out.squeeze(-1)


class DecayGate(nn.Module):
    """
    Data-dependent decay gate for momentum accumulator.

    Computes η_t ∈ [0, 1] for momentum decay in S_t = η_t * S_{t-1} - θ_t * grad.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)


class DeepMemoryMLP(nn.Module):
    """
    Deep MLP that serves as the neural memory.

    Paper-faithful: Memory M is an MLP with L_M >= 2 layers.
    M(k) is a simple forward pass through this MLP.

    The MLP weights ARE the memory - they get updated via gradient descent
    at test time based on the associative loss.

    Args:
        d_model: Input/output dimension
        n_layers: Number of MLP layers (paper recommends >= 2)
        hidden_dim: Hidden dimension (defaults to d_model)
        activation: Activation function (SiLU recommended)
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.d_model = d_model
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model

        # Build MLP layers
        layers: List[nn.Module] = []

        # Input layer
        if n_layers == 1:
            # Single layer: just linear projection
            layers.append(nn.Linear(d_model, d_model))
        else:
            # Multi-layer: input -> hidden
            layers.append(nn.Linear(d_model, self.hidden_dim))
            layers.append(self._get_activation(activation))

            # Hidden layers
            for i in range(n_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(self._get_activation(activation))

            # Output layer (no activation)
            layers.append(nn.Linear(self.hidden_dim, d_model))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights small for stable training
        self._init_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "silu":
            return nn.SiLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Memory retrieval: M(k) = MLP forward pass.

        Args:
            x: Key/query tensor [..., d_model]

        Returns:
            Retrieved values [..., d_model]
        """
        return self.mlp(x)

    def get_flat_params(self) -> torch.Tensor:
        """Get all MLP parameters as a single flat tensor."""
        return torch.cat([p.view(-1) for p in self.mlp.parameters()])

    def get_param_shapes(self) -> List[Tuple[int, ...]]:
        """Get shapes of all MLP parameters."""
        return [p.shape for p in self.mlp.parameters()]

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set MLP parameters from a flat tensor."""
        offset = 0
        for param in self.mlp.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset : offset + numel].view(param.shape))
            offset += numel


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory with gradient-based surprise updates.

    PAPER-FAITHFUL IMPLEMENTATION:
    - Memory M is a deep MLP (not an embedding table)
    - M(k) is a forward pass through the MLP
    - Memory updates are gradient descent on MLP weights
    - Uses momentum with data-dependent gates

    Implements Titans paper Section 3.1:
        Loss:   l(M; x) = ||M(k_t) - v_t||²     (Eq. 12)
        S_t:    η_t * S_{t-1} - θ_t * ∇l        (Eq. 13)
        M_t:    (1 - α_t) * M_{t-1} + S_t       (Eq. 14)

    Where M_t represents ALL MLP WEIGHTS at time t.

    Args:
        d_model: Input dimension from model hidden states
        d_memory: Memory MLP dimension (default: same as d_model)
        n_memory_layers: Number of layers in memory MLP (default: 2, paper recommends >= 2)
        theta_lr: Learning rate for memory updates (θ_t)
        forget_hidden: Hidden dim for forget gate MLP
        decay_hidden: Hidden dim for decay gate MLP
    """

    def __init__(
        self,
        d_model: int,
        d_memory: Optional[int] = None,
        n_memory_layers: int = 2,
        theta_lr: float = 0.01,
        forget_hidden: int = 32,
        decay_hidden: int = 32,
        # Legacy parameter - ignored but accepted for compatibility
        capacity: int = 512,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_memory_layers < 1:
            raise ValueError(f"n_memory_layers must be >= 1, got {n_memory_layers}")
        if theta_lr <= 0:
            raise ValueError(f"theta_lr must be positive, got {theta_lr}")

        self.d_model = d_model
        self.d_memory = d_memory if d_memory is not None else d_model
        self.n_memory_layers = n_memory_layers
        self.theta = theta_lr

        # Key and Value projections (project input to memory space)
        self.W_K = nn.Linear(d_model, self.d_memory)  # Key projection
        self.W_V = nn.Linear(d_model, self.d_memory)  # Value projection
        self.W_Q = nn.Linear(d_model, self.d_memory)  # Query projection

        # THE DEEP MEMORY MLP - This is where the memory lives!
        # Paper: "Memory M is a simple MLP with L_M >= 1 layers"
        # Paper Section 5.5 shows deeper memory (L_M=2,3,4) outperforms shallow
        self.memory_mlp = DeepMemoryMLP(
            d_model=self.d_memory,
            n_layers=n_memory_layers,
        )

        # Momentum buffer for MLP weights (same shape as flattened params)
        n_params = sum(p.numel() for p in self.memory_mlp.parameters())
        self.register_buffer("momentum_S", torch.zeros(n_params))

        # PERF: Pre-allocate flat parameter cache to avoid torch.cat() every update
        self.register_buffer("_flat_param_cache", torch.zeros(n_params))

        # PERF: Pre-allocate gradient cache too
        self.register_buffer("_flat_grad_cache", torch.zeros(n_params))

        # PERF: Store parameter offsets for zero-copy flatten/unflatten
        self._param_offsets: list[tuple[int, tuple[int, ...]]] = []
        offset = 0
        for p in self.memory_mlp.parameters():
            self._param_offsets.append((offset, p.shape))
            offset += p.numel()

        # Data-dependent gates
        self.forget_gate = ForgetGate(d_model, forget_hidden)
        self.decay_gate = DecayGate(d_model, decay_hidden)

        # Track last update stats for logging (saturation monitoring)
        self._last_update_stats: Optional[Dict[str, float]] = None

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values."""
        for proj in [self.W_K, self.W_V, self.W_Q]:
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _get_flat_params_into_cache(self):
        """
        Copy MLP parameters into pre-allocated flat cache.

        PERF: No torch.cat() allocation - writes directly into buffer.
        """
        for (offset, shape), param in zip(self._param_offsets, self.memory_mlp.parameters()):
            numel = param.numel()
            self._flat_param_cache[offset : offset + numel] = param.view(-1)

    def _set_params_from_cache(self):
        """
        Set MLP parameters from pre-allocated flat cache.

        PERF: Uses stored offsets to avoid Python iteration overhead.
        """
        for (offset, shape), param in zip(self._param_offsets, self.memory_mlp.parameters()):
            numel = param.numel()
            param.data.copy_(self._flat_param_cache[offset : offset + numel].view(shape))

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute associative memory loss: l(M; x) = ||M(k) - v||^2.

        PAPER-FAITHFUL: M(k) is an MLP forward pass, not softmax attention.

        The loss measures how well the MLP can map keys to values.
        This is the "surprise" signal for memory updates.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            loss: Scalar tensor (MSE between M(k) and v)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, d_model], got shape {x.shape}")

        # Project to key and value
        k = self.W_K(x)  # [batch, seq, d_memory]
        v = self.W_V(x)  # [batch, seq, d_memory]

        # PAPER-FAITHFUL: M(k) is MLP forward pass
        # Paper Eq. 12: l(M; x) = ||M(k_t) - v_t||²
        predicted = self.memory_mlp(k)  # [batch, seq, d_memory]

        # Associative loss: MSE between predicted and target values
        loss = F.mse_loss(predicted, v)

        return loss

    @torch._dynamo.disable
    def update(
        self,
        x: torch.Tensor,
        theta_t: Optional[float] = None,
        return_stats: bool = False,
    ) -> torch.Tensor:
        """
        Update memory MLP weights with gradient-based surprise.

        Note: Decorated with @torch._dynamo.disable because torch.autograd.grad()
        is explicitly marked as non-traceable by dynamo. This allows fullgraph=True
        compilation on the rest of the model.

        PAPER-FAITHFUL: Updates MLP weights, not embedding slots.

        Implements Equations 13-14 from Titans paper:
            S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)     (Eq. 13)
            M_t = (1 - α_t) * M_{t-1} + S_t          (Eq. 14)

        Where M_t refers to ALL MLP WEIGHTS at time t.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            theta_t: Optional override for memory learning rate
            return_stats: If True, return dict with loss and gate values

        Returns:
            loss: The associative memory loss (if return_stats=False)
            dict: {"loss": loss, "alpha_t": forget_rate, "eta_t": decay_rate,
                   "grad_norm": grad_norm, "grad_clipped": was_clipped} (if return_stats=True)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, d_model], got shape {x.shape}")

        theta = theta_t if theta_t is not None else self.theta

        # Compute associative loss
        loss = self.compute_loss(x)

        # Get gradients w.r.t. MLP parameters
        # This is ∇l(M; x) in the paper
        grads = torch.autograd.grad(
            loss,
            self.memory_mlp.parameters(),
            retain_graph=True,
            create_graph=False,
        )

        # PERF: Flatten gradients into pre-allocated cache (no torch.cat allocation)
        for (offset, shape), g in zip(self._param_offsets, grads):
            numel = g.numel()
            self._flat_grad_cache[offset : offset + numel] = g.view(-1)

        # Clip gradients for stability (in-place)
        grad_norm = self._flat_grad_cache.norm()
        max_grad_norm = 1.0
        grad_clipped = grad_norm > max_grad_norm
        if grad_clipped:
            self._flat_grad_cache.mul_(max_grad_norm / grad_norm)

        # Check for NaN/Inf
        if torch.isnan(self._flat_grad_cache).any() or torch.isinf(self._flat_grad_cache).any():
            if return_stats:
                return {
                    "loss": loss,
                    "alpha_t": 0.0,
                    "eta_t": 0.0,
                    "grad_norm": float("nan"),
                    "grad_clipped": False,
                    "skipped": True,
                }
            return loss  # Skip update if gradients are invalid

        # Compute data-dependent gates (aggregate over sequence)
        x_pooled = x.mean(dim=1)  # [batch, d_model]
        alpha_t = self.forget_gate(x_pooled).mean()  # scalar: forgetting rate
        eta_t = self.decay_gate(x_pooled).mean()  # scalar: momentum decay

        # Update momentum: S_t = η_t * S_{t-1} - θ_t * grad  (Eq. 13)
        # PERF: Keep eta_t as tensor - no .item() calls (avoids CUDA sync)
        with torch.no_grad():
            self.momentum_S.mul_(eta_t)
            self.momentum_S.add_(self._flat_grad_cache, alpha=-theta)

        # Update MLP weights: M_t = (1 - α_t) * M_{t-1} + S_t  (Eq. 14)
        # PERF: Use cached buffer and in-place ops to avoid allocations
        with torch.no_grad():
            # Get current flat params into pre-allocated cache
            self._get_flat_params_into_cache()

            # Apply update in-place: M_t = (1 - α) * M_{t-1} + S_t
            # PERF: Keep alpha_t as tensor - no .item() calls
            self._flat_param_cache.mul_(1.0 - alpha_t)
            self._flat_param_cache.add_(self.momentum_S)

            # Set updated params back to MLP
            self._set_params_from_cache()

        # Store stats as tensors for lazy eval (no .item() sync during training!)
        # Only convert to Python floats when actually logged via get_memory_stats()
        self._last_update_stats_tensors = {
            "alpha_t": alpha_t.detach(),
            "eta_t": eta_t.detach(),
            "grad_norm": (
                grad_norm.detach()
                if isinstance(grad_norm, torch.Tensor)
                else torch.tensor(grad_norm)
            ),
            "grad_clipped": grad_clipped,
            "skipped": False,
        }

        if return_stats:
            # Only sync when explicitly requested
            return {
                "loss": loss,
                "alpha_t": alpha_t.item(),
                "eta_t": eta_t.item(),
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "grad_clipped": grad_clipped,
                "skipped": False,
            }
        return loss

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory: y = M*(q)

        PAPER-FAITHFUL: Simple MLP forward pass.
        Uses query projection and MLP, with stop-gradient on MLP.

        Args:
            x: Query input [batch, seq_len, d_model]

        Returns:
            output: Retrieved values [batch, seq_len, d_memory]
        """
        # Project to query
        q = self.W_Q(x)  # [batch, seq, d_memory]

        # M*(q): MLP forward pass with stop gradient
        # Paper uses M* notation to indicate stop-gradient retrieval
        with torch.no_grad():
            output = self.memory_mlp(q)  # [batch, seq, d_memory]

        return output

    def reset_memory(self):
        """Reset memory MLP and momentum to initial state."""
        with torch.no_grad():
            # Re-initialize MLP weights
            self.memory_mlp._init_weights()
            # Reset momentum buffer
            self.momentum_S.zero_()

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get statistics about memory state for logging.

        Returns dict with:
            - memory_param_norm: L2 norm of MLP weights
            - momentum_norm: L2 norm of momentum buffer
            - memory_param_mean: Mean of MLP weights
            - memory_param_std: Std of MLP weights
            - n_memory_layers: Number of memory MLP layers
            - d_memory: Memory dimension
            - n_params: Number of memory parameters

        If update() was called, also includes saturation metrics:
            - alpha_t: Forget gate output (0=retain, 1=forget)
            - eta_t: Decay gate output (momentum decay)
            - grad_norm: Gradient norm before clipping
            - grad_clipped: Whether gradient was clipped
        """
        with torch.no_grad():
            flat_params = self.memory_mlp.get_flat_params()
            param_norm = flat_params.norm().item()
            momentum_norm = self.momentum_S.norm().item()
            param_mean = flat_params.mean().item()
            param_std = flat_params.std().item()

        stats = {
            "memory_param_norm": param_norm,
            "momentum_norm": momentum_norm,
            "memory_param_mean": param_mean,
            "memory_param_std": param_std,
            "n_memory_layers": float(self.n_memory_layers),
            "d_memory": float(self.d_memory),
            "n_params": float(len(flat_params)),
        }

        # Include last update stats if available (saturation monitoring)
        # Convert tensor stats to Python floats only when logging
        if (
            hasattr(self, "_last_update_stats_tensors")
            and self._last_update_stats_tensors is not None
        ):
            tensor_stats = self._last_update_stats_tensors
            stats["alpha_t"] = (
                tensor_stats["alpha_t"].item()
                if isinstance(tensor_stats["alpha_t"], torch.Tensor)
                else tensor_stats["alpha_t"]
            )
            stats["eta_t"] = (
                tensor_stats["eta_t"].item()
                if isinstance(tensor_stats["eta_t"], torch.Tensor)
                else tensor_stats["eta_t"]
            )
            stats["grad_norm"] = (
                tensor_stats["grad_norm"].item()
                if isinstance(tensor_stats["grad_norm"], torch.Tensor)
                else tensor_stats["grad_norm"]
            )
            stats["grad_clipped"] = tensor_stats["grad_clipped"]
            stats["skipped"] = tensor_stats["skipped"]

        return stats

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_model={self.d_model}, d_memory={self.d_memory}, "
            f"n_memory_layers={self.n_memory_layers}, theta={self.theta}"
        )
