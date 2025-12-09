"""
Neural Long-Term Memory module from Titans paper (arxiv 2501.00663).

This module implements the gradient-based memory with surprise-weighted updates,
adaptive forgetting gates, and parallel training formulation.

Key equations from Section 3.1:
    - Associative Memory Loss: l(M; x) = ||M(k) - v||^2  (Eq. 4)
    - Memory Update: M_t = (1 - α_t) * M_{t-1} + S_t  (Eq. 6)
    - Surprise Momentum: S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)  (Eq. 7)
    - Parallel Update: M_T = β_T * M_0 - Σ θ_i (β_T/β_i) ∇l  (Eq. 11, 13)

Version: 1.0.0
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgetGate(nn.Module):
    """
    Data-dependent forgetting gate that learns when to forget vs retain memory.

    Computes α_t ∈ [0, 1] based on input context.
    α_t = 0 means retain all memory, α_t = 1 means forget all.

    Architecture: 2-layer MLP with SiLU activation and sigmoid output.

    Args:
        d_model: Input dimension
        hidden_dim: Hidden layer dimension (default: 32)

    Returns:
        α_t: Forgetting factor in [0, 1], shape [batch] or [batch, seq_len]
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
    η_t = 0 means no momentum retention, η_t = 1 means full retention.

    Architecture: 2-layer MLP with SiLU activation and sigmoid output.

    Args:
        d_model: Input dimension
        hidden_dim: Hidden layer dimension (default: 32)

    Returns:
        η_t: Decay factor in [0, 1], shape [batch] or [batch, seq_len]
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
        Compute decay gate value.

        Args:
            x: Input tensor [batch, d_model] or [batch, seq_len, d_model]

        Returns:
            eta: Decay factor [batch] or [batch, seq_len]
        """
        out = self.net(x)
        return out.squeeze(-1)


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory with gradient-based surprise updates.

    Implements the core memory system from Titans paper (Section 3.1).
    Memory is updated based on associative loss gradients with momentum.

    Architecture:
        - Memory weights M ∈ R^{capacity × d_memory}
        - Key projection W_K: d_model → d_memory
        - Value projection W_V: d_model → d_memory
        - Query projection W_Q: d_model → d_memory (for retrieval)
        - ForgetGate: produces α_t for memory decay
        - DecayGate: produces η_t for momentum decay
        - Momentum buffer S ∈ R^{capacity × d_memory}

    Args:
        d_model: Input dimension from model hidden states
        d_memory: Memory dimension (default: same as d_model)
        capacity: Number of memory slots
        theta_lr: Learning rate for memory updates (θ_t)
        forget_hidden: Hidden dim for forget gate MLP
        decay_hidden: Hidden dim for decay gate MLP

    Example:
        >>> memory = NeuralMemory(d_model=640, capacity=512)
        >>> x = torch.randn(2, 128, 640)  # [batch, seq, d_model]
        >>> loss = memory.update(x)  # Updates memory, returns loss
        >>> retrieved = memory.retrieve(x)  # [batch, seq, d_memory]
    """

    def __init__(
        self,
        d_model: int,
        d_memory: Optional[int] = None,
        capacity: int = 512,
        theta_lr: float = 0.01,
        forget_hidden: int = 32,
        decay_hidden: int = 32,
    ):
        super().__init__()

        # Validate inputs
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if theta_lr <= 0:
            raise ValueError(f"theta_lr must be positive, got {theta_lr}")

        self.d_model = d_model
        self.d_memory = d_memory if d_memory is not None else d_model
        self.capacity = capacity
        self.theta = theta_lr

        # Memory weights (learnable parameter for gradient computation)
        self.memory = nn.Parameter(torch.zeros(capacity, self.d_memory))

        # Projection matrices
        self.W_K = nn.Linear(d_model, self.d_memory)  # Key projection
        self.W_V = nn.Linear(d_model, self.d_memory)  # Value projection
        self.W_Q = nn.Linear(d_model, self.d_memory)  # Query projection

        # Momentum accumulator (buffer, not parameter)
        self.register_buffer("momentum_S", torch.zeros(capacity, self.d_memory))

        # Data-dependent gates
        self.forget_gate = ForgetGate(d_model, forget_hidden)
        self.decay_gate = DecayGate(d_model, decay_hidden)

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values."""
        for proj in [self.W_K, self.W_V, self.W_Q]:
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute associative memory loss: l(M; x) = ||M(k) - v||^2.

        The loss measures how well the memory can reconstruct values
        from keys - this is the "surprise" signal for memory updates.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            loss: Scalar tensor (MSE between retrieved and target values)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, d_model], got shape {x.shape}")

        # Project to key and value
        k = self.W_K(x)  # [batch, seq, d_memory]
        v = self.W_V(x)  # [batch, seq, d_memory]

        # Memory retrieval: softmax attention over memory slots
        # scores = k @ M.T → [batch, seq, capacity]
        scores = torch.matmul(k, self.memory.T)
        weights = F.softmax(scores / math.sqrt(self.d_memory), dim=-1)

        # Retrieved values: weights @ M → [batch, seq, d_memory]
        retrieved = torch.matmul(weights, self.memory)

        # Associative loss: MSE between retrieved and target values
        loss = F.mse_loss(retrieved, v)

        return loss

    def update(
        self,
        x: torch.Tensor,
        theta_t: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Update memory with gradient-based surprise.

        Implements Equations 6-7 from Titans paper:
            S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)
            M_t = (1 - α_t) * M_{t-1} + S_t

        Args:
            x: Input tensor [batch, seq_len, d_model]
            theta_t: Optional override for memory learning rate

        Returns:
            loss: The associative memory loss used for update
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, d_model], got shape {x.shape}")

        theta = theta_t if theta_t is not None else self.theta

        # Compute associative loss
        loss = self.compute_loss(x)

        # Get gradient of loss w.r.t. memory
        # Use create_graph=False since we don't need higher-order gradients
        grad_M = torch.autograd.grad(
            loss,
            self.memory,
            retain_graph=True,
            create_graph=False,
        )[0]

        # Clip gradients for stability
        grad_norm = grad_M.norm()
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            grad_M = grad_M * (max_grad_norm / grad_norm)

        # Check for NaN/Inf
        if torch.isnan(grad_M).any() or torch.isinf(grad_M).any():
            # Skip update if gradients are invalid
            return loss

        # Compute data-dependent gates (aggregate over sequence)
        x_pooled = x.mean(dim=1)  # [batch, d_model]
        alpha_t = self.forget_gate(x_pooled).mean()  # scalar
        eta_t = self.decay_gate(x_pooled).mean()  # scalar

        # Update momentum: S_t = η_t * S_{t-1} - θ_t * grad
        with torch.no_grad():
            self.momentum_S.mul_(eta_t.item())
            self.momentum_S.add_(grad_M, alpha=-theta)

        # Update memory: M_t = (1 - α_t) * M_{t-1} + S_t
        with torch.no_grad():
            self.memory.data.mul_(1.0 - alpha_t.item())
            self.memory.data.add_(self.momentum_S)

        return loss

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory with stop-gradient.

        Uses query projection and softmax attention.
        Gradients do NOT flow through memory during retrieval.

        Args:
            x: Query input [batch, seq_len, d_model]

        Returns:
            output: Retrieved values [batch, seq_len, d_memory]
        """
        # Project to query
        q = self.W_Q(x)  # [batch, seq, d_memory]

        # Stop gradient on memory for retrieval
        with torch.no_grad():
            memory_detached = self.memory.detach()

        # Compute attention scores
        scores = torch.matmul(q, memory_detached.T)  # [batch, seq, capacity]
        weights = F.softmax(scores / math.sqrt(self.d_memory), dim=-1)

        # Retrieve values
        output = torch.matmul(weights, memory_detached)  # [batch, seq, d_memory]

        return output

    def parallel_update(
        self,
        x_batch: torch.Tensor,
        theta_schedule: Optional[torch.Tensor] = None,
        M_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch-parallel memory update for efficiency.

        Implements Equations 11-13 from Titans paper:
            β_i = ∏_{j=1}^i (1 - α_j)
            M_T = β_T * M_0 - Σ θ_i (β_T/β_i) ∇l(M; x_i)

        This is more efficient than sequential updates for training.

        Args:
            x_batch: Full sequence [batch, seq_len, d_model]
            theta_schedule: Per-position learning rates [seq_len] (default: uniform θ)
            M_0: Initial memory state (default: current memory)

        Returns:
            M_T: Final memory state [capacity, d_memory]
        """
        batch, seq_len, _ = x_batch.shape
        device = x_batch.device

        # Default initial memory
        if M_0 is None:
            M_0 = self.memory.data.clone()

        # Default theta schedule (uniform)
        if theta_schedule is None:
            theta_schedule = torch.full((seq_len,), self.theta, device=device)

        # Compute all forgetting gates at once
        # Need to compute alpha for each position
        alpha_list = []
        for t in range(seq_len):
            x_t = x_batch[:, t, :]  # [batch, d_model]
            alpha_t = self.forget_gate(x_t).mean()  # scalar
            alpha_list.append(alpha_t)

        alpha = torch.stack(alpha_list)  # [seq_len]

        # Cumulative product: β_i = ∏_{j=1}^i (1 - α_j)
        beta = torch.cumprod(1 - alpha, dim=0)  # [seq_len]
        beta_T = beta[-1]

        # Weight factors: θ_i * (β_T / β_i)
        # Add small epsilon to avoid division by zero
        weights = theta_schedule * (beta_T / (beta + 1e-8))  # [seq_len]

        # Compute gradients for all positions
        grads = []
        for t in range(seq_len):
            x_t = x_batch[:, t:t+1, :]  # [batch, 1, d_model]
            loss_t = self.compute_loss(x_t)
            grad_t = torch.autograd.grad(
                loss_t,
                self.memory,
                retain_graph=True,
                create_graph=False,
            )[0]
            grads.append(grad_t)

        grads = torch.stack(grads, dim=0)  # [seq_len, capacity, d_memory]

        # Weighted sum of gradients
        weighted_grads = (weights.view(-1, 1, 1) * grads).sum(dim=0)

        # Final memory state: M_T = β_T * M_0 - weighted_grads
        M_T = beta_T * M_0 - weighted_grads

        # Update memory in-place
        with torch.no_grad():
            self.memory.data.copy_(M_T)

        return M_T

    def reset_memory(self):
        """Reset memory and momentum to zero state."""
        with torch.no_grad():
            self.memory.data.zero_()
            self.momentum_S.zero_()

    # Note: theta is a hyperparameter, not a learned weight, so we don't
    # override state_dict/load_state_dict. The default nn.Module behavior
    # is correct - it saves memory, momentum_S, projections, and gates.
    # theta should be set via config, not checkpointing.

    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory state for logging."""
        with torch.no_grad():
            memory_norm = self.memory.norm().item()
            momentum_norm = self.momentum_S.norm().item()
            memory_mean = self.memory.mean().item()
            memory_std = self.memory.std().item()

        return {
            "memory_norm": memory_norm,
            "momentum_norm": momentum_norm,
            "memory_mean": memory_mean,
            "memory_std": memory_std,
            "capacity": float(self.capacity),
            "d_memory": float(self.d_memory),
        }

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_model={self.d_model}, d_memory={self.d_memory}, "
            f"capacity={self.capacity}, theta={self.theta}"
        )
