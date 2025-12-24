"""
DeepNestedOptimizer: Complete Nested Learning optimizer for MoE models.

Implements the Nested Learning paradigm from NeurIPS 2025:
- Deep Momentum Gradient Descent (DMGD) with L2 regression loss
- NestedController for learned LR multipliers
- Continuum Memory System (CMS) for multi-frequency updates
- UnrolledMetaTrainer for meta-learning via k-step unrolling

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025

Key features:
- Meta-objective: L_{t+k} (validation loss after k unrolled steps)
- MomentumMLP trained with L2 regression on gradient sequences
- Dual-mode API: 'simple' (automatic meta-updates) or 'explicit' (manual control)
- GradScaler compatible for mixed-precision training
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .nested_controller import NestedController
from .param_groups import group_moe_params, infer_param_depth
from .meta_trainer import SimplifiedMetaTrainer, UnrolledMetaTrainer


def _fused_clip_grad_norm_(
    parameters: Iterator[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    inplace: bool = True,
) -> Tensor:
    """
    Fused gradient clipping using torch._foreach_norm.

    Replaces clip_grad_norm_ with a version that uses fused operations
    to reduce kernel launches from O(n_params) to O(1).

    Args:
        parameters: Iterator of parameters with gradients
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm (default: 2.0 for L2 norm)
        error_if_nonfinite: Raise error if gradient norm is NaN/Inf
        inplace: If True, modify gradients in-place. If False, use standard
                 clip_grad_norm_ which is safer for meta-learning graphs.

    Returns:
        Total gradient norm (scalar tensor)
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    # Convert to list to allow multiple iterations
    parameters = list(parameters)

    # Collect gradients
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device
    dtype = grads[0].dtype

    if norm_type == float('inf'):
        # Max norm: find max absolute value across all grads
        norms = [g.abs().max() for g in grads]
        total_norm = torch.stack(norms).max()
    else:
        # Use fused foreach_norm for p-norm
        norms = torch._foreach_norm(grads, ord=norm_type)
        stacked = torch.stack(norms)
        total_norm = (stacked ** norm_type).sum() ** (1.0 / norm_type)

    if error_if_nonfinite and (torch.isnan(total_norm) or torch.isinf(total_norm)):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            f'`parameters` is non-finite, so it cannot be clipped.'
        )

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if inplace:
        # Fused in-place multiplication - fast but breaks autograd graph
        # Use this for CUDA graphs or when gradients aren't needed for meta-learning
        torch._foreach_mul_(grads, clip_coef_clamped)
    else:
        # Non-in-place: use standard PyTorch clipping which preserves graph
        # Slightly slower but safe for meta-learning
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

    return total_norm


def preprocess_gradient(g: Tensor, p: float = 10.0) -> Tuple[Tensor, Tensor]:
    """
    Log-sign preprocessing from Andrychowicz et al. 2016.

    This preprocessing is critical for learned optimizers because:
    1. Raw gradients can vary by many orders of magnitude
    2. MLPs struggle with such wide dynamic ranges
    3. Log-sign encoding compresses the range while preserving sign information

    The encoding produces two features per gradient element:
    - log_g: log(|g|)/p if |g| >= exp(-p), else -1 (indicating "small")
    - sign_g: sign(g) if |g| >= exp(-p), else exp(p)*g (smooth transition)

    Args:
        g: Gradient tensor (any shape)
        p: Precision parameter (default 10.0). Higher p = more precision for small values.

    Returns:
        log_g: Log-magnitude features, same shape as g
        sign_g: Sign features, same shape as g
    """
    abs_g = g.abs()
    # Use Python math.exp to avoid GPU->CPU sync from torch.tensor creation
    # exp(-10) ≈ 4.5e-5, so values smaller than this are "small"
    threshold = math.exp(-p)
    exp_p = math.exp(p)

    # For |g| >= threshold: use log encoding
    # For |g| < threshold: use linear encoding (smooth near zero)
    log_g = torch.where(
        abs_g >= threshold,
        torch.log(abs_g) / p,  # Normalized log (in range [-1, ~1] for typical gradients)
        torch.full_like(g, -1.0)  # Indicator for "small"
    )
    sign_g = torch.where(
        abs_g >= threshold,
        g.sign(),  # Just the sign
        exp_p * g  # Linear scaling for small values (scalar * tensor is efficient)
    )
    return log_g, sign_g


class L2RegressionMomentum(nn.Module):
    """
    Learned momentum with L2 regression internal objective.

    Replaces linear momentum (v = beta*v + g) with a neural network:
    v = MLP(concat(g, v_prev)) trained to minimize ||predicted - actual_improvement||^2

    The key insight from Nested Learning: momentum is associative memory
    mapping gradient keys to update values. L2 regression provides a more
    robust learning signal than dot-product similarity.

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers (default: 2)

    Architecture:
        Input: [grad_stats, momentum_stats, context] -> 9 features
        Hidden: num_layers x hidden_dim with SiLU activation
        Output: [scale, shift, damping] -> 3 transform parameters

    Transform semantics:
        v_new = scale * v_prev + shift * grad
        update = v_new * (1 - damping)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input: [v_stats(3), g_stats(3), context(3)] = 9 features
        # Output: [scale, shift, damping] = 3 parameters
        input_dim = 9

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.9))

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable initial behavior (close to standard momentum)."""
        with torch.no_grad():
            # Output layer: initialize to produce reasonable defaults
            output_layer = self.net[-1]
            nn.init.zeros_(output_layer.weight)
            # Target: scale~0.9, shift~1.0 (CRITICAL!), damping~0.0
            # Standard momentum: v = 0.9*v + 1.0*g, update = v
            #
            # scale = sigmoid(x)*0.49+0.5 → for 0.9: x=1.5
            # shift = tanh(x) → for ~1.0: x=3.0 gives tanh(3)=0.995
            # damping = sigmoid(x) → for ~0.0: x=-5.0 gives sigmoid(-5)=0.007
            #
            # The previous shift=0.1 was WRONG - it made updates 11x smaller!
            output_layer.bias.data = torch.tensor([1.5, 3.0, -5.0])

    def compute_stats(self, tensor: Tensor) -> Tensor:
        """Compute statistics [mean, std, norm] for a tensor."""
        with torch.no_grad():
            mean = tensor.mean()
            # Use tensor.new_tensor to avoid separate device/dtype specification
            std = tensor.std() if tensor.numel() > 1 else tensor.new_tensor(0.0)
            norm = tensor.norm()
            # Keep stack inside no_grad to avoid accidental gradient tracking
            return torch.stack([mean, std, norm])

    def forward(
        self,
        grad: Tensor,
        prev_momentum: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute momentum transformation parameters.

        Args:
            grad: Current gradient tensor (any shape, will be flattened for stats)
            prev_momentum: Previous momentum tensor (same shape as grad)
            context: Context vector [normalized_step, lr, normalized_loss]

        Returns:
            scale: Momentum retention factor [0.5, 1.5]
            shift: Gradient mixing factor [-1, 1]
            damping: Damping factor [0, 1]
        """
        # Compute statistics
        g_stats = self.compute_stats(grad.flatten())
        v_stats = self.compute_stats(prev_momentum.flatten())

        # Concatenate inputs
        x = torch.cat([v_stats, g_stats, context], dim=-1)

        # Forward through network
        out = self.net(x)

        # Transform to bounded ranges
        # CRITICAL: scale must be < 1.0 for momentum stability
        scale = torch.sigmoid(out[0]) * 0.49 + 0.5  # [0.5, 0.99]
        shift = torch.tanh(out[1])  # [-1, 1]
        damping = torch.sigmoid(out[2])  # [0, 1]

        return scale, shift, damping

    def compute_internal_loss(
        self,
        predicted_update: Tensor,
        actual_improvement: Tensor,
    ) -> Tensor:
        """
        L2 regression loss for momentum MLP training.

        Args:
            predicted_update: What the MLP predicted as the optimal update
            actual_improvement: param_{t+k} - param_t (what actually helped)

        Returns:
            L2 loss: ||predicted - actual||^2
        """
        return torch.nn.functional.mse_loss(predicted_update, actual_improvement)


class DirectUpdateMLP(nn.Module):
    """
    Per-element update MLP following Andrychowicz et al. 2016.

    Key differences from L2RegressionMomentum:
    1. Uses gradient preprocessing (log-sign encoding)
    2. Operates on per-element features, not summary statistics
    3. Outputs direct update tensor, not scalar coefficients

    For efficiency, we:
    1. Flatten gradients to 1D
    2. Stack features along a new dimension: [N, 4] for [log_g, sign_g, log_m, sign_m]
    3. Process through MLP that outputs [N, 1]
    4. Reshape back to original gradient shape

    The MLP is shared across all elements (coordinate-wise), making it
    parameter-efficient regardless of model size.

    Args:
        hidden_dim: Hidden layer dimension (default: 20, small as per paper)
        num_layers: Number of hidden layers (default: 2)
        use_momentum: Whether to include momentum features (default: True)

    Architecture:
        Input: 4 features per element [log_g, sign_g, log_m, sign_m]
        Hidden: num_layers x hidden_dim with SiLU activation
        Output: 1 value per element (the update)
    """

    def __init__(
        self,
        hidden_dim: int = 20,
        num_layers: int = 2,
        use_momentum: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_momentum = use_momentum

        # Input: [log_g, sign_g] + optionally [log_m, sign_m]
        input_dim = 4 if use_momentum else 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # Output: 1 value per element
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate SGD at start (output ≈ gradient magnitude * sign)."""
        with torch.no_grad():
            # Standard initialization for hidden layers
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Initialize output layer to approximate: output ≈ sign_g * exp(log_g * p)
            # This makes initial updates close to the raw gradient
            # Input features are [log_g, sign_g, log_m, sign_m]
            # We want output ≈ sign_g * 0.1 (conservative start, but not zero)
            output_layer = self.net[-1]
            nn.init.zeros_(output_layer.weight)
            # Set weight for sign_g feature (index 1) to produce reasonable output
            output_layer.weight.data[0, 1] = 0.1  # output ≈ 0.1 * sign_g
            nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        grad: Tensor,
        prev_momentum: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute per-element update from preprocessed gradient features.

        Args:
            grad: Current gradient tensor (any shape)
            prev_momentum: Previous momentum tensor (same shape as grad), optional

        Returns:
            update: Update tensor (same shape as grad)
        """
        original_shape = grad.shape
        grad_flat = grad.flatten()
        n_elements = grad_flat.numel()

        # Preprocess gradient
        log_g, sign_g = preprocess_gradient(grad_flat)

        if self.use_momentum and prev_momentum is not None:
            # Preprocess momentum
            mom_flat = prev_momentum.flatten()
            log_m, sign_m = preprocess_gradient(mom_flat)

            # Stack features: [N, 4]
            features = torch.stack([log_g, sign_g, log_m, sign_m], dim=-1)
        else:
            # Stack features: [N, 2]
            features = torch.stack([log_g, sign_g], dim=-1)

        # Forward through network: [N, 4] -> [N, hidden] -> [N, 1]
        update_flat = self.net(features).squeeze(-1)  # [N]

        # Reshape to original gradient shape
        return update_flat.view(original_shape)

    def compute_stats(self, tensor: Tensor) -> Tensor:
        """Compute statistics [mean, std, norm] for a tensor (for compatibility)."""
        with torch.no_grad():
            mean = tensor.mean()
            std = tensor.std() if tensor.numel() > 1 else torch.tensor(0.0, device=tensor.device)
            norm = tensor.norm()
        return torch.stack([mean, std, norm])


class ContinuumMemoryState:
    """
    Per-parameter state for multi-frequency CMS updates.

    Implements hierarchical memory with exponentially spaced update frequencies:
    - Level 0: Updates every step (fast adaptation)
    - Level 1: Every base_freq steps (short-term patterns)
    - Level 2: Every base_freq^2 steps (long-term knowledge)

    Args:
        param_shape: Shape of the parameter tensor
        num_levels: Number of frequency levels (default: 3)
        base_frequency: Base frequency multiplier (default: 10)
        device: Torch device
    """

    def __init__(
        self,
        param_shape: torch.Size,
        num_levels: int = 3,
        base_frequency: int = 10,
        device: torch.device = None,
    ):
        self.param_shape = param_shape
        self.num_levels = num_levels
        self.base_frequency = base_frequency
        self.device = device or torch.device('cpu')

        # Initialize per-level state
        self.levels: Dict[int, Dict[str, Any]] = {}
        for level in range(num_levels):
            freq = base_frequency ** level  # 1, 10, 100
            self.levels[level] = {
                'frequency': freq,
                'momentum': torch.zeros(param_shape, device=self.device),
                'step_count': 0,
                'ema_decay': 0.99 ** (1.0 / freq),  # Slower decay at higher levels
                'accumulated_grad': torch.zeros(param_shape, device=self.device),
                # Paper-aligned additions:
                'ema_grad': torch.zeros(param_shape, device=self.device),  # L2 regression target
                'v_sq': torch.zeros(param_shape, device=self.device),  # Second-moment (Adam-style)
                'beta2': 0.999,  # Standard Adam beta2 for variance tracking
            }

    def should_update(self, level: int, global_step: int) -> bool:
        """Check if this level should update at the current step."""
        if level not in self.levels:
            return False
        return global_step % self.levels[level]['frequency'] == 0

    def accumulate_grad(self, grad: Tensor):
        """Accumulate gradient for all levels (called every step)."""
        for level_state in self.levels.values():
            level_state['accumulated_grad'] += grad

    def get_update(self, level: int) -> Tensor:
        """Get accumulated gradient for a level and reset accumulator."""
        state = self.levels[level]
        update = state['accumulated_grad'].clone()
        state['accumulated_grad'].zero_()
        state['step_count'] += 1
        return update

    def update_momentum(self, level: int, new_momentum: Tensor):
        """Update momentum for a specific level."""
        self.levels[level]['momentum'] = new_momentum

    def get_momentum(self, level: int) -> Tensor:
        """Get current momentum for a level."""
        return self.levels[level]['momentum']

    def update_ema_grad(self, level: int, grad: Tensor):
        """Update EMA gradient for L2 regression target (paper-aligned)."""
        state = self.levels[level]
        decay = state['ema_decay']
        state['ema_grad'] = decay * state['ema_grad'] + (1 - decay) * grad

    def get_ema_grad(self, level: int) -> Tensor:
        """Get EMA gradient target for L2 regression."""
        return self.levels[level]['ema_grad']

    def update_second_moment(self, level: int, grad: Tensor):
        """Update running variance estimate (Adam-style v_t) for adaptive LR."""
        state = self.levels[level]
        beta2 = state['beta2']
        state['v_sq'] = beta2 * state['v_sq'] + (1 - beta2) * grad.pow(2)

    def get_adaptive_lr(self, level: int, eps: float = 1e-8) -> Tensor:
        """Get per-parameter adaptive learning rate from second moment.

        Uses bias-corrected estimate and clamping to prevent numerical issues.
        When v_sq is still near zero (early training), returns 1.0 (no adaptation).
        """
        state = self.levels[level]
        v_sq = state['v_sq']
        step_count = state['step_count']

        # Bias correction (like Adam)
        if step_count > 0:
            beta2 = state['beta2']
            bias_correction = 1 - (beta2 ** step_count)
            v_sq_corrected = v_sq / bias_correction
        else:
            # No updates yet - return no adaptation
            return torch.ones_like(v_sq)

        # Compute adaptive LR with clamping to prevent extreme values
        # Standard Adam uses 1/sqrt(v), but clamp to [0.1, 10.0] for stability
        adaptive_lr = 1.0 / (v_sq_corrected.sqrt() + eps)
        adaptive_lr = adaptive_lr.clamp(min=0.1, max=10.0)

        return adaptive_lr

    def to(self, device: torch.device) -> 'ContinuumMemoryState':
        """Move all tensors to device."""
        self.device = device
        for level_state in self.levels.values():
            level_state['momentum'] = level_state['momentum'].to(device)
            level_state['accumulated_grad'] = level_state['accumulated_grad'].to(device)
            level_state['ema_grad'] = level_state['ema_grad'].to(device)
            level_state['v_sq'] = level_state['v_sq'].to(device)
        return self


class DeepNestedOptimizer:
    """
    Complete Nested Learning optimizer for MoE models.

    Combines:
    - L2RegressionMomentum: Learned gradient compression (DMGD)
    - NestedController: Learned LR multipliers from training outcomes
    - ContinuumMemoryState: Multi-frequency update scheduling (CMS)
    - Meta-learning: Trains components via k-step unrolled differentiation

    Two operation modes:
    - 'simple': Automatic meta-updates (for BC training)
    - 'explicit': Manual meta-update calls (for PPO with GradScaler)

    Args:
        model: Model to optimize (MoEMinimalLLM)
        base_lr: Base learning rate
        meta_lr: Learning rate for meta-optimizer (trains MomentumMLP + Controller)
        k_unroll: Number of steps to unroll for meta-objective
        cms_frequencies: Update frequencies for CMS levels [1, 10, 100]
        momentum_hidden_dim: Hidden dimension for MomentumMLP
        controller_hidden_dim: Hidden dimension for NestedController
        use_gradient_checkpointing: Whether to use checkpointing in unrolling
        mode: 'simple' (auto meta-updates) or 'explicit' (manual control)
        meta_update_freq: How often to do meta-updates in simple mode

    Example (simple mode):
        >>> optimizer = DeepNestedOptimizer(model, mode='simple')
        >>> for batch in train_loader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step(loss.item())  # Auto meta-updates internally

    Example (explicit mode):
        >>> optimizer = DeepNestedOptimizer(model, mode='explicit')
        >>> for step, batch in enumerate(train_loader):
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step(loss.item())
        ...     if step % 100 == 0:
        ...         optimizer.meta_update(val_batch)  # Manual meta-update
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-3,
        meta_lr: float = 1e-4,
        k_unroll: int = 5,
        cms_frequencies: Optional[List[int]] = None,
        momentum_hidden_dim: int = 64,
        momentum_num_layers: int = 2,
        controller_hidden_dim: int = 32,
        controller_num_layers: int = 2,
        use_gradient_checkpointing: bool = True,
        mode: str = 'simple',
        meta_update_freq: int = 100,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        low_memory: bool = False,  # Use full CMS levels for proper forgetting prevention
        use_cms_updates: bool = False,  # AdamW + learned LR multipliers (proven to work)
        use_preprocessing: bool = True,  # Use Andrychowicz 2016 gradient preprocessing
        # NOTE: use_preprocessing=True (default) enables per-element MLP with log-sign
        # preprocessing, which should perform much better than summary-statistics approach.
        # Set to False to use legacy L2RegressionMomentum (scalar coefficients).
        use_compile: bool = False,  # Apply torch.compile to controller/MLP for faster inference
        controller_update_freq: int = 1,  # Update controller every N steps (1 = every step)
        # CUDA Graph parameters for eliminating Python dispatch overhead
        use_cuda_graph: bool = False,  # Enable CUDA graph capture and replay
        cuda_graph_warmup_steps: int = 3,  # Steps before capturing graph (ensures stable shapes)
    ):
        if mode not in ('simple', 'explicit'):
            raise ValueError(f"mode must be 'simple' or 'explicit', got {mode}")

        self.model = model
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.k_unroll = k_unroll
        # Low memory mode uses single-level CMS (saves ~4GB for 167M param model)
        if low_memory:
            self.cms_frequencies = cms_frequencies or [1]  # Single level
        else:
            self.cms_frequencies = cms_frequencies or [1, 10, 100]  # 3 levels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.mode = mode
        self.meta_update_freq = meta_update_freq
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_cms_updates = use_cms_updates
        self.use_preprocessing = use_preprocessing
        self.use_compile = use_compile
        self.controller_update_freq = controller_update_freq
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps

        # Get device from model
        self.device = next(model.parameters()).device

        # Group parameters (core vs embed) - MoE-specific grouping
        core_params, embed_params = group_moe_params(model)
        self.n_groups = 2

        # Handle empty embed group (unlikely but possible)
        if len(embed_params) == 0:
            _param_groups = [
                {'params': core_params, 'lr': base_lr, 'name': 'core'},
                {'params': [], 'lr': base_lr, 'name': 'embed'},
            ]
        else:
            _param_groups = [
                {'params': core_params, 'lr': base_lr, 'name': 'core'},
                {'params': embed_params, 'lr': base_lr, 'name': 'embed'},
            ]

        # Create base optimizer (AdamW) for actual parameter updates
        # Enable capturable mode for CUDA graph compatibility (PyTorch 2.0+)
        adamw_kwargs = {
            'weight_decay': weight_decay,
        }
        if use_cuda_graph:
            # capturable=True required for CUDA graph capture of optimizer.step()
            # LR must be a GPU tensor for dynamic updates with capturable mode
            # NOTE: We don't use fused=True because it can have different numerical
            # behavior with sparse gradients (e.g., from embeddings)
            adamw_kwargs['capturable'] = True
            # adamw_kwargs['fused'] = True  # Disabled for numerical compatibility
            # Create tensor LRs for each group (must be 0-dim scalar tensors)
            self._graph_lr_tensors = [
                torch.tensor(base_lr, device=self.device, dtype=torch.float32)
                for _ in range(len(_param_groups))
            ]
            # Set tensor LRs in param groups
            for i, group in enumerate(_param_groups):
                group['lr'] = self._graph_lr_tensors[i]
        else:
            # Use scalar LR for non-graph mode
            adamw_kwargs['lr'] = base_lr
            self._graph_lr_tensors = None

        self.base_optimizer = torch.optim.AdamW(
            _param_groups,
            **adamw_kwargs,
        )

        # Learned components
        # Choose between new per-element MLP (Andrychowicz 2016) and legacy scalar-coefficient MLP
        if use_preprocessing:
            # DirectUpdateMLP: per-element features -> per-element updates
            # Uses smaller hidden_dim (20 vs 64) since it processes elements, not stats
            self.momentum_mlp = DirectUpdateMLP(
                hidden_dim=min(momentum_hidden_dim, 20),  # Paper uses small hidden dim
                num_layers=momentum_num_layers,
                use_momentum=True,
            ).to(self.device)
        else:
            # Legacy: summary stats -> scalar coefficients -> linear combination
            self.momentum_mlp = L2RegressionMomentum(
                hidden_dim=momentum_hidden_dim,
                num_layers=momentum_num_layers,
            ).to(self.device)

        self.controller = NestedController(
            hidden_dim=controller_hidden_dim,
            num_layers=controller_num_layers,
            n_groups=self.n_groups,
        ).to(self.device)

        # Apply torch.compile for faster inference (optional, has warmup cost)
        if use_compile and hasattr(torch, 'compile'):
            # Use reduce-overhead mode for small MLPs (reduces kernel launch overhead)
            self.controller = torch.compile(self.controller, mode='reduce-overhead')
            self.momentum_mlp = torch.compile(self.momentum_mlp, mode='reduce-overhead')

        # Meta-optimizer (trains MomentumMLP + Controller)
        self.meta_optimizer = torch.optim.Adam(
            list(self.momentum_mlp.parameters()) +
            list(self.controller.parameters()),
            lr=meta_lr,
        )

        # Separate optimizer for DirectUpdateMLP training via surrogate loss
        # This runs on every CMS step for online learning
        if use_preprocessing:
            self.mlp_optimizer = torch.optim.Adam(
                self.momentum_mlp.parameters(),
                lr=1e-3,  # Higher LR for faster MLP adaptation
            )
            # Track previous MLP outputs for temporal smoothness loss
            self._mlp_output_history: Dict[int, Tensor] = {}
            # Training frequency for MLP (every N steps)
            self.mlp_train_freq = 5  # Train every 5 steps to reduce memory pressure
            # Accumulated surrogate losses for batch update
            self._surrogate_losses: List[Tensor] = []
        else:
            self.mlp_optimizer = None
            self._mlp_output_history = {}
            self.mlp_train_freq = 1
            self._surrogate_losses = []

        # Per-parameter state (CMS + momentum)
        self.state: Dict[Tensor, ContinuumMemoryState] = {}
        self._init_state()

        # Tracking
        self.global_step = 0
        self._lr_multipliers = torch.ones(self.n_groups, device=self.device)
        self._pending_loss: Optional[float] = None
        self.last_meta_loss: Optional[float] = None
        self.controller_grad_norm: float = 0.0

        # EMA loss for controller input
        self.ema_loss = torch.zeros(self.n_groups, device=self.device)
        self.beta_ema = 0.1

        # Meta-trainers
        self.simplified_meta_trainer = SimplifiedMetaTrainer(
            window_size=meta_update_freq,
            improvement_threshold=0.001,
        )
        self.unrolled_meta_trainer = UnrolledMetaTrainer(
            k_steps=k_unroll,
            use_checkpointing=use_gradient_checkpointing,
        )

        # Compute parameter depths
        self._compute_group_depths()

        # CUDA Graph state
        # Graph is captured after warmup_steps to ensure stable tensor shapes
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cuda_graph_stream: Optional[torch.cuda.Stream] = None
        self._cuda_graph_captured = False
        # Static buffers for graph inputs (addresses must be stable for replay)
        self._graph_loss_buffer: Optional[Tensor] = None
        self._graph_stats_buffer: Optional[Tensor] = None

    def _init_state(self):
        """Initialize CMS state for all parameters."""
        num_levels = len(self.cms_frequencies)
        for group in self.param_groups:
            for param in group['params']:
                self.state[param] = ContinuumMemoryState(
                    param_shape=param.shape,
                    num_levels=num_levels,
                    base_frequency=self.cms_frequencies[1] if len(self.cms_frequencies) > 1 else 10,
                    device=self.device,
                )

    def _compute_group_depths(self):
        """Compute average depth for each parameter group."""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'n_layers'):
            n_layers = self.model.config.n_layers
        elif hasattr(self.model, 'transformer_blocks'):
            n_layers = len(self.model.transformer_blocks)
        elif hasattr(self.model, 'layers'):
            n_layers = len(self.model.layers)
        else:
            n_layers = 1

        self.group_depths = []
        for group in self.param_groups:
            if len(group['params']) == 0:
                self.group_depths.append(0.5)
                continue

            depths = []
            for param in group['params']:
                for name, p in self.model.named_parameters():
                    if p is param:
                        depth = infer_param_depth(name, n_layers)
                        depths.append(depth)
                        break

            avg_depth = sum(depths) / len(depths) if depths else 0.5
            self.group_depths.append(avg_depth)

        self.group_depths = torch.tensor(self.group_depths, device=self.device)

    def _compute_group_stats(self) -> Tensor:
        """
        Compute gradient statistics for each parameter group.

        Returns:
            Tensor of shape [n_groups, 3] with [grad_norm, param_norm, avg_depth]

        Note: Uses torch._foreach_norm for fused kernel execution.
              This reduces ~1600 small kernel launches to ~4 fused operations.
        """
        # Use cached stats tensor if available (avoid allocation)
        if not hasattr(self, '_stats_cache'):
            self._stats_cache = torch.zeros(self.n_groups, 3, device=self.device, dtype=torch.float32)
        stats = self._stats_cache
        stats.zero_()

        for i, group in enumerate(self.param_groups):
            if len(group['params']) == 0:
                stats[i, 2] = self.group_depths[i]
                continue

            # Collect params and grads as lists for fused operations
            params = [p for p in group['params']]
            grads = [p.grad for p in group['params'] if p.grad is not None]

            # Fused norm computation using torch._foreach_norm
            # This launches ONE kernel for all params instead of one per param
            if params:
                # _foreach_norm returns a list of norms, we need sum of squares
                # Use torch.no_grad() to avoid creating autograd graph through params
                # (we don't need gradients through model params for meta-learning)
                with torch.no_grad():
                    param_norms = torch._foreach_norm(params)
                    # Stack and compute total: sqrt(sum(norms^2))
                    if param_norms:
                        stacked_norms = torch.stack(param_norms)
                        stats[i, 1] = (stacked_norms ** 2).sum().sqrt()

            if grads:
                # Gradients don't need autograd graph either
                with torch.no_grad():
                    grad_norms = torch._foreach_norm(grads)
                    if grad_norms:
                        stacked_norms = torch.stack(grad_norms)
                        stats[i, 0] = (stacked_norms ** 2).sum().sqrt()

            stats[i, 2] = self.group_depths[i]

        # NaN/Inf guard: branchless replace with safe defaults (no .item() sync!)
        nan_mask = torch.isnan(stats) | torch.isinf(stats)
        # Default: grad_norm=1.0, param_norm=1.0, depth from group_depths
        defaults = torch.zeros_like(stats)
        defaults[:, 0] = 1.0  # grad_norm default
        defaults[:, 1] = 1.0  # param_norm default
        defaults[:, 2] = self.group_depths  # depth from cache
        # Branchless replace: use torch.where (no CPU sync!)
        stats = torch.where(nan_mask, defaults, stats)

        # Detach stats to ensure meta-learning doesn't backprop through model params
        return stats.detach()

    def _get_context(self, loss_value: float) -> Tensor:
        """Create context vector for MomentumMLP.

        Uses cached tensor to avoid CPU->GPU transfer overhead.
        """
        # Lazy init cached context tensor
        if not hasattr(self, '_context_cache'):
            self._context_cache = torch.zeros(3, device=self.device, dtype=torch.float32)

        # Update in-place on GPU (no CPU->GPU transfer)
        self._context_cache[0] = self.global_step / 1000.0
        self._context_cache[1] = self.base_lr
        self._context_cache[2] = math.log(max(loss_value, 1e-8) + 1.0)

        return self._context_cache

    def _init_cuda_graph_buffers(self):
        """
        Initialize static buffers for CUDA Graph capture.

        These buffers have fixed addresses that the graph can reference.
        Values are copied into these buffers before graph replay.
        """
        # Stats buffer: [n_groups, 3] for gradient statistics
        self._graph_stats_buffer = torch.zeros(
            self.n_groups, 3, device=self.device, dtype=torch.float32
        )
        # LR multipliers buffer: [n_groups]
        self._graph_lr_mult_buffer = torch.ones(
            self.n_groups, device=self.device, dtype=torch.float32
        )

    def _capture_cuda_graph(self):
        """
        Capture the optimizer step as a CUDA Graph.

        This method captures the non-CMS (AdamW) path since it's the most common
        and has deterministic operations. The captured graph includes:
        1. Computing gradient statistics (_compute_group_stats)
        2. Controller forward pass
        3. Fused gradient clipping
        4. AdamW step

        Prerequisites:
        - Must be called after warmup steps (tensor shapes stable)
        - Gradients must exist on all parameters
        - Device must support CUDA Graphs (compute capability >= 7.0)

        Reference: NVIDIA CUDA Programming Guide, Section 3.2.8 "CUDA Graphs"
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, disabling CUDA Graph.")
            self.use_cuda_graph = False
            return

        # Check compute capability (graphs require CC >= 7.0)
        device = torch.device(self.device)
        if device.type == 'cuda':
            capability = torch.cuda.get_device_capability(device)
            if capability[0] < 7:
                warnings.warn(
                    f"CUDA Graphs require compute capability >= 7.0, "
                    f"got {capability[0]}.{capability[1]}. Disabling."
                )
                self.use_cuda_graph = False
                return

        # Initialize static buffers
        self._init_cuda_graph_buffers()

        # Synchronize before capture to ensure all prior work is complete
        torch.cuda.synchronize()

        # Clip gradients BEFORE capture (outside graph - addresses may change)
        if self.max_grad_norm > 0:
            _fused_clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Capture the CUDA graph
        # NOTE: All lazy CUDA initializations should be complete by now since
        # we ran cuda_graph_warmup_steps iterations through the normal path
        self._cuda_graph = torch.cuda.CUDAGraph()

        # Capture the graph (records ONLY AdamW step, not gradient clipping)
        with torch.cuda.graph(self._cuda_graph):
            self._step_impl_for_graph()

        # Synchronize to ensure capture is complete
        torch.cuda.synchronize()

        self._cuda_graph_captured = True

        # NOTE: The first replay is done inside the capture context above,
        # so no need to replay again here. The graph was executed during capture.

    def _step_impl_for_graph(self):
        """
        Graph-safe step implementation for CUDA Graph capture/replay.

        CRITICAL: This method must NOT contain:
        - Python conditionals on tensor values (use unconditional ops)
        - CPU synchronization (no .item(), .tolist(), no Python prints)
        - Dynamic memory allocation (shapes must be stable)
        - Operations that change tensor addresses

        This captures ONLY the AdamW step. Gradient clipping is done OUTSIDE
        the graph because gradient tensor addresses may change between backward
        passes (PyTorch may allocate new tensors or reuse existing ones).

        NOTE: LR updates and controller forward pass are done BEFORE graph replay
        in _replay_cuda_graph() since they involve CPU operations.

        Reference: NVIDIA CUDA C++ Best Practices Guide,
        Section 11.1.2 "Applicability of CUDA Graphs"
        """
        # === AdamW step ONLY ===
        # AdamW.step() is graph-safe after warmup (state tensors have stable addresses)
        # The step() method uses foreach operations internally which are graph-compatible
        #
        # NOTE: Gradient clipping is intentionally NOT included here because:
        # - Gradient tensors may have different addresses between backward passes
        # - CUDA Graphs require stable tensor addresses for all captured operations
        # - Clipping is done BEFORE graph replay in _replay_cuda_graph()
        self.base_optimizer.step()

    def _replay_cuda_graph(self, loss_value: float) -> Dict[str, Any]:
        """
        Replay the captured CUDA Graph.

        Before replay, we handle:
        1. EMA loss update (CPU computation)
        2. Controller forward pass and LR updates (involves CPU-GPU sync)
        3. Gradient clipping (must be outside graph - gradient addresses may change)

        The graph itself only contains AdamW step.

        Args:
            loss_value: Current loss for EMA tracking

        Returns:
            Dict with step info (lr_multipliers, ema_loss, etc.)
        """
        # === Pre-graph operations (must complete before graph replay) ===

        # Update EMA loss
        self.ema_loss = (1 - self.beta_ema) * self.ema_loss + self.beta_ema * loss_value

        # Controller update (every controller_update_freq steps)
        update_controller = (
            self.global_step % self.controller_update_freq == 0 or
            self.global_step == 1
        )

        if update_controller:
            # Compute stats and get LR multipliers from controller
            stats = self._compute_group_stats()

            with torch.no_grad():
                lr_mults = self.controller(stats)
                self._lr_multipliers = lr_mults.clone()

            # Update base optimizer LRs using tensor operations (graph-compatible)
            # This modifies the tensor values IN-PLACE so the graph sees the new values
            if self._graph_lr_tensors is not None:
                for i in range(len(self._graph_lr_tensors)):
                    if len(self.base_optimizer.param_groups[i]['params']) > 0:
                        new_lr = self.base_lr * self._lr_multipliers[i]
                        self._graph_lr_tensors[i].fill_(new_lr)

        # === Gradient clipping (BEFORE graph replay) ===
        # This must be done outside the graph because gradient tensor addresses
        # may change between backward passes. The graph captures specific memory
        # addresses, so if gradients are in different locations, it would corrupt memory.
        if self.max_grad_norm > 0:
            _fused_clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # === Replay the captured graph ===
        # The graph contains ONLY AdamW step (gradient clipping is done above)
        self._cuda_graph.replay()

        # === Post-graph operations ===
        return {
            'global_step': self.global_step,
            'lr_multipliers': self._lr_multipliers.clone(),
            'ema_loss': self.ema_loss.clone(),
            'cuda_graph_replay': True,
        }

    def _compute_surrogate_loss(
        self,
        mlp_output: Tensor,
        grad: Tensor,
        prev_output: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute surrogate loss for training DirectUpdateMLP during CMS mode.

        This is the key training signal for the MLP - it learns to output updates
        that align with gradient direction while preserving appropriate magnitude.

        Based on erikl2/nested-learning deep_momentum.py approach:
        1. Cosine similarity loss: output should point in same direction as gradient
        2. Magnitude preservation: output magnitude should match gradient magnitude
        3. Temporal smoothness (optional): output shouldn't change too fast

        Args:
            mlp_output: Output from DirectUpdateMLP (same shape as grad)
            grad: Current gradient (target direction)
            prev_output: Previous MLP output for temporal smoothness (optional)

        Returns:
            Scalar loss tensor for backpropagation
        """
        # Component 1: Reconstruction/Direction loss (cosine similarity)
        # MLP output should point in the same direction as the gradient
        mlp_flat = mlp_output.flatten()
        grad_flat = grad.flatten()

        grad_norm = grad_flat.norm()
        output_norm = mlp_flat.norm()

        if grad_norm > 1e-8 and output_norm > 1e-8:
            cosine_sim = F.cosine_similarity(
                mlp_flat.unsqueeze(0),
                grad_flat.unsqueeze(0),
            )
            reconstruction_loss = 1.0 - cosine_sim.mean()
        else:
            reconstruction_loss = torch.tensor(0.0, device=grad.device, requires_grad=True)

        # Component 2: Magnitude preservation
        # Output magnitude should be similar to gradient magnitude
        # This ensures MLP doesn't learn to output tiny or huge updates
        if grad_norm > 1e-8:
            magnitude_ratio = output_norm / grad_norm
            # Penalize deviation from ratio of 1.0
            magnitude_loss = (magnitude_ratio - 1.0).pow(2)
        else:
            # If gradient is near-zero, penalize large outputs
            magnitude_loss = output_norm.pow(2)

        # Component 3: Temporal smoothness (optional)
        # Prevents MLP from oscillating wildly between steps
        temporal_loss = torch.tensor(0.0, device=grad.device, requires_grad=True)
        if prev_output is not None:
            prev_flat = prev_output.flatten()
            if prev_flat.shape == mlp_flat.shape:
                # Penalize large changes in output
                temporal_loss = (mlp_flat - prev_flat).pow(2).mean() * 0.1

        # Combine losses with appropriate weighting
        # Reconstruction is primary, magnitude is secondary, temporal is tertiary
        total_loss = reconstruction_loss + 0.1 * magnitude_loss + temporal_loss

        return total_loss

    def set_loss(self, loss_value: float):
        """
        Set loss value for next step (GradScaler compatibility).

        Call this before scaler.step() when using mixed precision.

        Args:
            loss_value: Current loss value (scalar)
        """
        self._pending_loss = loss_value

    def step(self, loss_value: Optional[float] = None):
        """
        Perform optimization step with Continuum Memory System (CMS).

        Implements multi-frequency updates per the Nested Learning paper:
        - Level 0: Updates every step (fast adaptation)
        - Level 1: Updates every 10 steps (short-term patterns)
        - Level 2: Updates every 100 steps (long-term consolidation)

        Each level accumulates gradients and applies learned momentum transformation.
        Slower levels preserve consolidated knowledge (anti-forgetting mechanism).

        CUDA Graph Mode (use_cuda_graph=True):
        - Warmup: First cuda_graph_warmup_steps run normally
        - Capture: Graph is captured after warmup
        - Replay: Subsequent steps replay the captured graph

        Args:
            loss_value: Current loss (required for controller updates)

        Returns:
            Dict with step info including lr_multipliers, ema_loss, etc.

        Reference: NVIDIA CUDA Programming Guide, Section 3.2.8 "CUDA Graphs"
        """
        self.global_step += 1

        # Get loss value
        actual_loss = loss_value if loss_value is not None else self._pending_loss
        if actual_loss is None:
            if not hasattr(self, '_warned_no_loss'):
                warnings.warn(
                    "DeepNestedOptimizer: No loss_value provided. "
                    "Call optimizer.set_loss(value) before step() or pass directly."
                )
                self._warned_no_loss = True
            actual_loss = 0.0

        self._pending_loss = None

        # NaN guard on loss value
        if actual_loss is not None and (math.isnan(actual_loss) or math.isinf(actual_loss)):
            print(f"  [NaN GUARD] step {self.global_step}: loss value is {actual_loss}, skipping update")
            self.base_optimizer.zero_grad(set_to_none=True)
            return {
                'global_step': self.global_step,
                'lr_multipliers': self._lr_multipliers.tolist() if self._lr_multipliers is not None else [1.0, 1.0],
                'skipped': True,
                'reason': 'nan_loss'
            }

        # === CUDA Graph Path ===
        # Only used for non-CMS (AdamW) mode to avoid complexity of graph-capturing CMS
        if self.use_cuda_graph and not self.use_cms_updates:
            # Warmup phase: run normally to stabilize tensor shapes
            if self.global_step <= self.cuda_graph_warmup_steps:
                # Fall through to normal execution below
                pass
            # Capture phase: capture graph on first post-warmup step
            elif not self._cuda_graph_captured:
                # Save RNG state before capture attempt
                rng_state = torch.cuda.get_rng_state()

                capture_succeeded = False
                try:
                    # Ensure we're on the default stream before capture
                    torch.cuda.synchronize()
                    # Update LR tensors BEFORE capture so graph uses correct values
                    if self._graph_lr_tensors is not None:
                        for i in range(len(self._graph_lr_tensors)):
                            if len(self.base_optimizer.param_groups[i]['params']) > 0:
                                new_lr = self.base_lr * self._lr_multipliers[i]
                                self._graph_lr_tensors[i].fill_(new_lr)
                    self._capture_cuda_graph()
                    # Ensure capture is complete and we're back on default stream
                    torch.cuda.synchronize()
                    capture_succeeded = True
                except Exception as e:
                    warnings.warn(
                        f"CUDA Graph capture failed: {e}. "
                        "Falling back to eager mode. This may indicate "
                        "incompatible operations or insufficient GPU memory."
                    )
                    self.use_cuda_graph = False
                    self._cuda_graph_captured = False

                    # CRITICAL: Restore RNG state to recover from capture failure
                    # Failed graph capture can corrupt CUDA RNG offset tracking
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.set_rng_state(rng_state)
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                    # Fall through to normal execution

                # After successful capture, the first replay already happened inside
                # _capture_cuda_graph(), so we can return early with results
                if capture_succeeded:
                    # Update EMA loss for the capture step
                    self.ema_loss = (1 - self.beta_ema) * self.ema_loss + self.beta_ema * actual_loss

                    return {
                        'global_step': self.global_step,
                        'lr_multipliers': self._lr_multipliers.clone(),
                        'ema_loss': self.ema_loss.clone(),
                        'cuda_graph_capture': True,
                    }

            # Replay phase: use captured graph
            elif self._cuda_graph_captured:
                # Simple mode meta-update check (outside graph)
                result = self._replay_cuda_graph(actual_loss)

                # Simple mode: auto meta-update
                if self.mode == 'simple' and self.global_step % self.meta_update_freq == 0:
                    self._update_meta_components(actual_loss)

                return result

        # === Normal (Eager) Execution Path ===

        # Update EMA loss
        if self.global_step == 1:
            self.ema_loss.fill_(actual_loss)
        else:
            self.ema_loss = (1 - self.beta_ema) * self.ema_loss + self.beta_ema * actual_loss

        # Compute gradient statistics and LR multipliers only on controller update steps
        # This reduces overhead when controller_update_freq > 1
        if self.global_step % self.controller_update_freq == 0 or self.global_step == 1:
            # Compute gradient statistics for controller
            stats = self._compute_group_stats()

            # Get LR multipliers from controller
            with torch.no_grad():
                self._lr_multipliers = self.controller(stats)
        # Else: reuse self._lr_multipliers from previous controller update

        # Clip gradients
        # Use inplace=True only for CUDA graph mode (fast but breaks autograd graph)
        # Use inplace=False for normal mode (preserves graph for meta-learning)
        if self.max_grad_norm > 0:
            use_inplace = self.use_cuda_graph and self._cuda_graph_captured
            grad_norm = _fused_clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
                inplace=use_inplace
            )
            # Check for NaN/Inf gradients after clipping
            # Use tensor comparison directly (no .item() sync!)
            grad_is_nan = torch.isnan(grad_norm) | torch.isinf(grad_norm)
            if grad_is_nan:
                # Only sync for print if we're actually logging (rare case)
                # Note: This branch is only taken when training has diverged
                print(f"  [NaN GUARD] step {self.global_step}: gradient norm is NaN/Inf, skipping update")
                self.base_optimizer.zero_grad(set_to_none=True)
                return {
                    'global_step': self.global_step,
                    'lr_multipliers': self._lr_multipliers.tolist() if self._lr_multipliers is not None else [1.0, 1.0],
                    'skipped': True,
                    'reason': 'nan_gradient'
                }

        if not self.use_cms_updates:
            # === ADAMW MODE (proven to work) ===
            # Update base optimizer learning rates with controller's multipliers
            if self._graph_lr_tensors is not None:
                # CUDA Graph mode: use tensor LRs (update in-place)
                for i in range(len(self._graph_lr_tensors)):
                    if len(self.base_optimizer.param_groups[i]['params']) > 0:
                        new_lr = self.base_lr * self._lr_multipliers[i]
                        self._graph_lr_tensors[i].fill_(new_lr)
            else:
                # Non-graph mode: use scalar LRs
                effective_lrs = (self.base_lr * self._lr_multipliers).tolist()
                for i, group in enumerate(self.base_optimizer.param_groups):
                    if len(group['params']) > 0:
                        group['lr'] = effective_lrs[i]

            # Let AdamW handle the actual update (has adaptive per-param LR)
            self.base_optimizer.step()
        else:
            # === CMS MODE (experimental) ===
            # Get context for momentum MLP
            context = self._get_context(actual_loss)

            # CMS-based parameter updates (replaces base_optimizer.step())
            # Pre-compute effective LRs (one GPU op instead of per-group .item())
            effective_lrs = (self.base_lr * self._lr_multipliers).tolist()

            # === PHASE 1: Train DirectUpdateMLP via surrogate loss ===
            # This must happen WITH gradients before we apply updates
            should_train_mlp = (
                self.use_preprocessing and
                self.mlp_optimizer is not None and
                self.global_step % self.mlp_train_freq == 0
            )

            if should_train_mlp:
                # Sample a subset of parameters for MLP training (for efficiency)
                # We sample ~10% or max 20 parameters
                training_samples = []
                sample_count = 0
                max_samples = 20

                for group_idx, group in enumerate(self.base_optimizer.param_groups):
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        # Sample every 10th parameter, or all if small model
                        if sample_count < max_samples:
                            cms = self.state[param]
                            # Only train on level 0 (fast level) for efficiency
                            if cms.should_update(0, self.global_step):
                                training_samples.append((param, cms))
                                sample_count += 1
                        else:
                            break
                    if sample_count >= max_samples:
                        break

                # Compute surrogate losses for sampled parameters
                if training_samples:
                    surrogate_losses = []

                    for param, cms in training_samples:
                        grad = param.grad.detach()  # Detach from main model graph
                        # IMPORTANT: Detach accumulated grad to prevent gradient graph leak
                        level_grad = cms.levels[0]['accumulated_grad'].detach().clone()
                        freq = self.cms_frequencies[0] if len(self.cms_frequencies) > 0 else 1
                        level_grad = level_grad / max(freq, 1)
                        prev_momentum = cms.get_momentum(0).detach()

                        # Forward through MLP WITH gradients
                        mlp_output = self.momentum_mlp(level_grad, prev_momentum)

                        # Get previous output for temporal smoothness
                        param_id = id(param)
                        prev_output = self._mlp_output_history.get(param_id, None)

                        # Compute surrogate loss
                        loss = self._compute_surrogate_loss(mlp_output, level_grad, prev_output)
                        surrogate_losses.append(loss)

                        # Store output for next step's temporal smoothness
                        self._mlp_output_history[param_id] = mlp_output.detach().clone()

                    # Update MLP weights
                    if surrogate_losses:
                        total_surrogate_loss = torch.stack(surrogate_losses).mean()

                        self.mlp_optimizer.zero_grad()
                        total_surrogate_loss.backward()

                        # Clip MLP gradients for stability (fused)
                        _fused_clip_grad_norm_(self.momentum_mlp.parameters(), max_norm=1.0)

                        self.mlp_optimizer.step()

                        # Store as tensor for lazy eval - only convert to .item() when logging
                        self._last_surrogate_loss = total_surrogate_loss.detach()

            # === PHASE 2: Apply parameter updates (no gradients needed) ===
            with torch.no_grad():
                for group_idx, group in enumerate(self.base_optimizer.param_groups):
                    lr = effective_lrs[group_idx]

                    for param in group['params']:
                        if param.grad is None:
                            continue

                        grad = param.grad
                        cms = self.state[param]

                        # Accumulate gradient into all CMS levels
                        cms.accumulate_grad(grad)

                        # Compute update from each active level
                        total_update = torch.zeros_like(param)
                        num_active_levels = 0

                        for level in range(len(self.cms_frequencies)):
                            if cms.should_update(level, self.global_step):
                                # Get accumulated gradient for this level
                                level_grad = cms.get_update(level)
                                prev_momentum = cms.get_momentum(level)

                                # CRITICAL: Normalize by accumulation steps to get average gradient
                                freq = self.cms_frequencies[level] if level < len(self.cms_frequencies) else 1
                                level_grad = level_grad / max(freq, 1)

                                # Paper-aligned: Update EMA gradient target for L2 regression
                                cms.update_ema_grad(level, level_grad)

                                # Paper-aligned: Update second-moment for adaptive per-param LR
                                cms.update_second_moment(level, level_grad)

                                # Apply learned momentum transformation
                                # Branch based on MLP type (preprocessing vs legacy)
                                if self.use_preprocessing:
                                    # DirectUpdateMLP: outputs per-element update directly
                                    # The MLP takes gradient and momentum, outputs update
                                    level_update = self.momentum_mlp(level_grad, prev_momentum)

                                    # Update momentum with exponential decay + new update
                                    # This is similar to Adam's approach: track running average
                                    beta1 = 0.9  # Momentum coefficient
                                    new_momentum = beta1 * prev_momentum + (1 - beta1) * level_update

                                    # Soft clamp momentum at 100x gradient magnitude
                                    grad_norm = level_grad.norm().clamp(min=1e-8)
                                    momentum_norm = new_momentum.norm()
                                    if momentum_norm > 100.0 * grad_norm:
                                        new_momentum = new_momentum * (100.0 * grad_norm / momentum_norm)

                                    cms.update_momentum(level, new_momentum)

                                    # Use the MLP output directly as the level update
                                    # No damping factor needed - MLP learns to output proper magnitude
                                    level_weight = 1.0
                                else:
                                    # Legacy L2RegressionMomentum: outputs scalar coefficients
                                    scale, shift, damping = self.momentum_mlp(
                                        level_grad, prev_momentum, context
                                    )

                                    # Update momentum: v = scale * v_prev + shift * grad
                                    # MLP outputs scale in [0.5, 0.99] via sigmoid*0.49+0.5
                                    # Scale < 1.0 ensures momentum converges (no exponential growth)
                                    new_momentum = scale * prev_momentum + shift * level_grad

                                    # Soft clamp at 100x gradient magnitude (safety net, not primary control)
                                    grad_norm = level_grad.norm().clamp(min=1e-8)
                                    momentum_norm = new_momentum.norm()
                                    if momentum_norm > 100.0 * grad_norm:
                                        new_momentum = new_momentum * (100.0 * grad_norm / momentum_norm)

                                    cms.update_momentum(level, new_momentum)

                                    # Equal weighting for all levels
                                    level_weight = 1.0

                                    # Compute level contribution with damping
                                    level_update = new_momentum * (1 - damping) * level_weight

                                # NaN check
                                if torch.isnan(level_update).any() or torch.isinf(level_update).any():
                                    continue  # Skip this level's contribution

                                total_update += level_update
                                num_active_levels += 1

                        # Apply combined update if any levels were active
                        if num_active_levels > 0:
                            # Use sqrt(n) normalization to prevent magnitude explosion
                            # when multiple levels fire simultaneously
                            if num_active_levels > 1:
                                geom_factor = num_active_levels ** 0.5  # sqrt(n)
                                total_update = total_update / geom_factor

                            # Final NaN/Inf check
                            if torch.isnan(total_update).any() or torch.isinf(total_update).any():
                                continue  # Skip this parameter entirely

                            # Weight decay
                            if self.weight_decay > 0:
                                total_update += self.weight_decay * param

                            # Apply update with learning rate
                            param.add_(total_update, alpha=-lr)

        # Simple mode: auto meta-update
        if self.mode == 'simple' and self.global_step % self.meta_update_freq == 0:
            self._update_meta_components(actual_loss)

        # Build return dict with surrogate loss info if available
        result = {
            'global_step': self.global_step,
            'lr_multipliers': self._lr_multipliers.clone(),
            'ema_loss': self.ema_loss.clone(),
        }

        # Add surrogate loss if we trained the MLP this step
        if hasattr(self, '_last_surrogate_loss'):
            result['surrogate_loss'] = self._last_surrogate_loss

        return result

    def _update_meta_components(self, loss_value: float):
        """
        Update MomentumMLP and Controller via simplified meta-learning.

        Both components are now included in the computation graph:
        - Controller: trained via proxy loss on LR multipliers
        - MomentumMLP: trained via proxy loss on scale/shift/damping outputs

        The proxy objective rewards outputs that correlate with loss improvement.
        """
        # Record step in simplified trainer
        momentum_stats = self.get_momentum_stats()
        self.simplified_meta_trainer.record_step(
            loss=loss_value,
            multipliers=self._lr_multipliers,
            momentum_norm=momentum_stats.get('momentum_avg_norm', 0.0),
        )

        # Compute gradient statistics
        stats = self._compute_group_stats()

        self.meta_optimizer.zero_grad()

        # === Controller Loss ===
        multipliers = self.controller(stats)
        controller_loss = self.simplified_meta_trainer.compute_proxy_loss(
            current_multipliers=multipliers,
            current_loss=loss_value,
        )

        # === MomentumMLP Loss ===
        # Sample a few parameters to get representative MLP outputs
        # This puts the MLP in the computation graph so it gets gradients
        mlp_loss = self._compute_mlp_proxy_loss(loss_value)

        # Combined meta-loss
        meta_loss = controller_loss + mlp_loss

        meta_loss.backward()

        # Keep as tensor - only convert to .item() when logging to avoid GPU sync
        self._controller_grad_norm_tensor = _fused_clip_grad_norm_(
            list(self.momentum_mlp.parameters()) + list(self.controller.parameters()),
            max_norm=1.0,
        )
        self.controller_grad_norm = self._controller_grad_norm_tensor  # Lazy eval

        self.meta_optimizer.step()
        self._last_meta_loss_tensor = meta_loss.detach()
        self.last_meta_loss = self._last_meta_loss_tensor  # Lazy eval

    def _compute_mlp_proxy_loss(self, loss_value: float) -> Tensor:
        """
        Compute L2 regression loss for MomentumMLP training (paper-aligned).

        Per the Nested Learning paper:
        - target = EMA of gradients (what standard momentum would compute)
        - predicted = MLP-transformed update
        - loss = ||predicted_stats - target_stats||^2

        For DirectUpdateMLP (use_preprocessing=True):
        - Compares MLP output directly to EMA target

        For L2RegressionMomentum (use_preprocessing=False):
        - Compares scalar-combined momentum to EMA target
        """
        # Check if we have enough history
        if len(self.simplified_meta_trainer.loss_history) < 10:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Get context for legacy MLP (not needed for DirectUpdateMLP)
        context = self._get_context(loss_value)

        # Sample a subset of parameters for efficiency (max 10)
        sampled_params = []
        for group in self.param_groups:
            for param in group['params'][:5]:  # First 5 from each group
                if param.grad is not None:
                    sampled_params.append(param)
        sampled_params = sampled_params[:10]

        if not sampled_params:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for param in sampled_params:
            cms = self.state[param]
            # Detach grad from model's computation graph
            # We want gradients through MLP, not through model params
            grad = param.grad.detach()

            # Level 0 (fast) is always active and has the freshest EMA
            # Detach momentum to avoid graph through CMS state
            prev_momentum = cms.get_momentum(0).detach()
            ema_target = cms.get_ema_grad(0).detach()

            # Skip if EMA hasn't been initialized yet
            if ema_target.abs().max() < 1e-10:
                continue

            if self.use_preprocessing:
                # DirectUpdateMLP: outputs per-element update directly
                # Forward through MLP WITH gradients
                predicted_update = self.momentum_mlp(grad, prev_momentum)

                # L2 regression: ||predicted - target||^2
                # For efficiency, compare statistics (mean, std, norm)
                pred_stats = self.momentum_mlp.compute_stats(predicted_update.flatten())
                target_stats = self.momentum_mlp.compute_stats(ema_target.flatten())

                # L2 loss on statistics (3-dim vector: mean, std, norm)
                param_loss = (pred_stats - target_stats).pow(2).sum()
            else:
                # Legacy L2RegressionMomentum: outputs scalar coefficients
                scale, shift, damping = self.momentum_mlp(grad, prev_momentum, context)

                # Predicted momentum from MLP (what CMS would produce)
                predicted_momentum = scale * prev_momentum + shift * grad

                # Paper-aligned L2 regression: ||predicted - target||^2
                # Compare statistics for efficiency (mean, std, norm)
                pred_stats = self.momentum_mlp.compute_stats(predicted_momentum.flatten())
                target_stats = self.momentum_mlp.compute_stats(ema_target.flatten())

                # L2 loss on statistics (3-dim vector: mean, std, norm)
                param_loss = (pred_stats - target_stats).pow(2).sum()

            total_loss = total_loss + param_loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        # Add regularization toward good defaults when loss is stagnating
        recent_improvement = (
            self.simplified_meta_trainer.loss_history[0] -
            self.simplified_meta_trainer.loss_history[-1]
        )

        if recent_improvement < 0.001:
            # Loss is stagnating - add exploration bonus
            # Small penalty to push away from stuck configurations
            total_loss = total_loss + 0.01

        return total_loss

    def meta_update(
        self,
        val_batch: Dict[str, Tensor],
        train_batches: Optional[List[Dict[str, Tensor]]] = None,
        loss_fn: Optional[Callable] = None,
        use_unrolled: bool = True,
    ):
        """
        Explicit meta-learning update using validation data.

        Uses k-step unrolled differentiation:
        1. Clone model state
        2. Run k inner optimization steps on train data
        3. Compute validation loss
        4. Backprop through unrolled steps to update MLP + Controller

        Args:
            val_batch: Validation batch for meta-objective
            train_batches: List of k training batches for inner steps.
                          If None, uses val_batch for simplified update.
            loss_fn: Loss function(model, batch) -> loss.
                    If None, uses model.forward() and expects 'loss' key.
            use_unrolled: If True, uses full k-step unrolling.
                         If False, uses simplified proxy loss.

        Note: Full unrolled mode is computationally expensive. Call sparingly.
        """
        if loss_fn is None:
            def loss_fn(model, batch):
                output = model(batch)
                if isinstance(output, dict):
                    return output.get('loss', output.get('total', torch.tensor(0.0)))
                return output

        if use_unrolled and train_batches is not None and len(train_batches) >= self.k_unroll:
            # Full k-step unrolled meta-learning
            self.meta_optimizer.zero_grad()

            meta_loss = self.unrolled_meta_trainer.compute_meta_loss(
                model=self.model,
                optimizer_components={
                    'momentum_mlp': self.momentum_mlp,
                    'controller': self.controller,
                },
                train_batches=train_batches,
                val_batch=val_batch,
                loss_fn=loss_fn,
                base_lr=self.base_lr,
            )

            meta_loss.backward()

            # Keep as tensor - only convert to .item() when logging to avoid GPU sync
            self._controller_grad_norm_tensor = _fused_clip_grad_norm_(
                list(self.momentum_mlp.parameters()) + list(self.controller.parameters()),
                max_norm=1.0,
            )
            self.controller_grad_norm = self._controller_grad_norm_tensor  # Lazy eval

            self.meta_optimizer.step()
            self._last_meta_loss_tensor = meta_loss.detach()
            self.last_meta_loss = self._last_meta_loss_tensor  # Lazy eval

        else:
            # Simplified: just use validation loss as proxy
            # NOTE: We don't use torch.no_grad() here because some models
            # (like TitanMAC with neural memory) use torch.autograd.grad()
            # internally and need gradient tracking enabled.
            # We just compute the loss value without backpropagating.

            # Clear cache before forward pass to reduce OOM risk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Use smaller batch if provided batch is large (reduce memory)
            if isinstance(val_batch, dict) and 'input_ids' in val_batch:
                batch_size = val_batch['input_ids'].size(0)
                if batch_size > 4:
                    # Use only first 4 samples to reduce memory
                    val_batch = {
                        k: v[:4] if torch.is_tensor(v) else v
                        for k, v in val_batch.items()
                    }

            val_loss = loss_fn(self.model, val_batch)

            # Detach to avoid any gradient accumulation from this forward pass
            if hasattr(val_loss, 'detach'):
                val_loss_value = val_loss.detach().item()
            else:
                val_loss_value = float(val_loss)

            # Clear any gradients that accumulated during the forward pass
            self.base_optimizer.zero_grad(set_to_none=True)

            # Clear cache after forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._update_meta_components(val_loss_value)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def get_lr_multipliers(self) -> Tensor:
        """Get current LR multipliers per group."""
        return self._lr_multipliers.clone()

    def get_effective_lrs(self) -> List[float]:
        """Get effective learning rates per group."""
        return (self.base_lr * self._lr_multipliers).tolist()

    def get_momentum_stats(self) -> Dict[str, float]:
        """Get statistics about momentum states for logging.

        Uses batched GPU computation to avoid per-tensor .item() calls.
        """
        # Collect all momentum norms on GPU
        momentum_norms = []
        for param, state in self.state.items():
            for level in range(state.num_levels):
                m = state.get_momentum(level)
                momentum_norms.append(m.pow(2).sum())

        if momentum_norms:
            # Single sqrt and sum on GPU, one .item() at the end
            stacked = torch.stack(momentum_norms)
            total_norm = stacked.sqrt().sum().item()
            count = len(momentum_norms)
        else:
            total_norm = 0.0
            count = 1

        return {
            'momentum_total_norm': total_norm,
            'momentum_avg_norm': total_norm / max(count, 1),
            'global_step': float(self.global_step),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        # Serialize CMS state (including paper-aligned additions)
        cms_state = {}
        for i, (param, state) in enumerate(self.state.items()):
            cms_state[i] = {
                level: {
                    'momentum': state.levels[level]['momentum'].clone(),
                    'step_count': state.levels[level]['step_count'],
                    'accumulated_grad': state.levels[level]['accumulated_grad'].clone(),
                    # Paper-aligned additions
                    'ema_grad': state.levels[level]['ema_grad'].clone(),
                    'v_sq': state.levels[level]['v_sq'].clone(),
                }
                for level in state.levels
            }

        state = {
            'base_optimizer': self.base_optimizer.state_dict(),
            'momentum_mlp': self.momentum_mlp.state_dict(),
            'controller': self.controller.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'global_step': self.global_step,
            'lr_multipliers': self._lr_multipliers.clone(),
            'ema_loss': self.ema_loss.clone(),
            'cms_state': cms_state,
        }

        # Save MLP optimizer state if it exists (CMS mode with preprocessing)
        if self.mlp_optimizer is not None:
            state['mlp_optimizer'] = self.mlp_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.momentum_mlp.load_state_dict(state_dict['momentum_mlp'])
        self.controller.load_state_dict(state_dict['controller'])
        self.meta_optimizer.load_state_dict(state_dict['meta_optimizer'])
        self.global_step = state_dict['global_step']
        self._lr_multipliers = state_dict.get('lr_multipliers', torch.ones(self.n_groups))
        self.ema_loss = state_dict.get('ema_loss', torch.zeros(self.n_groups))

        # Restore CMS state (including paper-aligned additions)
        if 'cms_state' in state_dict:
            cms_state = state_dict['cms_state']
            for i, (param, state) in enumerate(self.state.items()):
                if i in cms_state:
                    for level, level_data in cms_state[i].items():
                        state.levels[level]['momentum'] = level_data['momentum'].to(self.device)
                        state.levels[level]['step_count'] = level_data['step_count']
                        state.levels[level]['accumulated_grad'] = level_data['accumulated_grad'].to(self.device)
                        # Paper-aligned additions (with backwards compatibility)
                        state.levels[level]['ema_grad'] = level_data.get(
                            'ema_grad',
                            torch.zeros_like(state.levels[level]['momentum'])
                        ).to(self.device)
                        state.levels[level]['v_sq'] = level_data.get(
                            'v_sq',
                            torch.zeros_like(state.levels[level]['momentum'])
                        ).to(self.device)

        # Restore MLP optimizer state if it exists
        if 'mlp_optimizer' in state_dict and self.mlp_optimizer is not None:
            self.mlp_optimizer.load_state_dict(state_dict['mlp_optimizer'])

    @property
    def param_groups(self) -> List[Dict]:
        """Access parameter groups (for compatibility)."""
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set parameter groups."""
        self._param_groups = value
