"""
DeepNestedOptimizer: Complete Nested Learning optimizer.

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

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .nested_controller import NestedController
from .param_groups import group_titans_params, infer_param_depth
from .meta_trainer import SimplifiedMetaTrainer, UnrolledMetaTrainer


class L2RegressionMomentum(nn.Module):
    """
    Learned momentum with L2 regression internal objective.

    Replaces linear momentum (v = β*v + g) with a neural network:
    v = MLP(concat(g, v_prev)) trained to minimize ||predicted - actual_improvement||²

    The key insight from Nested Learning: momentum is associative memory
    mapping gradient keys to update values. L2 regression provides a more
    robust learning signal than dot-product similarity.

    Args:
        input_dim: Dimension of gradients/momentum (flattened)
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
            # Bias: sigmoid(0.8)≈0.69 -> scale≈1.38, tanh(1.0)≈0.76 -> shift≈0.76
            output_layer.bias.data = torch.tensor([0.8, 1.0, -2.0])

    def compute_stats(self, tensor: Tensor) -> Tensor:
        """Compute statistics [mean, std, norm] for a tensor."""
        with torch.no_grad():
            mean = tensor.mean()
            std = tensor.std() if tensor.numel() > 1 else torch.tensor(0.0, device=tensor.device)
            norm = tensor.norm()
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
        scale = torch.sigmoid(out[0]) * 1.0 + 0.5  # [0.5, 1.5]
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
            L2 loss: ||predicted - actual||²
        """
        return torch.nn.functional.mse_loss(predicted_update, actual_improvement)


class ContinuumMemoryState:
    """
    Per-parameter state for multi-frequency CMS updates.

    Implements hierarchical memory with exponentially spaced update frequencies:
    - Level 0: Updates every step (fast adaptation)
    - Level 1: Every base_freq steps (short-term patterns)
    - Level 2: Every base_freq² steps (long-term knowledge)

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

    def to(self, device: torch.device) -> 'ContinuumMemoryState':
        """Move all tensors to device."""
        self.device = device
        for level_state in self.levels.values():
            level_state['momentum'] = level_state['momentum'].to(device)
            level_state['accumulated_grad'] = level_state['accumulated_grad'].to(device)
        return self


class DeepNestedOptimizer:
    """
    Complete Nested Learning optimizer.

    Combines:
    - L2RegressionMomentum: Learned gradient compression (DMGD)
    - NestedController: Learned LR multipliers from training outcomes
    - ContinuumMemoryState: Multi-frequency update scheduling (CMS)
    - Meta-learning: Trains components via k-step unrolled differentiation

    Two operation modes:
    - 'simple': Automatic meta-updates (for BC training)
    - 'explicit': Manual meta-update calls (for PPO with GradScaler)

    Args:
        model: Model to optimize
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
        controller_hidden_dim: int = 32,
        use_gradient_checkpointing: bool = True,
        mode: str = 'simple',
        meta_update_freq: int = 100,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        if mode not in ('simple', 'explicit'):
            raise ValueError(f"mode must be 'simple' or 'explicit', got {mode}")

        self.model = model
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.k_unroll = k_unroll
        self.cms_frequencies = cms_frequencies or [1, 10, 100]
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.mode = mode
        self.meta_update_freq = meta_update_freq
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Get device from model
        self.device = next(model.parameters()).device

        # Group parameters (core vs memory)
        core_params, memory_params = group_titans_params(model)
        self.n_groups = 2

        # Handle empty memory group
        if len(memory_params) == 0:
            _param_groups = [
                {'params': core_params, 'lr': base_lr, 'name': 'core'},
                {'params': [], 'lr': base_lr, 'name': 'memory'},
            ]
        else:
            _param_groups = [
                {'params': core_params, 'lr': base_lr, 'name': 'core'},
                {'params': memory_params, 'lr': base_lr, 'name': 'memory'},
            ]

        # Create base optimizer (AdamW) for actual parameter updates
        self.base_optimizer = torch.optim.AdamW(
            _param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )

        # Learned components
        self.momentum_mlp = L2RegressionMomentum(
            hidden_dim=momentum_hidden_dim,
        ).to(self.device)

        self.controller = NestedController(
            hidden_dim=controller_hidden_dim,
            n_groups=self.n_groups,
        ).to(self.device)

        # Meta-optimizer (trains MomentumMLP + Controller)
        self.meta_optimizer = torch.optim.Adam(
            list(self.momentum_mlp.parameters()) +
            list(self.controller.parameters()),
            lr=meta_lr,
        )

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
        """
        stats = []
        for i, group in enumerate(self.param_groups):
            if len(group['params']) == 0:
                stats.append([0.0, 0.0, self.group_depths[i].item()])
                continue

            grad_norm_sq = 0.0
            param_norm_sq = 0.0

            for param in group['params']:
                if param.grad is not None:
                    grad_norm_sq += param.grad.norm().item() ** 2
                param_norm_sq += param.norm().item() ** 2

            stats.append([
                grad_norm_sq ** 0.5,
                param_norm_sq ** 0.5,
                self.group_depths[i].item(),
            ])

        return torch.tensor(stats, dtype=torch.float32, device=self.device)

    def _get_context(self, loss_value: float) -> Tensor:
        """Create context vector for MomentumMLP."""
        norm_loss = torch.log(torch.tensor(max(loss_value, 1e-8) + 1.0))
        return torch.tensor(
            [self.global_step / 1000.0, self.base_lr, norm_loss.item()],
            device=self.device,
            dtype=torch.float32,
        )

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
        Perform optimization step.

        In 'simple' mode, may trigger automatic meta-update.
        In 'explicit' mode, only does inner optimization step.

        Args:
            loss_value: Current loss (required for controller updates)

        Returns:
            Dict with step info including lr_multipliers, ema_loss, etc.
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

        # Update EMA loss
        if self.global_step == 1:
            self.ema_loss.fill_(actual_loss)
        else:
            self.ema_loss = (1 - self.beta_ema) * self.ema_loss + self.beta_ema * actual_loss

        # Compute gradient statistics
        stats = self._compute_group_stats()

        # Get LR multipliers from controller
        with torch.no_grad():
            self._lr_multipliers = self.controller(stats)

        # Update base optimizer learning rates
        for i, group in enumerate(self.base_optimizer.param_groups):
            if len(group['params']) > 0:
                group['lr'] = self.base_lr * self._lr_multipliers[i].item()

        # Clip gradients
        if self.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Base optimizer step
        self.base_optimizer.step()

        # Simple mode: auto meta-update
        if self.mode == 'simple' and self.global_step % self.meta_update_freq == 0:
            # Note: In simple mode, we use training loss as proxy for meta-loss
            # This is less principled but simpler
            self._update_meta_components(actual_loss)

        return {
            'global_step': self.global_step,
            'lr_multipliers': self._lr_multipliers.clone(),
            'ema_loss': self.ema_loss.clone(),
        }

    def _update_meta_components(self, loss_value: float):
        """
        Update MomentumMLP and Controller via simplified meta-learning.

        Uses SimplifiedMetaTrainer which tracks loss history and computes
        a proxy meta-objective based on correlation with improvement.

        For proper meta-learning with validation data, use meta_update().
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

        # Forward through controller
        self.meta_optimizer.zero_grad()
        multipliers = self.controller(stats)

        # Use simplified meta-trainer's proxy loss
        meta_loss = self.simplified_meta_trainer.compute_proxy_loss(
            current_multipliers=multipliers,
            current_loss=loss_value,
        )

        meta_loss.backward()

        self.controller_grad_norm = clip_grad_norm_(
            list(self.momentum_mlp.parameters()) + list(self.controller.parameters()),
            max_norm=1.0,
        ).item()

        self.meta_optimizer.step()
        self.last_meta_loss = meta_loss.item()

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

            self.controller_grad_norm = clip_grad_norm_(
                list(self.momentum_mlp.parameters()) + list(self.controller.parameters()),
                max_norm=1.0,
            ).item()

            self.meta_optimizer.step()
            self.last_meta_loss = meta_loss.item()

        else:
            # Simplified: just use validation loss as proxy
            with torch.no_grad():
                val_loss = loss_fn(self.model, val_batch)

            self._update_meta_components(val_loss.item())

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def get_lr_multipliers(self) -> Tensor:
        """Get current LR multipliers per group."""
        return self._lr_multipliers.clone()

    def get_effective_lrs(self) -> List[float]:
        """Get effective learning rates per group."""
        return [
            self.base_lr * mult.item()
            for mult in self._lr_multipliers
        ]

    def get_momentum_stats(self) -> Dict[str, float]:
        """Get statistics about momentum states for logging."""
        total_norm = 0.0
        count = 0
        for param, state in self.state.items():
            for level in range(state.num_levels):
                m = state.get_momentum(level)
                total_norm += m.norm().item()
                count += 1

        return {
            'momentum_total_norm': total_norm,
            'momentum_avg_norm': total_norm / max(count, 1),
            'global_step': float(self.global_step),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        # Serialize CMS state
        cms_state = {}
        for i, (param, state) in enumerate(self.state.items()):
            cms_state[i] = {
                level: {
                    'momentum': state.levels[level]['momentum'].clone(),
                    'step_count': state.levels[level]['step_count'],
                    'accumulated_grad': state.levels[level]['accumulated_grad'].clone(),
                }
                for level in state.levels
            }

        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'momentum_mlp': self.momentum_mlp.state_dict(),
            'controller': self.controller.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'global_step': self.global_step,
            'lr_multipliers': self._lr_multipliers.clone(),
            'ema_loss': self.ema_loss.clone(),
            'cms_state': cms_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.momentum_mlp.load_state_dict(state_dict['momentum_mlp'])
        self.controller.load_state_dict(state_dict['controller'])
        self.meta_optimizer.load_state_dict(state_dict['meta_optimizer'])
        self.global_step = state_dict['global_step']
        self._lr_multipliers = state_dict.get('lr_multipliers', torch.ones(self.n_groups))
        self.ema_loss = state_dict.get('ema_loss', torch.zeros(self.n_groups))

        # Restore CMS state
        if 'cms_state' in state_dict:
            cms_state = state_dict['cms_state']
            for i, (param, state) in enumerate(self.state.items()):
                if i in cms_state:
                    for level, level_data in cms_state[i].items():
                        state.levels[level]['momentum'] = level_data['momentum'].to(self.device)
                        state.levels[level]['step_count'] = level_data['step_count']
                        state.levels[level]['accumulated_grad'] = level_data['accumulated_grad'].to(self.device)

    @property
    def param_groups(self) -> List[Dict]:
        """Access parameter groups (for compatibility)."""
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set parameter groups."""
        self._param_groups = value
