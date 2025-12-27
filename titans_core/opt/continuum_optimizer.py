"""
ContinuumOptimizer: Nested optimizer with controller-modulated learning rates.

Wraps a base optimizer with a small controller network that adjusts
per-group learning rates based on gradient dynamics.

Key features:
- EMA loss tracking per group
- Controller-predicted LR multipliers
- Multi-timescale learning (core vs memory params)

Tasks: T048-T051
"""

from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

from .nested_controller import NestedController
from .param_groups import group_titans_params, infer_param_depth
from .cms import ContinuumMemorySystem
from .meta_trainer import SimplifiedMetaTrainer


class ContinuumOptimizer:
    """
    Nested optimizer with controller-modulated learning rates.

    Wraps a base optimizer (AdamW) with a small controller network
    that adjusts per-group learning rates based on gradient dynamics.

    Args:
        model: Model to optimize
        base_lr: Base learning rate
        update_freq: Controller update frequency in optimizer steps
        beta_meta: EMA smoothing factor for meta-objective
        controller_lr: Learning rate for controller optimizer
        min_lr_mult: Minimum LR multiplier
        max_lr_mult: Maximum LR multiplier
        base_optim_cls: Base optimizer class (default: AdamW)
        base_optim_kwargs: Additional kwargs for base optimizer

    Example:
        >>> model = TitanMAC(config)
        >>> optimizer = ContinuumOptimizer(
        ...     model=model,
        ...     base_lr=5e-4,
        ...     update_freq=50,
        ... )
        >>> for batch in dataloader:
        ...     loss = model(batch)["loss"]
        ...     loss.backward()
        ...     optimizer.step(loss.item())
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 5e-4,
        update_freq: int = 50,
        beta_meta: float = 0.1,
        controller_lr: float = 1e-3,
        min_lr_mult: float = 0.1,
        max_lr_mult: float = 2.0,
        base_optim_cls: type = torch.optim.AdamW,
        base_optim_kwargs: Optional[Dict] = None,
        use_cms: bool = False,
        cms_frequencies: Optional[Dict[int, int]] = None,
    ):
        self.model = model
        self.base_lr = base_lr
        self.update_freq = update_freq
        self.beta_meta = beta_meta
        self.min_lr_mult = min_lr_mult
        self.max_lr_mult = max_lr_mult
        self.use_cms = use_cms

        # Group parameters
        core_params, memory_params = group_titans_params(model)

        # Handle case where there are no memory params
        if len(memory_params) == 0:
            # Add a dummy group with empty params
            self.param_groups_list = [
                {"params": core_params, "lr": base_lr},
                {"params": [], "lr": base_lr},  # Empty memory group
            ]
            self.n_groups = 2
            self._has_memory = False
        else:
            self.param_groups_list = [
                {"params": core_params, "lr": base_lr},
                {"params": memory_params, "lr": base_lr},
            ]
            self.n_groups = 2
            self._has_memory = True

        # Create base optimizer
        base_optim_kwargs = base_optim_kwargs or {}
        self.base_optimizer = base_optim_cls(
            self.param_groups_list,
            lr=base_lr,
            **base_optim_kwargs,
        )

        # Create controller
        self.controller = NestedController(
            hidden_dim=32,
            min_lr_mult=min_lr_mult,
            max_lr_mult=max_lr_mult,
            n_groups=self.n_groups,
        )

        # Move controller to same device as model
        device = next(model.parameters()).device
        self.controller = self.controller.to(device)

        # Controller optimizer
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(),
            lr=controller_lr,
        )

        # State tracking
        self.global_step = 0
        self.ema_loss = torch.zeros(self.n_groups)
        self._lr_multipliers = torch.ones(self.n_groups)
        self._pending_loss = None  # For set_loss() / step() workflow
        self.last_meta_loss = None  # For compatibility with DeepNestedOptimizer interface

        # Compute parameter depths for each group
        self._compute_group_depths()

        # Optional Continuum Memory System (multi-frequency updates)
        if self.use_cms:
            # Default frequencies if not provided
            if cms_frequencies is None:
                cms_frequencies = {
                    0: 1,  # Core params: update every step
                    1: 10,  # Memory params: update every 10 steps
                }

            self.cms = ContinuumMemorySystem(
                param_groups=self.param_groups_list,
                frequencies=cms_frequencies,
            )
        else:
            self.cms = None

        # Simplified meta-trainer for proper meta-loss computation
        # This replaces the broken pure-regularization approach
        self.meta_trainer = SimplifiedMetaTrainer(
            window_size=100,
            improvement_threshold=0.001,
        )

    def set_loss(self, loss_value: float) -> None:
        """Set the loss value for the next controller update.

        Call this method before step() when using with GradScaler or other
        wrappers that don't pass loss_value to step().

        Args:
            loss_value: Current loss value (scalar)

        Example:
            >>> optimizer.set_loss(loss.item())
            >>> scaler.step(optimizer)
            >>> scaler.update()
        """
        self._pending_loss = loss_value

    def _compute_group_depths(self):
        """Compute average depth for each parameter group."""
        # Get number of layers from model
        if hasattr(self.model, "config") and hasattr(self.model.config, "n_layers"):
            n_layers = self.model.config.n_layers
        elif hasattr(self.model, "layers"):
            n_layers = len(self.model.layers)
        else:
            n_layers = 1  # Default

        self.group_depths = []

        for group in self.param_groups_list:
            if len(group["params"]) == 0:
                # Empty group (dummy memory group)
                self.group_depths.append(0.5)
                continue

            depths = []
            for param in group["params"]:
                # Find parameter name
                for name, p in self.model.named_parameters():
                    if p is param:
                        depth = infer_param_depth(name, n_layers)
                        depths.append(depth)
                        break

            avg_depth = sum(depths) / len(depths) if depths else 0.5
            self.group_depths.append(avg_depth)

        self.group_depths = torch.tensor(self.group_depths)

    def step(self, loss_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform optimization step.

        Args:
            loss_value: Current loss value (required for controller updates)

        Returns:
            Dictionary with step info:
            - "global_step": Current optimizer step
            - "lr_multipliers": Current LR multipliers per group (if updated)
            - "ema_loss": Current EMA loss per group (if updated)
            - "controller_updated": Whether controller was updated this step
            - "cms_groups_updated": List of group indices updated (if CMS enabled)

        Raises:
            ValueError: If loss_value is None when controller update needed

        Note:
            Call after loss.backward() and before zero_grad().
        """
        self.global_step += 1

        result = {
            "global_step": self.global_step,
            "controller_updated": False,
        }

        # Check if we should update controller
        should_update = self.global_step % self.update_freq == 0

        if should_update:
            # Use stored loss if not provided directly
            actual_loss = loss_value if loss_value is not None else self._pending_loss

            if actual_loss is None:
                # Skip controller update if no loss available (warn once)
                if not hasattr(self, "_warned_no_loss"):
                    import warnings

                    warnings.warn(
                        f"ContinuumOptimizer: No loss_value available at step {self.global_step}. "
                        "Controller update skipped. Call optimizer.set_loss(value) before step() "
                        "or pass loss_value directly to step().",
                        UserWarning,
                    )
                    self._warned_no_loss = True
            else:
                # Update controller with available loss
                update_info = self._update_controller(actual_loss)
                result.update(update_info)
                result["controller_updated"] = True
                self._pending_loss = None  # Clear after use

        # Perform base optimizer step
        if self.use_cms and self.cms is not None:
            # Multi-frequency updates via CMS
            groups_to_update = self.cms.step()
            result["cms_groups_updated"] = groups_to_update

            # Only step parameters in groups that should update
            # Save original requires_grad and temporarily disable others
            original_requires_grad = []
            for i, group in enumerate(self.base_optimizer.param_groups):
                group_requires_grad = []
                for param in group["params"]:
                    group_requires_grad.append(param.requires_grad)
                    # Disable grad for groups not updating
                    if i not in groups_to_update:
                        param.requires_grad = False
                original_requires_grad.append(group_requires_grad)

            # Step optimizer (only updates enabled groups)
            self.base_optimizer.step()

            # Restore requires_grad
            for i, group in enumerate(self.base_optimizer.param_groups):
                for j, param in enumerate(group["params"]):
                    param.requires_grad = original_requires_grad[i][j]
        else:
            # Standard update (all groups)
            self.base_optimizer.step()

        return result

    def _update_controller(self, loss_value: float) -> Dict[str, Any]:
        """
        Update controller and LR multipliers.

        Args:
            loss_value: Current loss value

        Returns:
            Dictionary with update info
        """
        # Update EMA loss for each group
        # Note: In practice, we track the same loss for both groups
        # A more sophisticated version could compute per-group losses
        for i in range(self.n_groups):
            if self.global_step == self.update_freq:
                # First update: initialize EMA
                self.ema_loss[i] = loss_value
            else:
                # EMA update: ℓ̄_g(t) = (1 - β) * ℓ̄_g(t-1) + β * ℓ_t
                self.ema_loss[i] = (1 - self.beta_meta) * self.ema_loss[
                    i
                ] + self.beta_meta * loss_value

        # Compute gradient statistics for each group
        stats = self._compute_group_stats()

        # Get LR multipliers from controller
        device = next(self.controller.parameters()).device
        stats = stats.to(device)

        with torch.enable_grad():
            multipliers = self.controller(stats)

        # Store multipliers
        self._lr_multipliers = multipliers.detach()

        # Update base optimizer learning rates
        for i, group in enumerate(self.base_optimizer.param_groups):
            if len(group["params"]) > 0:  # Skip empty groups
                group["lr"] = self.base_lr * self._lr_multipliers[i].item()

        # Record step in meta-trainer for loss-improvement tracking
        self.meta_trainer.record_step(
            loss=loss_value,
            multipliers=multipliers.detach(),
            momentum_norm=0.0,  # Not tracking momentum in ContinuumOptimizer
        )

        # Controller meta-objective: Use loss-improvement-based proxy loss
        # This replaces the broken pure-regularization approach
        meta_loss = self.meta_trainer.compute_proxy_loss(
            current_multipliers=multipliers,
            current_loss=loss_value,
        )
        self.last_meta_loss = meta_loss.detach()  # Store for logging

        # Update controller
        self.controller_optimizer.zero_grad()
        meta_loss.backward()

        # Compute controller grad norm for logging
        controller_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(),
            max_norm=1.0,
        )

        self.controller_optimizer.step()

        return {
            "lr_multipliers": self._lr_multipliers.clone(),
            "ema_loss": self.ema_loss.clone(),
            "controller_grad_norm": controller_grad_norm.item(),
        }

    def _compute_group_stats(self) -> torch.Tensor:
        """
        Compute gradient statistics for each parameter group.

        Uses torch._foreach_norm for batched GPU computation, avoiding
        per-parameter .item() calls that cause CPU-GPU synchronization.

        Returns:
            Tensor of shape [n_groups, 3] with normalized stats:
            - log_grad_norm: Log-scaled gradient norm (handles large range)
            - log_param_norm: Log-scaled parameter norm
            - avg_depth: Average depth of parameters (already in [0, 1])
        """
        device = next(self.model.parameters()).device
        stats = torch.zeros(self.n_groups, 3, device=device, dtype=torch.float32)

        for i, group in enumerate(self.base_optimizer.param_groups):
            # Set depth (already a tensor)
            stats[i, 2] = self.group_depths[i]

            if len(group["params"]) == 0:
                continue

            # Collect gradients and params as flat views for batched norm
            grads = [p.grad.view(-1) for p in group["params"] if p.grad is not None]
            params = [p.view(-1) for p in group["params"]]

            # Batched gradient norm using fused kernel
            if grads:
                grad_norms = torch._foreach_norm(grads)
                grad_norm_sq = torch.stack(grad_norms).pow(2).sum()
                grad_norm = grad_norm_sq.sqrt()
                # Log-scale on GPU: log1p(x) / 10 -> ~[0, 1] for typical norms
                stats[i, 0] = torch.log1p(grad_norm) / 10.0

            # Batched param norm using fused kernel
            param_norms = torch._foreach_norm(params)
            param_norm_sq = torch.stack(param_norms).pow(2).sum()
            param_norm = param_norm_sq.sqrt()
            stats[i, 1] = torch.log1p(param_norm) / 10.0

        return stats

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients for base optimizer.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self) -> List[Dict]:
        """Access base optimizer parameter groups."""
        return self.base_optimizer.param_groups

    def state_dict(self) -> Dict[str, Any]:
        """
        Get optimizer state for checkpointing.

        Returns:
            Dictionary with:
            - "base_optimizer": Base optimizer state dict
            - "controller": Controller network state dict
            - "controller_optimizer": Controller optimizer state dict
            - "global_step": Current step counter
            - "ema_loss": EMA loss values
            - "cms": CMS state dict (if CMS enabled)
        """
        state = {
            "base_optimizer": self.base_optimizer.state_dict(),
            "controller": self.controller.state_dict(),
            "controller_optimizer": self.controller_optimizer.state_dict(),
            "global_step": self.global_step,
            "ema_loss": self.ema_loss.clone(),
            "lr_multipliers": self._lr_multipliers.clone(),
            "use_cms": self.use_cms,
        }

        # Include CMS state if enabled
        if self.use_cms and self.cms is not None:
            state["cms"] = self.cms.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state from checkpoint.

        Args:
            state_dict: State dictionary from state_dict()
        """
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.controller.load_state_dict(state_dict["controller"])
        self.controller_optimizer.load_state_dict(state_dict["controller_optimizer"])
        self.global_step = state_dict["global_step"]
        self.ema_loss = state_dict["ema_loss"]
        self._lr_multipliers = state_dict.get("lr_multipliers", torch.ones(self.n_groups))

        # Load CMS state if present
        if "cms" in state_dict and self.cms is not None:
            self.cms.load_state_dict(state_dict["cms"])

    def get_lr_multipliers(self) -> torch.Tensor:
        """
        Get current LR multipliers per group.

        Returns:
            Tensor of shape [n_groups] with current multipliers
        """
        return self._lr_multipliers.clone()

    def get_effective_lrs(self) -> List[float]:
        """
        Get effective learning rates per group.

        Returns:
            List of effective LRs (base_lr * multiplier per group)
        """
        return [self.base_lr * mult.item() for mult in self._lr_multipliers]
