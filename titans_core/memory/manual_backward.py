"""
Manual Backward Pass for Neural Memory MLP.

This module provides a pure PyTorch implementation of backward pass
without using torch.autograd.grad(). This enables:
1. torch.compile(fullgraph=True) compatibility
2. CUDA graphs compatibility
3. donated_buffer optimization

The manual backward is numerically equivalent to autograd but avoids
the problematic retain_graph=True requirement.

Usage:
    Instead of:
        grads = torch.autograd.grad(loss, mlp.parameters(), retain_graph=True)

    Use:
        grads = manual_mlp_backward(k, v, z1, h, y, W1, B1, W2, B2)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def silu_backward(z: torch.Tensor) -> torch.Tensor:
    """
    Compute derivative of SiLU activation.

    SiLU: silu(x) = x * sigmoid(x)
    SiLU': silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Args:
        z: Pre-activation tensor (input to silu, NOT output)

    Returns:
        Derivative of silu at z
    """
    sigma = torch.sigmoid(z)
    return sigma * (1.0 + z * (1.0 - sigma))


def manual_mlp_backward(
    k: torch.Tensor,  # [N, D_in] keys (input to MLP)
    v: torch.Tensor,  # [N, D_out] target values
    z1: torch.Tensor,  # [N, H] pre-activation (before silu)
    h: torch.Tensor,  # [N, H] hidden (after silu)
    y: torch.Tensor,  # [N, D_out] output
    W1: torch.Tensor,  # [H, D_in] layer 1 weight
    B1: torch.Tensor,  # [H] layer 1 bias
    W2: torch.Tensor,  # [D_out, H] layer 2 weight
    B2: torch.Tensor,  # [D_out] layer 2 bias
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients of MSE loss w.r.t. MLP parameters.

    This replaces torch.autograd.grad() with manual backpropagation,
    enabling fullgraph=True compilation.

    Forward pass (for reference):
        z1 = k @ W1.T + B1     # [N, H]
        h = silu(z1)           # [N, H]
        y = h @ W2.T + B2      # [N, D_out]
        loss = MSE(y, v)       # scalar

    Args:
        k: Keys [N, D_in]
        v: Target values [N, D_out]
        z1: Pre-activation from layer 1 [N, H]
        h: Hidden activation [N, H]
        y: MLP output [N, D_out]
        W1, B1: Layer 1 parameters
        W2, B2: Layer 2 parameters

    Returns:
        Tuple of (d_W1, d_B1, d_W2, d_B2) gradients
    """
    N, D_out = y.shape
    _, H = h.shape
    _, D_in = k.shape

    # ==================== Step 1: d_loss/d_y ====================
    # MSE loss: L = (1/N/D) * sum((y - v)^2)
    # dL/dy = (2/N/D) * (y - v)
    d_y = (2.0 / (N * D_out)) * (y - v)  # [N, D_out]

    # ==================== Step 2: Layer 2 backward ====================
    # y = h @ W2.T + B2
    # dL/dW2 = (dL/dy).T @ h   (chain rule + matmul derivative)
    # dL/dB2 = sum(dL/dy, dim=0)
    # dL/dh = dL/dy @ W2

    d_W2 = d_y.T @ h  # [D_out, H]
    d_B2 = d_y.sum(dim=0)  # [D_out]
    d_h = d_y @ W2  # [N, H]

    # ==================== Step 3: SiLU backward ====================
    # h = silu(z1)
    # dL/dz1 = dL/dh * silu'(z1)

    silu_grad = silu_backward(z1)  # [N, H]
    d_z1 = d_h * silu_grad  # [N, H]

    # ==================== Step 4: Layer 1 backward ====================
    # z1 = k @ W1.T + B1
    # dL/dW1 = (dL/dz1).T @ k
    # dL/dB1 = sum(dL/dz1, dim=0)

    d_W1 = d_z1.T @ k  # [H, D_in]
    d_B1 = d_z1.sum(dim=0)  # [H]

    return d_W1, d_B1, d_W2, d_B2


def forward_with_intermediates(
    k: torch.Tensor,
    W1: torch.Tensor,
    B1: torch.Tensor,
    W2: torch.Tensor,
    B2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass through 2-layer MLP, returning intermediates for backward.

    Args:
        k: Input keys [N, D_in]
        W1, B1: Layer 1 parameters
        W2, B2: Layer 2 parameters

    Returns:
        Tuple of (z1, h, y):
        - z1: Pre-activation [N, H]
        - h: Hidden activation [N, H]
        - y: Output [N, D_out]
    """
    z1 = F.linear(k, W1, B1)  # [N, H]
    h = F.silu(z1)  # [N, H]
    y = F.linear(h, W2, B2)  # [N, D_out]
    return z1, h, y


def verify_against_autograd(
    k: torch.Tensor,
    v: torch.Tensor,
    W1: torch.Tensor,
    B1: torch.Tensor,
    W2: torch.Tensor,
    B2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """
    Verify that manual backward matches torch.autograd.grad().

    Args:
        k, v: Input keys and target values
        W1, B1, W2, B2: MLP parameters (must have requires_grad=True for autograd)
        rtol, atol: Tolerance for allclose comparison

    Returns:
        True if all gradients match within tolerance

    Raises:
        AssertionError if gradients don't match
    """
    # Ensure parameters require grad for autograd comparison
    W1_copy = W1.clone().detach().requires_grad_(True)
    B1_copy = B1.clone().detach().requires_grad_(True)
    W2_copy = W2.clone().detach().requires_grad_(True)
    B2_copy = B2.clone().detach().requires_grad_(True)

    # Forward pass for autograd
    z1_auto = F.linear(k, W1_copy, B1_copy)
    h_auto = F.silu(z1_auto)
    y_auto = F.linear(h_auto, W2_copy, B2_copy)
    loss_auto = F.mse_loss(y_auto, v)

    # Autograd backward
    grads_auto = torch.autograd.grad(
        loss_auto,
        [W1_copy, B1_copy, W2_copy, B2_copy],
        retain_graph=False,
    )

    # Forward pass for manual (use same activations)
    z1_manual, h_manual, y_manual = forward_with_intermediates(k, W1, B1, W2, B2)

    # Manual backward
    grads_manual = manual_mlp_backward(k, v, z1_manual, h_manual, y_manual, W1, B1, W2, B2)

    # Compare
    names = ["d_W1", "d_B1", "d_W2", "d_B2"]
    all_match = True
    for name, g_auto, g_manual in zip(names, grads_auto, grads_manual):
        if not torch.allclose(g_auto, g_manual, rtol=rtol, atol=atol):
            max_diff = (g_auto - g_manual).abs().max().item()
            print(f"MISMATCH in {name}: max_diff = {max_diff}")
            all_match = False
        else:
            print(f"OK: {name} matches")

    return all_match


class ManualBackwardMemoryUpdate:
    """
    Memory update using manual backward pass (no autograd.grad).

    This is a drop-in replacement for the autograd-based update in NeuralMemory.
    Compatible with torch.compile(fullgraph=True).

    Example:
        updater = ManualBackwardMemoryUpdate()

        # In update():
        z1, h, y = forward_with_intermediates(k, W1, B1, W2, B2)
        loss = F.mse_loss(y, v)

        with torch.no_grad():
            grad_norm = updater(
                k, v, z1, h, y,
                W1, B1, W2, B2,
                momentum_S,
                alpha_t, eta_t, theta,
            )
    """

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        z1: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        W1: torch.Tensor,
        B1: torch.Tensor,
        W2: torch.Tensor,
        B2: torch.Tensor,
        momentum_S: torch.Tensor,
        alpha_t: float,
        eta_t: float,
        theta: float,
        max_grad_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute gradients, apply clipping, momentum, and weight update.

        Args:
            k: Keys [N, D_in]
            v: Target values [N, D_out]
            z1, h, y: Saved activations from forward pass
            W1, B1, W2, B2: MLP parameters (will be updated in-place)
            momentum_S: Momentum buffer [n_params] (will be updated in-place)
            alpha_t: Forget gate output (0=retain, 1=forget)
            eta_t: Decay gate output (momentum decay)
            theta: Memory learning rate
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            grad_norm: Gradient norm before clipping
        """
        # Compute gradients manually
        d_W1, d_B1, d_W2, d_B2 = manual_mlp_backward(k, v, z1, h, y, W1, B1, W2, B2)

        # Flatten gradients
        flat_grad = torch.cat(
            [
                d_W1.view(-1),
                d_B1.view(-1),
                d_W2.view(-1),
                d_B2.view(-1),
            ]
        )

        # Gradient clipping
        grad_norm = flat_grad.norm()
        if grad_norm > max_grad_norm:
            flat_grad = flat_grad * (max_grad_norm / grad_norm)

        # Momentum update: S = η * S - θ * g
        momentum_S.mul_(eta_t)
        momentum_S.add_(flat_grad, alpha=-theta)

        # Weight update: W = (1 - α) * W + S
        # Unflatten momentum and apply to each parameter
        offset = 0
        for p in [W1, B1, W2, B2]:
            numel = p.numel()
            p.mul_(1.0 - alpha_t)
            p.add_(momentum_S[offset : offset + numel].view(p.shape))
            offset += numel

        return grad_norm


# =============================================================================
# Functional API (for torch.func compatibility)
# =============================================================================


def functional_memory_loss(
    params: dict,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute memory loss as a pure function (for torch.func.grad).

    Args:
        params: Dict with keys 'W1', 'B1', 'W2', 'B2'
        k: Keys [N, D_in]
        v: Target values [N, D_out]

    Returns:
        MSE loss scalar
    """
    z1 = F.linear(k, params["W1"], params["B1"])
    h = F.silu(z1)
    y = F.linear(h, params["W2"], params["B2"])
    return F.mse_loss(y, v)


# =============================================================================
# torch.func.grad Implementation (RECOMMENDED)
# =============================================================================
#
# VERIFIED: torch.func.grad DOES work with torch.compile(fullgraph=True)!
# This is the recommended approach - simpler than manual backward, and
# enables fullgraph compilation.

from torch.func import grad  # noqa: E402


def _memory_loss_fn(W1, B1, W2, B2, k, v):
    """
    Pure function computing MLP forward + MSE loss.

    Args order matters for grad(argnums=...).
    First 4 args are parameters we want gradients for.
    """
    z1 = F.linear(k, W1, B1)
    h = F.silu(z1)
    y = F.linear(h, W2, B2)
    return F.mse_loss(y, v)


# Create grad function for first 4 arguments (the weights)
_grad_memory_loss = grad(_memory_loss_fn, argnums=(0, 1, 2, 3))


class FunctionalGradMemoryUpdate:
    """
    Memory update using torch.func.grad.

    RECOMMENDED APPROACH: Works with torch.compile(fullgraph=True)!

    This is simpler than manual backward and doesn't require saving
    intermediate activations (z1, h) - just needs k, v and weights.

    Example:
        updater = FunctionalGradMemoryUpdate(compile=True)

        # In update():
        loss = updater.compute_loss(k, W1, B1, W2, B2, v)

        with torch.no_grad():
            grad_norm = updater.update_weights(
                k, v,
                W1, B1, W2, B2,
                momentum_S,
                alpha_t, eta_t, theta,
            )
    """

    def __init__(self, compile: bool = True):
        """
        Args:
            compile: If True, compile the grad function with fullgraph=True
        """
        self.compile = compile
        self._compiled_grad_fn = None

    def _get_grad_fn(self):
        """Get the gradient function (compile on first use)."""
        if self._compiled_grad_fn is not None:
            return self._compiled_grad_fn

        if self.compile:
            self._compiled_grad_fn = torch.compile(
                _grad_memory_loss,
                fullgraph=True,
                dynamic=False,
            )
        else:
            self._compiled_grad_fn = _grad_memory_loss

        return self._compiled_grad_fn

    def compute_loss(
        self,
        k: torch.Tensor,
        W1: torch.Tensor,
        B1: torch.Tensor,
        W2: torch.Tensor,
        B2: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss (forward pass)."""
        z1 = F.linear(k, W1, B1)
        h = F.silu(z1)
        y = F.linear(h, W2, B2)
        return F.mse_loss(y, v)

    def update_weights(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        W1: torch.Tensor,
        B1: torch.Tensor,
        W2: torch.Tensor,
        B2: torch.Tensor,
        momentum_S: torch.Tensor,
        alpha_t: float,
        eta_t: float,
        theta: float,
        max_grad_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute gradients via torch.func.grad and update weights in-place.

        This method:
        1. Computes gradients using torch.func.grad (fullgraph compatible!)
        2. Clips gradients by norm
        3. Updates momentum: S = η * S - θ * g
        4. Updates weights: W = (1 - α) * W + S

        Args:
            k: Keys [N, D_in]
            v: Target values [N, D_out]
            W1, B1, W2, B2: MLP parameters (will be updated in-place)
            momentum_S: Momentum buffer [n_params] (will be updated in-place)
            alpha_t: Forget gate output (0=retain, 1=forget)
            eta_t: Decay gate output (momentum decay)
            theta: Memory learning rate
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            grad_norm: Gradient norm before clipping
        """
        # Get compiled grad function
        grad_fn = self._get_grad_fn()

        # Compute gradients (this is the magic - fullgraph compatible!)
        d_W1, d_B1, d_W2, d_B2 = grad_fn(W1, B1, W2, B2, k, v)

        # Flatten gradients
        flat_grad = torch.cat(
            [
                d_W1.view(-1),
                d_B1.view(-1),
                d_W2.view(-1),
                d_B2.view(-1),
            ]
        )

        # Gradient clipping
        grad_norm = flat_grad.norm()
        if grad_norm > max_grad_norm:
            flat_grad = flat_grad * (max_grad_norm / grad_norm)

        # Momentum update: S = η * S - θ * g
        momentum_S.mul_(eta_t)
        momentum_S.add_(flat_grad, alpha=-theta)

        # Weight update: W = (1 - α) * W + S
        offset = 0
        for p in [W1, B1, W2, B2]:
            numel = p.numel()
            p.mul_(1.0 - alpha_t)
            p.add_(momentum_S[offset : offset + numel].view(p.shape))
            offset += numel

        return grad_norm


def verify_func_grad_against_autograd(
    k: torch.Tensor,
    v: torch.Tensor,
    W1: torch.Tensor,
    B1: torch.Tensor,
    W2: torch.Tensor,
    B2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """
    Verify that torch.func.grad matches torch.autograd.grad.

    Returns True if all gradients match within tolerance.
    """
    # Method 1: autograd.grad
    W1_copy = W1.clone().detach().requires_grad_(True)
    B1_copy = B1.clone().detach().requires_grad_(True)
    W2_copy = W2.clone().detach().requires_grad_(True)
    B2_copy = B2.clone().detach().requires_grad_(True)

    z1 = F.linear(k, W1_copy, B1_copy)
    h = F.silu(z1)
    y = F.linear(h, W2_copy, B2_copy)
    loss = F.mse_loss(y, v)

    grads_auto = torch.autograd.grad(
        loss,
        [W1_copy, B1_copy, W2_copy, B2_copy],
        retain_graph=False,
    )

    # Method 2: torch.func.grad
    grads_func = _grad_memory_loss(W1, B1, W2, B2, k, v)

    # Compare
    names = ["d_W1", "d_B1", "d_W2", "d_B2"]
    all_match = True
    for name, g_auto, g_func in zip(names, grads_auto, grads_func):
        if not torch.allclose(g_auto, g_func, rtol=rtol, atol=atol):
            max_diff = (g_auto - g_func).abs().max().item()
            print(f"MISMATCH in {name}: max_diff = {max_diff}")
            all_match = False
        else:
            print(f"OK: {name} matches (func.grad)")

    return all_match


if __name__ == "__main__":
    # Quick verification test
    print("=" * 60)
    print("Running verification tests...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Device: {device}")

    N, D, H = 256, 128, 128

    k = torch.randn(N, D, device=device, dtype=dtype)
    v = torch.randn(N, D, device=device, dtype=dtype)
    W1 = torch.randn(H, D, device=device, dtype=dtype) * 0.02
    B1 = torch.zeros(H, device=device, dtype=dtype)
    W2 = torch.randn(D, H, device=device, dtype=dtype) * 0.02
    B2 = torch.zeros(D, device=device, dtype=dtype)

    # Test 1: Manual backward vs autograd
    print("\n[Test 1] Manual backward vs autograd.grad:")
    success1 = verify_against_autograd(k, v, W1, B1, W2, B2)
    print(f"Result: {'PASS' if success1 else 'FAIL'}")

    # Test 2: torch.func.grad vs autograd
    print("\n[Test 2] torch.func.grad vs autograd.grad:")
    success2 = verify_func_grad_against_autograd(k, v, W1, B1, W2, B2)
    print(f"Result: {'PASS' if success2 else 'FAIL'}")

    # Test 3: torch.func.grad with torch.compile(fullgraph=True)
    print("\n[Test 3] torch.func.grad with torch.compile(fullgraph=True):")
    try:
        compiled_grad_fn = torch.compile(_grad_memory_loss, fullgraph=True, dynamic=False)
        grads = compiled_grad_fn(W1, B1, W2, B2, k, v)
        print("Compilation succeeded!")
        print(f"Gradient shapes: {[g.shape for g in grads]}")
        success3 = True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        success3 = False
    print(f"Result: {'PASS' if success3 else 'FAIL'}")

    # Summary
    print("\n" + "=" * 60)
    all_success = success1 and success2 and success3
    print(f"Overall: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'}")
    print("=" * 60)

    if all_success:
        print("\nRecommended approach: FunctionalGradMemoryUpdate")
        print("  - Uses torch.func.grad")
        print("  - Works with torch.compile(fullgraph=True)")
        print("  - Simpler than manual backward")
        print("  - Enables CUDA graphs and donated_buffer")
