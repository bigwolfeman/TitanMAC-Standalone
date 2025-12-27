"""Triton kernels for Block-ELL sparse backward pass.

This module provides GPU-accelerated backward pass for block-sparse matrix
multiplication using the Block-ELL format. The kernels compute:
- Gradient w.r.t. input (dx): for propagating gradients back
- Gradient w.r.t. values (dW): for weight updates
- Gradient w.r.t. bias (db): if bias is present

Shapes:
- R = out_features // tile_size (output block-rows)
- C = in_features // tile_size (input block-columns)
- K = active blocks per row
- B = tile_size (default 16)

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

# Triton import with fallback for CPU-only systems
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Triton kernel: Block-ELL sparse backward w.r.t. input
# =============================================================================


def _check_triton_available() -> None:
    """Raise error if Triton is not available."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is required for block_ell_backward kernels. " "Install with: pip install triton"
        )


if TRITON_AVAILABLE:

    # We use the reference implementation for backward dx since the Triton version
    # has compatibility issues with atomic operations. The dw kernel is simpler
    # and works correctly with Triton.
    pass  # backward dx kernel moved to reference implementation

    @triton.jit
    def block_ell_backward_dw_kernel(
        # Input tensor
        x_ptr,
        # Gradient w.r.t. output
        dout_ptr,
        # Column indices [R, K]
        col_indices_ptr,
        # Gradient w.r.t. values (output of this kernel)
        dvalues_ptr,
        # Dimensions
        batch_size,
        in_features,
        out_features,
        R,
        K,
        B: tl.constexpr,
        # Strides for x [batch, in_features]
        stride_x_batch,
        stride_x_feat,
        # Strides for dout [batch, out_features]
        stride_dout_batch,
        stride_dout_feat,
        # Strides for col_indices [R, K]
        stride_ci_r,
        stride_ci_k,
        # Strides for dvalues [R, K, B, B]
        stride_dv_r,
        stride_dv_k,
        stride_dv_b1,
        stride_dv_b2,
        # Block dimensions
        BLOCK_BATCH: tl.constexpr,
    ):
        """Block-ELL backward kernel for weight gradient.

        Computes: dW[r,k] = dout[:, r*B:(r+1)*B].T @ x[:, c*B:(c+1)*B]
        where c = col_indices[r, k]

        The forward was: y = x @ W^T, so dW = dout.T @ x

        Each program instance handles one block (r, k).

        Grid: (R, K)
        """
        # Program IDs
        pid_r = tl.program_id(0)  # Output block-row
        pid_k = tl.program_id(1)  # Block within row

        # Block indices
        out_b_offs = tl.arange(0, B)  # B_out dimension
        in_b_offs = tl.arange(0, B)  # B_in dimension

        # Load column index for this block
        col_idx_ptr = col_indices_ptr + pid_r * stride_ci_r + pid_k * stride_ci_k
        col_idx = tl.load(col_idx_ptr)

        # Initialize accumulator for dW: [B_out, B_in]
        dw_acc = tl.zeros((B, B), dtype=tl.float32)

        # Loop over batches in chunks
        for batch_start in range(0, batch_size, BLOCK_BATCH):
            batch_offs = batch_start + tl.arange(0, BLOCK_BATCH)
            batch_mask = batch_offs < batch_size

            # Load dout block: dout[batch_offs, pid_r*B : (pid_r+1)*B]
            # Shape: [BLOCK_BATCH, B_out]
            dout_block_ptr = (
                dout_ptr
                + batch_offs[:, None] * stride_dout_batch
                + (pid_r * B + out_b_offs[None, :]) * stride_dout_feat
            )
            dout_block = tl.load(dout_block_ptr, mask=batch_mask[:, None], other=0.0)

            # Load x block: x[batch_offs, col_idx*B : (col_idx+1)*B]
            # Shape: [BLOCK_BATCH, B_in]
            x_block_ptr = (
                x_ptr
                + batch_offs[:, None] * stride_x_batch
                + (col_idx * B + in_b_offs[None, :]) * stride_x_feat
            )
            x_block = tl.load(x_block_ptr, mask=batch_mask[:, None], other=0.0)

            # Accumulate outer product: dout.T @ x
            # dout_block.T: [B_out, BLOCK_BATCH]
            # x_block: [BLOCK_BATCH, B_in]
            # Result: [B_out, B_in]
            # Note: allow_tf32=False ensures full FP32 precision (~1e-7 error)
            # instead of TF32 which has ~1e-3 error due to mantissa truncation
            dw_acc += tl.dot(tl.trans(dout_block), x_block, allow_tf32=False)

        # Store dvalues[pid_r, pid_k, :, :]
        dv_ptr = (
            dvalues_ptr
            + pid_r * stride_dv_r
            + pid_k * stride_dv_k
            + out_b_offs[:, None] * stride_dv_b1
            + in_b_offs[None, :] * stride_dv_b2
        )
        tl.store(dv_ptr, dw_acc)


def block_ell_backward_dx(
    dout: Tensor,
    values: Tensor,
    col_indices: Tensor,
    in_features: int,
) -> Tensor:
    """Compute gradient w.r.t. input.

    Uses the reference implementation since the Triton kernel for dx has
    compatibility issues with atomic operations on modern Triton versions.

    Computes: dx = dout @ W where W is in Block-ELL format

    Args:
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]
        values: Block weight values [R, K, B, B]
        col_indices: Column indices [R, K]
        in_features: Input feature dimension

    Returns:
        Gradient w.r.t. input [batch, in_features] or [batch, seq, in_features]
    """
    R, K, B, _ = values.shape
    return block_ell_backward_dx_reference(dout, values, col_indices, in_features, tile_size=B)


def block_ell_backward_dw(
    x: Tensor,
    dout: Tensor,
    col_indices: Tensor,
    R: int,
    K: int,
    B: int,
) -> Tensor:
    """Compute gradient w.r.t. block values using Triton kernel.

    Computes: dW[r,k] = dout[:, r*B:(r+1)*B].T @ x[:, c*B:(c+1)*B]

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]
        col_indices: Column indices [R, K]
        R: Number of output block-rows
        K: Blocks per row
        B: Block size

    Returns:
        Gradient w.r.t. values [R, K, B, B]
    """
    _check_triton_available()

    in_features = x.shape[-1]
    out_features = dout.shape[-1]

    # Handle 2D vs 3D input - flatten batch dimensions
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(batch_size * seq_len, in_features)
        dout_flat = dout.view(batch_size * seq_len, out_features)
    else:
        x_flat = x
        dout_flat = dout

    total_batch = x_flat.shape[0]

    # Ensure tensors are contiguous
    x_flat = x_flat.contiguous()
    dout_flat = dout_flat.contiguous()
    col_indices = col_indices.contiguous()

    # Allocate dvalues
    dvalues = torch.zeros(R, K, B, B, dtype=x_flat.dtype, device=x_flat.device)

    # Choose BLOCK_BATCH for batching over samples
    if total_batch <= 16:
        BLOCK_BATCH = 16
    elif total_batch <= 32:
        BLOCK_BATCH = 32
    else:
        BLOCK_BATCH = 64

    # Grid: (R, K) - one program per block
    grid = (R, K)

    # Launch kernel
    block_ell_backward_dw_kernel[grid](
        x_flat,
        dout_flat,
        col_indices,
        dvalues,
        total_batch,
        in_features,
        out_features,
        R,
        K,
        B,
        x_flat.stride(0),
        x_flat.stride(1),
        dout_flat.stride(0),
        dout_flat.stride(1),
        col_indices.stride(0),
        col_indices.stride(1),
        dvalues.stride(0),
        dvalues.stride(1),
        dvalues.stride(2),
        dvalues.stride(3),
        BLOCK_BATCH=BLOCK_BATCH,
    )

    return dvalues


def block_ell_backward_db(dout: Tensor) -> Tensor:
    """Compute gradient w.r.t. bias.

    Computes: db = dout.sum(dim=0) over batch dimensions

    Args:
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]

    Returns:
        Gradient w.r.t. bias [out_features]
    """
    # Sum over all batch dimensions (everything except the last dimension)
    if dout.dim() == 2:
        # [batch, out_features] -> sum over batch
        return dout.sum(dim=0)
    elif dout.dim() == 3:
        # [batch, seq, out_features] -> sum over batch and seq
        return dout.sum(dim=(0, 1))
    else:
        # General case: sum over all but last dimension
        dims = tuple(range(dout.dim() - 1))
        return dout.sum(dim=dims)


def block_ell_backward(
    dout: Tensor,
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
    needs_bias_grad: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Combined backward pass for Block-ELL sparse layer.

    Computes all requested gradients in one call for efficiency.
    Uses reference implementations - Triton kernels called separately when available.

    Args:
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]
        x: Input tensor (saved from forward) [batch, in_features] or [batch, seq, in_features]
        values: Block weight values [R, K, B, B]
        col_indices: Column indices [R, K]
        needs_input_grad: Whether to compute dx
        needs_weight_grad: Whether to compute dW
        needs_bias_grad: Whether to compute db

    Returns:
        Tuple of (dx, dW, db), with None for gradients not requested
    """
    R, K, B, _ = values.shape
    in_features = x.shape[-1]

    dx = None
    dw = None
    db = None

    if needs_input_grad:
        dx = block_ell_backward_dx_reference(
            dout=dout,
            values=values,
            col_indices=col_indices,
            in_features=in_features,
            tile_size=B,
        )

    if needs_weight_grad:
        dw = block_ell_backward_dw_reference(
            x=x,
            dout=dout,
            col_indices=col_indices,
            R=R,
            K=K,
            tile_size=B,
        )

    if needs_bias_grad:
        db = block_ell_backward_db(dout)

    return dx, dw, db


# =============================================================================
# Reference implementations for testing
# =============================================================================


def block_ell_backward_dx_reference(
    dout: Tensor,
    values: Tensor,
    col_indices: Tensor,
    in_features: int,
    tile_size: int = 16,
) -> Tensor:
    """Reference (slow) implementation of dx backward.

    For testing and debugging only.

    Computes: dx = dout @ W where W is in Block-ELL format

    The forward pass was: y = x @ W^T + bias
    So for backward: dx = dout @ W

    For block-sparse: dx[:, c*B:(c+1)*B] += sum over (r,k) where col_indices[r,k]=c of:
        dout[:, r*B:(r+1)*B] @ values[r,k]

    Args:
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]
        values: Block weight values [R, K, B, B]
        col_indices: Column indices [R, K]
        in_features: Input feature dimension
        tile_size: Block size B

    Returns:
        Gradient w.r.t. input [batch, in_features] or [batch, seq, in_features]
    """
    R, K, B_out, B_in = values.shape
    assert B_out == B_in == tile_size

    # Handle 2D vs 3D input
    if dout.dim() == 2:
        dout = dout.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, seq_len, out_features = dout.shape
    C = in_features // tile_size

    # Reshape dout to block view: [batch, seq, R, B]
    dout_blocks = dout.view(batch_size, seq_len, R, tile_size)

    # Initialize dx: [batch, seq, C, B]
    dx_blocks = torch.zeros(batch_size, seq_len, C, tile_size, dtype=dout.dtype, device=dout.device)

    # For each output block-row r and each active column k:
    # dx[:, :, c, :] += dout[:, :, r, :] @ values[r, k, :, :]
    # where c = col_indices[r, k]
    for r in range(R):
        for k in range(K):
            c = col_indices[r, k].item()

            # dout_block: [batch, seq, B]
            dout_block = dout_blocks[:, :, r, :]

            # weight: [B_out, B_in] = [B, B]
            weight = values[r, k]

            # dx contribution: [batch, seq, B] = dout_block @ weight
            # dout_block: [batch, seq, B_out], weight: [B_out, B_in]
            # result: [batch, seq, B_in]
            dx_contrib = torch.einsum("bso,oi->bsi", dout_block, weight)

            # Scatter-add to the correct input column
            dx_blocks[:, :, c, :] = dx_blocks[:, :, c, :] + dx_contrib

    # Reshape dx: [batch, seq, C, B] -> [batch, seq, in_features]
    dx = dx_blocks.view(batch_size, seq_len, in_features)

    if squeeze_output:
        dx = dx.squeeze(1)

    return dx


def block_ell_backward_dw_reference(
    x: Tensor,
    dout: Tensor,
    col_indices: Tensor,
    R: int,
    K: int,
    tile_size: int = 16,
) -> Tensor:
    """Reference (slow) implementation of dW backward.

    For testing and debugging only.

    Computes: dW[r,k] = dout[:, r*B:(r+1)*B].T @ x[:, c*B:(c+1)*B]
    where c = col_indices[r, k]

    The forward pass was: y = x @ W^T
    So for backward w.r.t. W: dW = dout.T @ x
    In block form: dW[r,k] = dout_block[r].T @ x_block[c]

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        dout: Gradient w.r.t. output [batch, out_features] or [batch, seq, out_features]
        col_indices: Column indices [R, K]
        R: Number of output block-rows
        K: Blocks per row
        tile_size: Block size B

    Returns:
        Gradient w.r.t. values [R, K, B, B]
    """
    B = tile_size

    # Handle 2D vs 3D input - flatten batch dimensions for gradient computation
    if x.dim() == 2:
        x = x.unsqueeze(1)
        dout = dout.unsqueeze(1)

    batch_size, seq_len, in_features = x.shape
    _, _, out_features = dout.shape
    C = in_features // B

    # Flatten batch and seq: [batch * seq, features]
    x_flat = x.view(-1, in_features)
    dout_flat = dout.view(-1, out_features)
    N = x_flat.shape[0]  # batch * seq

    # Reshape to block views
    x_blocks = x_flat.view(N, C, B)  # [N, C, B]
    dout_blocks = dout_flat.view(N, R, B)  # [N, R, B]

    # Initialize dvalues: [R, K, B, B]
    dvalues = torch.zeros(R, K, B, B, dtype=x.dtype, device=x.device)

    # For each block (r, k):
    # dW[r,k] = sum over batch of: dout[:, r, :].T @ x[:, c, :]
    # where c = col_indices[r, k]
    for r in range(R):
        for k in range(K):
            c = col_indices[r, k].item()

            # dout_block: [N, B_out]
            dout_block = dout_blocks[:, r, :]  # [N, B]

            # x_block: [N, B_in]
            x_block = x_blocks[:, c, :]  # [N, B]

            # dW = dout.T @ x: [B_out, N] @ [N, B_in] = [B_out, B_in]
            # Using einsum: 'no,ni->oi'
            dvalues[r, k] = torch.einsum("no,ni->oi", dout_block, x_block)

    return dvalues


# =============================================================================
# Autograd Function
# =============================================================================


class BlockELLFunction(torch.autograd.Function):
    """Autograd function for Block-ELL sparse linear.

    Handles forward and backward passes with proper gradient computation.
    Uses Triton kernels when available, otherwise falls back to reference implementations.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        values: Tensor,
        col_indices: Tensor,
        bias: Optional[Tensor],
        in_features: int,
        R: int,
        K: int,
        B: int,
        use_triton: bool = True,
    ) -> Tensor:
        """Forward pass with save_for_backward.

        Args:
            ctx: Autograd context
            x: Input tensor [batch, in_features] or [batch, seq, in_features]
            values: Block weight values [R, K, B, B]
            col_indices: Column indices [R, K]
            bias: Optional bias [out_features]
            in_features: Input feature dimension
            R: Number of output block-rows
            K: Blocks per row
            B: Block size
            use_triton: Whether to use Triton kernels (default True)

        Returns:
            Output tensor
        """
        # Import forward function here to avoid circular imports
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        # Determine which implementation to use
        use_triton_impl = use_triton and TRITON_AVAILABLE and x.is_cuda and B in (8, 16, 32, 64)

        if use_triton_impl:
            output = block_ell_forward(x, values, col_indices, bias)
        else:
            output = block_ell_forward_reference(x, values, col_indices, bias, tile_size=B)

        # Save for backward
        ctx.save_for_backward(x, values, col_indices, bias)
        ctx.in_features = in_features
        ctx.R = R
        ctx.K = K
        ctx.B = B
        ctx.use_triton = use_triton_impl

        return output

    @staticmethod
    def backward(
        ctx, dout: Tensor
    ) -> Tuple[
        Optional[Tensor], Optional[Tensor], None, Optional[Tensor], None, None, None, None, None
    ]:
        """Backward pass computing gradients.

        Args:
            ctx: Autograd context with saved tensors
            dout: Gradient w.r.t. output

        Returns:
            Tuple of gradients: (dx, dvalues, None, dbias, None, None, None, None, None)
            None values correspond to non-tensor inputs that don't need gradients
        """
        x, values, col_indices, bias = ctx.saved_tensors
        in_features = ctx.in_features
        R = ctx.R
        K = ctx.K
        B = ctx.B
        use_triton = ctx.use_triton

        dx = None
        dvalues = None
        dbias = None

        # Determine which implementation to use
        use_triton_impl = use_triton and TRITON_AVAILABLE and dout.is_cuda

        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        needs_bias_grad = ctx.needs_input_grad[3] and bias is not None

        if use_triton_impl:
            # Use Triton kernels
            if needs_input_grad:
                dx = block_ell_backward_dx(dout, values, col_indices, in_features)

            if needs_weight_grad:
                dvalues = block_ell_backward_dw(x, dout, col_indices, R, K, B)

            if needs_bias_grad:
                dbias = block_ell_backward_db(dout)
        else:
            # Use reference implementations
            if needs_input_grad:
                dx = block_ell_backward_dx_reference(
                    dout, values, col_indices, in_features, tile_size=B
                )

            if needs_weight_grad:
                dvalues = block_ell_backward_dw_reference(x, dout, col_indices, R, K, tile_size=B)

            if needs_bias_grad:
                dbias = block_ell_backward_db(dout)

        # Return gradients in same order as forward inputs
        # (x, values, col_indices, bias, in_features, R, K, B, use_triton)
        return dx, dvalues, None, dbias, None, None, None, None, None


def block_ell_autograd(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    bias: Optional[Tensor],
    in_features: int,
    R: int,
    K: int,
    B: int,
    use_triton: bool = True,
) -> Tensor:
    """Apply Block-ELL linear with autograd support.

    This is the main entry point for Block-ELL sparse linear with gradient tracking.
    It wraps the forward and backward passes in an autograd Function.

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        values: Block weights [R, K, B, B]
        col_indices: Column indices [R, K]
        bias: Optional bias [out_features]
        in_features: Input dimension
        R: Output block-rows
        K: Blocks per row
        B: Block size
        use_triton: Whether to use Triton kernels when available (default True)

    Returns:
        Output tensor with gradient tracking
    """
    return BlockELLFunction.apply(x, values, col_indices, bias, in_features, R, K, B, use_triton)
