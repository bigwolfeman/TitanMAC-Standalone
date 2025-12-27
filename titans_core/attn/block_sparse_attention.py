"""
Block-sparse windowed attention with O(T*w) complexity.

PAPER-FAITHFUL IMPLEMENTATION

Current WindowedAttention uses SDPA with mask, which is still O(T²)
because it computes full Q @ K^T then masks.

This module implements TRUE O(T*w) attention by:
1. Processing in blocks of size `block_size`
2. Each block only attends to relevant blocks (within window)
3. Never materializes full T×T attention matrix

Complexity: O(T * w) where w = window_size

Version: 1.0.0
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSparseAttention(nn.Module):
    """
    Block-sparse windowed attention with O(T*w) complexity.

    Implements sliding window attention by processing in blocks:
    - Divide sequence into blocks of `block_size`
    - Each block attends to: persistent tokens + nearby blocks within window
    - Never materializes full T×T attention matrix

    PAPER-FAITHFUL: Includes 1D depthwise-separable convolutions after QKV projections
    (Section 4.4: "Following recent modern linear recurrent models...")

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Attention window size (each token attends to this many past tokens)
        block_size: Block size for sparse computation (should divide window_size)
        n_persistent: Number of persistent tokens (always attend to these)
        dropout: Dropout probability
        causal: Apply causal masking
        use_conv: Add 1D conv after QKV projections (paper-faithful, default True)
        conv_kernel: Kernel size for QKV convolutions (Mamba convention: 4)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,
        block_size: int = 64,
        n_persistent: int = 0,
        dropout: float = 0.0,
        causal: bool = True,
        use_conv: bool = True,
        conv_kernel: int = 4,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.block_size = block_size
        self.n_persistent = n_persistent
        self.dropout_p = dropout
        self.causal = causal
        self.use_conv = use_conv

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # 1D depthwise-separable convolutions after QKV (paper Section 4.4)
        # "Following recent modern linear recurrent models, we incorporate
        #  a 1D depthwise-separable convolution layer after each of the
        #  query, key, and value projections."
        if use_conv:
            # Depthwise conv: each channel processed separately
            # Causal padding: pad left only
            self.conv_q = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel - 1,
                groups=d_model,
                bias=False,
            )
            self.conv_k = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel - 1,
                groups=d_model,
                bias=False,
            )
            self.conv_v = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel - 1,
                groups=d_model,
                bias=False,
            )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scale factor
        self.scale = 1.0 / math.sqrt(self.d_head)

        # PERF: Pre-allocate causal mask buffer (max size)
        # Will be sliced as needed - avoids per-block allocation
        max_context = window_size + block_size + n_persistent
        self.register_buffer(
            "_causal_mask_cache",
            torch.ones(block_size, max_context, dtype=torch.bool),
            persistent=False,
        )

    def _get_causal_mask(
        self,
        block_len: int,
        context_len: int,
        block_start: int,
        context_start: int,
        n_persistent: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate causal mask for a block using vectorized operations.

        PERF: Uses torch.arange + broadcasting instead of nested Python loops.

        Args:
            block_len: Number of query positions
            context_len: Number of key positions (excluding persistent)
            block_start: Global position of first query
            context_start: Global position of first key
            n_persistent: Number of persistent tokens (always attend)
            device: Target device

        Returns:
            Causal mask [block_len, n_persistent + context_len]
        """
        total_context = n_persistent + context_len

        # Start with ones (can attend everywhere)
        mask = self._causal_mask_cache[:block_len, :total_context].clone()
        mask.fill_(True)

        if context_len > 0:
            # Query positions: block_start, block_start+1, ..., block_start+block_len-1
            query_pos = torch.arange(block_start, block_start + block_len, device=device)

            # Key positions: context_start, context_start+1, ..., context_start+context_len-1
            key_pos = torch.arange(context_start, context_start + context_len, device=device)

            # Causal: query_pos[i] >= key_pos[j] to attend
            # Shape: [block_len, 1] >= [1, context_len] -> [block_len, context_len]
            causal_part = query_pos.unsqueeze(1) >= key_pos.unsqueeze(0)

            # Place causal constraint on sequence part (after persistent tokens)
            mask[:, n_persistent : n_persistent + context_len] = causal_part

        return mask

    def _compute_block_attention(
        self,
        q_block: torch.Tensor,
        k_context: torch.Tensor,
        v_context: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention for a single query block.

        Args:
            q_block: Query block [B, H, block_size, d_head]
            k_context: Key context [B, H, context_size, d_head]
            v_context: Value context [B, H, context_size, d_head]
            causal_mask: Optional causal mask [block_size, context_size]

        Returns:
            Output [B, H, block_size, d_head]
        """
        # Compute attention scores
        scores = torch.matmul(q_block, k_context.transpose(-2, -1)) * self.scale
        # [B, H, block_size, context_size]

        # Apply causal mask if provided
        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Dropout
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Compute output
        output = torch.matmul(attn_weights, v_context)
        # [B, H, block_size, d_head]

        return output

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with block-sparse attention.

        Args:
            x: Input tensor [B, T, d_model]
            attn_mask: Optional custom mask (ignored, we use block-sparse pattern)

        Returns:
            Output tensor [B, T, d_model]
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply 1D depthwise convolutions (paper Section 4.4)
        if self.use_conv:
            # Conv1d expects [B, C, T], so transpose
            q = q.transpose(1, 2)  # [B, d_model, T]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply causal convs (trim future positions)
            q = self.conv_q(q)[:, :, :T]  # Trim to original length
            k = self.conv_k(k)[:, :, :T]
            v = self.conv_v(v)[:, :, :T]

            # Transpose back
            q = q.transpose(1, 2)  # [B, T, d_model]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Reshape for multi-head attention
        # [B, T, d_model] -> [B, H, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Handle persistent tokens separately (they have global attention)
        if self.n_persistent > 0:
            # Persistent tokens
            q_persistent = q[:, :, : self.n_persistent, :]
            k_persistent = k[:, :, : self.n_persistent, :]
            v_persistent = v[:, :, : self.n_persistent, :]

            # Sequence tokens
            q_seq = q[:, :, self.n_persistent :, :]
            k_seq = k[:, :, self.n_persistent :, :]
            v_seq = v[:, :, self.n_persistent :, :]

            T_seq = T - self.n_persistent
        else:
            q_persistent = k_persistent = v_persistent = None
            q_seq = q
            k_seq = k
            v_seq = v
            T_seq = T

        # PERF: Pre-allocate output tensor instead of list.append + torch.cat
        output = torch.empty(B, self.n_heads, T, self.d_head, device=x.device, dtype=x.dtype)

        # Process sequence in blocks
        num_blocks = (T_seq + self.block_size - 1) // self.block_size
        output_offset = 0

        # First, handle persistent tokens if they exist
        if self.n_persistent > 0:
            # Persistent tokens attend to all tokens (global attention)
            persistent_output = self._compute_block_attention(
                q_persistent,
                k,  # All keys
                v,  # All values
                causal_mask=None,  # Persistent tokens are bidirectional within themselves
            )
            output[:, :, : self.n_persistent, :] = persistent_output
            output_offset = self.n_persistent

        # Process sequence blocks
        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, T_seq)
            block_len = block_end - block_start

            # Query block
            q_block = q_seq[:, :, block_start:block_end, :]  # [B, H, block_len, d_head]

            # Determine context window for this block
            # Include tokens from max(0, block_start - window_size) to block_end
            context_start = max(0, block_start - self.window_size)
            context_end = block_end  # Only past tokens (causal)
            seq_context_len = context_end - context_start

            # Build context: persistent tokens + relevant sequence tokens
            if self.n_persistent > 0:
                k_context = torch.cat(
                    [k_persistent, k_seq[:, :, context_start:context_end, :]], dim=2
                )
                v_context = torch.cat(
                    [v_persistent, v_seq[:, :, context_start:context_end, :]], dim=2
                )

                # PERF: Use vectorized mask generation instead of nested Python loops
                if self.causal:
                    causal_mask = self._get_causal_mask(
                        block_len,
                        seq_context_len,
                        block_start,
                        context_start,
                        self.n_persistent,
                        x.device,
                    )
                else:
                    causal_mask = None
            else:
                k_context = k_seq[:, :, context_start:context_end, :]
                v_context = v_seq[:, :, context_start:context_end, :]

                # PERF: Use vectorized mask generation
                if self.causal:
                    causal_mask = self._get_causal_mask(
                        block_len, seq_context_len, block_start, context_start, 0, x.device
                    )
                else:
                    causal_mask = None

            # Compute block attention and write directly to output
            block_output = self._compute_block_attention(q_block, k_context, v_context, causal_mask)
            output[:, :, output_offset : output_offset + block_len, :] = block_output
            output_offset += block_len

        # Reshape back
        output = output.transpose(1, 2)  # [B, T, H, d_head]
        output = output.reshape(B, T, self.d_model)

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


def create_block_sparse_attention(config) -> BlockSparseAttention:
    """
    Create BlockSparseAttention from TitanMACConfig.

    Args:
        config: TitanMACConfig instance

    Returns:
        BlockSparseAttention instance
    """
    # Use block_size = window_size / 8 as default
    block_size = getattr(config, "block_size", max(64, config.window_size // 8))

    return BlockSparseAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        window_size=config.window_size,
        block_size=block_size,
        n_persistent=config.n_persistent,
        dropout=config.attention_dropout,
        causal=config.causal,
    )
