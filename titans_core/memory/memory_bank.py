"""
Memory bank for MAC (Memory-Augmented Context) retrieval pattern.

This module provides a key-value memory bank with topk retrieval and
surprise-based writing.
"""

import torch
import torch.nn as nn
from typing import Dict


class MemoryBank(nn.Module):
    """
    Memory bank for MAC (Memory-Augmented Context) retrieval pattern.

    This is separate from NeuralMemory's fast-weight system. MemoryBank
    stores explicit key-value pairs for topk retrieval in the MAC dataflow.

    Architecture (from Titans paper):
        - Separate K_bank and V_bank buffers
        - Topk retrieval: read(query, topk) → [B, topk, D]
        - Surprise-based writes with LRU eviction
        - EMA decay for existing memories

    Args:
        d_model: Model dimension
        capacity: Maximum number of memory slots (default: 1024)

    Shape:
        read: query [B, D] → output [B, topk, D]
        write: keys [B, T, D], values [B, T, D], gate [B, T]

    Example:
        >>> memory = MemoryBank(d_model=128, capacity=1024)
        >>> query = torch.randn(2, 128)
        >>> h_t = memory.read(query, topk=32)  # [2, 32, 128]
        >>> keys = torch.randn(2, 10, 128)
        >>> values = torch.randn(2, 10, 128)
        >>> surprise = torch.rand(2, 10)
        >>> memory.write(keys, values, gate=surprise, decay=0.99)
    """

    def __init__(self, d_model: int, capacity: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity

        # Separate key and value banks
        self.register_buffer("K_bank", torch.zeros(capacity, d_model))
        self.register_buffer("V_bank", torch.zeros(capacity, d_model))

        # Usage tracking for LRU eviction
        self.register_buffer("usage", torch.zeros(capacity))

        # Number of filled slots
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    def read(self, query: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Retrieve top-k memory tokens by dot-product similarity.

        Args:
            query: Query tensor [B, D]
            topk: Number of tokens to retrieve

        Returns:
            h_t: Retrieved memory tokens [B, topk, D]
                 Zero-padded if fewer than topk tokens available
        """
        B, D = query.shape

        # Number of filled slots
        n_filled = min(self.filled.item(), self.capacity)

        if n_filled == 0:
            # No memory yet - return zeros
            return torch.zeros(B, topk, D, device=query.device, dtype=query.dtype)

        # Compute similarity: [B, D] @ [n_filled, D].T = [B, n_filled]
        K_active = self.K_bank[:n_filled, :]  # [n_filled, D]
        scores = query @ K_active.T  # [B, n_filled]

        # Top-k retrieval
        k = min(topk, n_filled)
        top_indices = torch.topk(scores, k=k, dim=1).indices  # [B, k]

        # Gather values
        # Expand indices for gathering: [B, k] → [B, k, D]
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)
        V_active = self.V_bank[:n_filled, :].unsqueeze(0).expand(B, -1, -1)  # [B, n_filled, D]
        h_t = torch.gather(V_active, dim=1, index=top_indices_expanded)  # [B, k, D]

        # Zero-pad if k < topk
        if k < topk:
            padding = torch.zeros(B, topk - k, D, device=query.device, dtype=query.dtype)
            h_t = torch.cat([h_t, padding], dim=1)

        # Update usage (LRU tracking)
        with torch.no_grad():
            # Accumulate scores for accessed slots
            usage_update = scores.sum(dim=0)  # [n_filled]
            self.usage[:n_filled] += usage_update

        return h_t

    def write(
        self,
        keys: torch.Tensor,  # [B, T, D]
        values: torch.Tensor,  # [B, T, D]
        gate: torch.Tensor,  # [B, T] - surprise signal
        decay: float = 0.99,
        threshold: float = 0.5,
    ):
        """
        Write key-value pairs with surprise gating and capacity enforcement.

        Args:
            keys: Key tensor [B, T, D]
            values: Value tensor [B, T, D]
            gate: Write strength (surprise) [B, T]
            decay: EMA decay for existing memories (default: 0.99)
            threshold: Only write if gate > threshold (default: 0.5)
        """
        B, T, D = keys.shape

        with torch.no_grad():
            # Flatten batch and time
            keys_flat = keys.reshape(-1, D)  # [B*T, D]
            values_flat = values.reshape(-1, D)  # [B*T, D]
            gate_flat = gate.reshape(-1)  # [B*T]

            # Filter by threshold
            mask = gate_flat > threshold
            keys_write = keys_flat[mask]  # [N_write, D]
            values_write = values_flat[mask]

            N_write = keys_write.shape[0]
            if N_write == 0:
                return  # Nothing to write

            # Decay existing memories (EMA)
            n_filled = min(self.filled.item(), self.capacity)
            if n_filled > 0:
                self.V_bank[:n_filled] *= decay

            # Add new memories with LRU eviction
            for i in range(N_write):
                if self.filled < self.capacity:
                    # Fill next empty slot
                    idx = self.filled.item()
                    self.filled += 1
                else:
                    # Evict LRU slot
                    idx = torch.argmin(self.usage[: self.capacity]).item()

                # Write to slot
                self.K_bank[idx] = keys_write[i]
                self.V_bank[idx] = values_write[i]
                self.usage[idx] = 1.0  # Reset usage for new entry

    def reset_memory(self):
        """Reset memory bank to zero state."""
        self.K_bank.zero_()
        self.V_bank.zero_()
        self.usage.zero_()
        self.filled.zero_()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory state."""
        n_filled = min(self.filled.item(), self.capacity)

        stats = {
            "capacity": float(self.capacity),
            "filled": float(n_filled),
            "utilization": n_filled / self.capacity if self.capacity > 0 else 0.0,
        }

        if n_filled > 0:
            stats.update(
                {
                    "K_norm_mean": torch.norm(self.K_bank[:n_filled], p=2, dim=1).mean().item(),
                    "V_norm_mean": torch.norm(self.V_bank[:n_filled], p=2, dim=1).mean().item(),
                    "usage_mean": self.usage[:n_filled].mean().item(),
                    "usage_max": self.usage[:n_filled].max().item(),
                }
            )

        return stats
