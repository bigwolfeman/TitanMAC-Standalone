"""
TitanMAC: Titan Memory-Augmented Context Model.

Architecture:
    - Token + position embeddings
    - Stack of TitanBlock layers (windowed attention + MLP)
    - Output RMSNorm
    - LM head (optionally tied to embeddings)
    - Optional MemoryBank for MAC retrieval

Contract:
    - forward(input_ids, labels=None, attention_mask=None) -> Dict[str, Tensor]
    - Returns {"logits": [B, T, vocab_size], "loss": Optional[scalar]}
    - get_num_params(non_embedding: bool) -> int

Label Convention (standard causal LM):
    Labels should be the same as input_ids (labels = input_ids.clone()).
    The model handles shifting internally:
        - logits[:, :-1] predicts labels[:, 1:]
    This is consistent with HuggingFace transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.checkpoint import checkpoint

from ..config import TitanMACConfig
from ..blocks.titan_block import TitanBlock
from ..blocks.norms import RMSNorm
from ..memory.memory_bank import MemoryBank
from ..memory.neural_memory import NeuralMemory


class TitanMAC(nn.Module):
    """
    TitanMAC: Titan Memory-Augmented Context Model.

    Architecture:
        1. Token embeddings + learned position embeddings
        2. Stack of TitanBlock layers (windowed attention + MLP)
        3. Final RMSNorm
        4. LM head (optionally tied to token embeddings)
        5. Optional MemoryBank for MAC retrieval

    Args:
        config: TitanMACConfig with architecture parameters

    Shape:
        Input: input_ids [B, T] (long)
        Output: {"logits": [B, T, vocab_size], "loss": Optional[scalar]}

    Example:
        >>> config = TitanMACConfig(
        ...     vocab_size=128000,
        ...     d_model=640,
        ...     n_heads=10,
        ...     n_layers=16
        ... )
        >>> model = TitanMAC(config)
        >>> input_ids = torch.randint(0, 128000, (2, 64))
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([2, 64, 128000])
    """

    def __init__(self, config: TitanMACConfig):
        super().__init__()

        self.config = config
        config.validate()

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Position embeddings (learned)
        self.embed_positions = nn.Embedding(config.max_seq_len, config.d_model)

        # Persistent tokens (global context)
        # These are prepended to the sequence and attend bidirectionally
        if config.n_persistent > 0:
            self.persistent_tokens = nn.Parameter(
                torch.zeros(config.n_persistent, config.d_model)
            )
        else:
            # Register empty parameter for n_persistent=0 case
            self.register_parameter("persistent_tokens", None)

        # Stack of Titan blocks
        self.layers = nn.ModuleList([
            TitanBlock(config) for _ in range(config.n_layers)
        ])

        # Output normalization
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        # Optional MemoryBank for MAC retrieval (legacy)
        if config.use_memory_bank:
            self.memory_bank = MemoryBank(
                d_model=config.d_model,
                capacity=1024  # Default capacity
            )
        else:
            self.memory_bank = None

        # Neural Long-Term Memory (Titans paper, Section 3.1)
        if config.use_neural_memory:
            self.neural_memory = NeuralMemory(
                d_model=config.d_model,
                d_memory=config.d_memory if config.d_memory is not None else config.d_model,
                capacity=config.memory_capacity,
                theta_lr=config.memory_theta_lr,
                forget_hidden=config.memory_forget_hidden,
                decay_hidden=config.memory_decay_hidden,
            )
            # Projection layers for MAC dataflow (T039, T040)
            d_memory = config.d_memory if config.d_memory is not None else config.d_model
            self.memory_proj = nn.Linear(d_memory, config.d_model)
            self.gate_proj = nn.Linear(d_memory, config.d_model)
        else:
            self.neural_memory = None
            self.memory_proj = None
            self.gate_proj = None

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practices."""
        # Embeddings: normal distribution
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_positions.weight, mean=0.0, std=0.02)

        # Persistent tokens: small random initialization
        if self.persistent_tokens is not None:
            nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.02)

        # LM head: normal distribution (if not tied)
        if not self.config.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Memory projection layers
        if self.memory_proj is not None:
            nn.init.normal_(self.memory_proj.weight, mean=0.0, std=0.02)
        if self.gate_proj is not None:
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)

    def forward_mac(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MAC (Memory as Context) dataflow.

        Implements Titans Equation 16:
            h_t = M*_{t-1}(q_t)                    # Memory retrieval (stop-grad)
            S_tilde = [p_1...p_np] || h_t || S_t   # Concatenate
            y_t = Attn(S_tilde)                    # Attention
            M_t = M_{t-1}(y_t)                     # Update memory
            o_t = y_t * sigmoid(M*_t(y_t))         # Gated output

        Args:
            input_ids: Input token IDs [B, T]
            labels: Target token IDs for loss computation [B, T]
            attention_mask: Attention mask [B, T] (optional)

        Returns:
            Dictionary with:
                - "logits": [B, T, vocab_size]
                - "loss": scalar (if labels provided)
                - "memory_loss": scalar memory update loss
        """
        if self.neural_memory is None:
            raise ValueError("forward_with_memory requires use_neural_memory=True")

        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)  # [B, T, d_model]

        # Prepend persistent tokens and add position embeddings
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            # Position embeddings only for input sequence (not persistent tokens)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb

            # Prepend persistent tokens
            persistent = self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            # No persistent tokens - standard behavior
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb
            persistent = None

        # 1. Memory retrieval (stop-gradient) - T041
        h = self.neural_memory.retrieve(x)  # [B, T, d_memory]
        h_proj = self.memory_proj(h)  # [B, T, d_model]

        # 2. MAC dataflow concatenation: [persistent | memory | sequence] - T041
        if persistent is not None:
            x_combined = torch.cat([persistent, h_proj, x], dim=1)
            # [B, n_persistent + T + T, d_model]
        else:
            x_combined = torch.cat([h_proj, x], dim=1)
            # [B, 2*T, d_model]

        # 3. Pass through Titan blocks (attention over combined sequence)
        y = x_combined
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(hidden_states, attn_mask=None)
                    return custom_forward
                y = checkpoint(create_custom_forward(layer), y, use_reentrant=False)
            else:
                y = layer(y, attn_mask=None)

        # 4. Update memory with sequence part (skip persistent and memory retrieval)
        # Extract the original sequence region from y
        if persistent is not None:
            seq_start = self.config.n_persistent + T
            y_seq = y[:, seq_start:, :]  # [B, T, d_model]
        else:
            y_seq = y[:, T:, :]  # [B, T, d_model]

        # Memory update - T043, T044
        # Only update if training or test-time memory enabled
        if self.training or self.config.enable_test_time_memory:
            memory_loss = self.neural_memory.update(y_seq)
        else:
            # In eval mode without test-time memory, just compute loss without update
            memory_loss = self.neural_memory.compute_loss(y_seq)

        # 5. Gated output: o = y * sigmoid(M*(y)) - T041
        gate_values = self.neural_memory.retrieve(y)  # [B, seq_len, d_memory]
        gate_proj = self.gate_proj(gate_values)  # [B, seq_len, d_model]
        gate = torch.sigmoid(gate_proj)  # Bounded to [0, 1]
        o = y * gate

        # Final normalization
        o = self.norm(o)

        # LM head - only on the original sequence positions
        if persistent is not None:
            # Extract logits for original sequence (skip persistent + memory retrieval)
            seq_start = self.config.n_persistent + T
            o_seq = o[:, seq_start:, :]  # [B, T, d_model]
        else:
            o_seq = o[:, T:, :]  # [B, T, d_model]

        logits = self.lm_head(o_seq)  # [B, T, vocab_size]

        # Prepare output
        output = {
            "logits": logits,
            "memory_loss": memory_loss,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Standard causal LM shift: predict next token
            # logits[i] predicts labels[i+1], so we align:
            #   shift_logits = logits[:, :-1, :]  (positions 0..T-2)
            #   shift_labels = labels[:, 1:]      (positions 1..T-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy (ignore_index=-100 for padding)
            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            output["ce_loss"] = ce_loss
            # Total loss = CE loss + memory loss (for gradient flow through memory)
            output["loss"] = ce_loss + memory_loss

        return output

    def forward_mag(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MAG (Memory as Gate) dataflow.

        Memory gates the attention output instead of being concatenated:
            y_t = Attn(x_t)                        # Standard attention
            g_t = sigmoid(M*_t(y_t))               # Memory-based gating
            o_t = y_t * g_t                        # Gated output
            M_t = M_{t-1}(y_t)                     # Update memory

        Args:
            input_ids: Input token IDs [B, T]
            labels: Target token IDs for loss computation [B, T]
            attention_mask: Attention mask [B, T] (optional)

        Returns:
            Dictionary with:
                - "logits": [B, T, vocab_size]
                - "loss": scalar (if labels provided)
                - "memory_loss": scalar memory update loss
        """
        if self.neural_memory is None:
            raise ValueError("forward_mag requires use_neural_memory=True")

        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)  # [B, T, d_model]

        # Prepend persistent tokens and add position embeddings
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb

            persistent = self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([persistent, x], dim=1)  # [B, n_persistent + T, d_model]
        else:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb

        # Pass through Titan blocks (standard attention)
        y = x
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(hidden_states, attn_mask=None)
                    return custom_forward
                y = checkpoint(create_custom_forward(layer), y, use_reentrant=False)
            else:
                y = layer(y, attn_mask=None)

        # Extract sequence region for memory update
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            y_seq = y[:, self.config.n_persistent:, :]  # [B, T, d_model]
        else:
            y_seq = y  # [B, T, d_model]

        # MAG: Memory gates the output
        gate_values = self.neural_memory.retrieve(y)  # [B, seq_len, d_memory]
        gate_proj = self.gate_proj(gate_values)  # [B, seq_len, d_model]
        gate = torch.sigmoid(gate_proj)  # Bounded to [0, 1]
        o = y * gate  # Gated output

        # Update memory with sequence part
        if self.training or self.config.enable_test_time_memory:
            memory_loss = self.neural_memory.update(y_seq)
        else:
            memory_loss = self.neural_memory.compute_loss(y_seq)

        # Final normalization
        o = self.norm(o)

        # LM head - only on the original sequence positions
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            o_seq = o[:, self.config.n_persistent:, :]  # [B, T, d_model]
        else:
            o_seq = o  # [B, T, d_model]

        logits = self.lm_head(o_seq)  # [B, T, vocab_size]

        # Prepare output
        output = {
            "logits": logits,
            "memory_loss": memory_loss,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Standard causal LM shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            output["ce_loss"] = ce_loss
            # Total loss = CE loss + memory loss (for gradient flow through memory)
            output["loss"] = ce_loss + memory_loss

        return output

    def forward_mal(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MAL (Memory as Layer) dataflow.

        Memory replaces attention in alternating layers:
            For even layers: y_t = Attn(x_t)       # Standard attention
            For odd layers: y_t = Memory(x_t)      # Memory retrieval
            M_t = M_{t-1}(y_t)                     # Update memory

        Args:
            input_ids: Input token IDs [B, T]
            labels: Target token IDs for loss computation [B, T]
            attention_mask: Attention mask [B, T] (optional)

        Returns:
            Dictionary with:
                - "logits": [B, T, vocab_size]
                - "loss": scalar (if labels provided)
                - "memory_loss": scalar memory update loss
        """
        if self.neural_memory is None:
            raise ValueError("forward_mal requires use_neural_memory=True")

        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)  # [B, T, d_model]

        # Prepend persistent tokens and add position embeddings
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb

            persistent = self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([persistent, x], dim=1)  # [B, n_persistent + T, d_model]
        else:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.embed_positions(positions)
            x = x + pos_emb

        # Pass through layers, alternating between attention and memory
        y = x
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx % 2 == 0:
                # Even layers: use standard attention
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(hidden_states):
                            return module(hidden_states, attn_mask=None)
                        return custom_forward
                    y = checkpoint(create_custom_forward(layer), y, use_reentrant=False)
                else:
                    y = layer(y, attn_mask=None)
            else:
                # Odd layers: use memory retrieval
                h = self.neural_memory.retrieve(y)  # [B, seq_len, d_memory]
                h_proj = self.memory_proj(h)  # [B, seq_len, d_model]
                # Residual connection with memory
                y = y + h_proj

        # Extract sequence region for memory update
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            y_seq = y[:, self.config.n_persistent:, :]  # [B, T, d_model]
        else:
            y_seq = y  # [B, T, d_model]

        # Update memory with sequence part
        if self.training or self.config.enable_test_time_memory:
            memory_loss = self.neural_memory.update(y_seq)
        else:
            memory_loss = self.neural_memory.compute_loss(y_seq)

        # Final normalization
        y = self.norm(y)

        # LM head - only on the original sequence positions
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            y_seq_out = y[:, self.config.n_persistent:, :]  # [B, T, d_model]
        else:
            y_seq_out = y  # [B, T, d_model]

        logits = self.lm_head(y_seq_out)  # [B, T, vocab_size]

        # Prepare output
        output = {
            "logits": logits,
            "memory_loss": memory_loss,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Standard causal LM shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            output["ce_loss"] = ce_loss
            # Total loss = CE loss + memory loss (for gradient flow through memory)
            output["loss"] = ce_loss + memory_loss

        return output

    # Alias for backward compatibility
    def forward_with_memory(self, *args, **kwargs):
        """Alias for forward_mac for backward compatibility."""
        return self.forward_mac(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TitanMAC.

        Dispatches to variant-specific forward method based on config.titans_variant:
            - "MAC": Memory as Context (concatenated dataflow)
            - "MAG": Memory as Gate (multiplicative gating)
            - "MAL": Memory as Layer (replaces attention in some layers)

        If neural_memory is disabled, uses standard windowed attention.

        Args:
            input_ids: Input token IDs [B, T]
            labels: Target token IDs for loss computation [B, T]
            attention_mask: Attention mask [B, T] (optional)

        Returns:
            Dictionary with:
                - "logits": [B, T, vocab_size]
                - "loss": scalar (if labels provided)
                - "memory_loss": scalar (if neural_memory enabled)

        Process (without memory):
            1. Embed tokens (no positions yet)
            2. Prepend persistent tokens (if enabled)
            3. Add position embeddings (only to input sequence, not persistent tokens)
            4. Pass through Titan blocks
            5. Final norm
            6. LM head projection
            7. Optional loss computation
        """
        # Dispatch to variant-specific forward if neural memory is enabled (T083)
        if self.neural_memory is not None and self.config.use_neural_memory:
            variant = self.config.titans_variant
            if variant == "MAC":
                return self.forward_mac(input_ids, labels, attention_mask)
            elif variant == "MAG":
                return self.forward_mag(input_ids, labels, attention_mask)
            elif variant == "MAL":
                return self.forward_mal(input_ids, labels, attention_mask)
            else:
                raise ValueError(f"Unknown titans_variant: {variant}")
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)  # [B, T, d_model]

        # Prepend persistent tokens if enabled
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            # Expand persistent tokens for batch
            persistent = self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, n_persistent, d_model]

            # Position embeddings only for input sequence (not persistent tokens)
            # Persistent tokens don't get position embeddings - they're position-invariant
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
            pos_emb = self.embed_positions(positions)  # [B, T, d_model]
            x = x + pos_emb  # Add position embeddings to input sequence

            # Concatenate: [persistent | input_sequence]
            x = torch.cat([persistent, x], dim=1)  # [B, n_persistent + T, d_model]
        else:
            # No persistent tokens - standard behavior
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
            pos_emb = self.embed_positions(positions)  # [B, T, d_model]
            x = x + pos_emb

        # Pass through Titan blocks
        # Note: attention_mask not passed to layers - windowed attention
        # handles causal masking internally
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory savings
                # Create a closure that checkpointing can use
                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(hidden_states, attn_mask=None)
                    return custom_forward

                x = checkpoint(create_custom_forward(layer), x, use_reentrant=False)
            else:
                x = layer(x, attn_mask=None)

            # Optional: Apply memory bank retrieval
            if self.memory_bank is not None and self._should_update_memory():
                # Memory bank read/write (MAC pattern)
                # For now, we skip memory updates in eval mode unless enabled
                if self.training or self.config.enable_test_time_memory:
                    # Simple memory integration: add retrieved tokens
                    # (In full implementation, this would be integrated into attention)
                    pass

        # Final normalization
        x = self.norm(x)  # [B, n_persistent + T, d_model] or [B, T, d_model]

        # LM head
        logits = self.lm_head(x)  # [B, n_persistent + T, vocab_size] or [B, T, vocab_size]

        # Slice out persistent token logits for output - always return [B, T, vocab_size]
        # This ensures the output shape matches the input shape for distillation
        if self.persistent_tokens is not None and self.config.n_persistent > 0:
            output_logits = logits[:, self.config.n_persistent:, :]  # [B, T, vocab_size]
        else:
            output_logits = logits

        # Prepare output
        output = {"logits": output_logits}

        # Compute loss if labels provided
        if labels is not None:
            # Standard causal LM shift: predict next token
            # logits[i] predicts labels[i+1], so we align:
            #   shift_logits = logits[:, :-1, :]  (positions 0..T-2)
            #   shift_labels = labels[:, 1:]      (positions 1..T-1)
            shift_logits = output_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy (ignore_index=-100 for padding)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def _should_update_memory(self) -> bool:
        """
        Determine if memory updates should be performed.

        Returns:
            True if memory updates are enabled for current mode.
        """
        if self.training:
            return True  # Always update during training

        # In eval mode, only update if test-time memory is enabled
        return self.config.enable_test_time_memory

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters

        Example:
            >>> model = TitanMAC(config)
            >>> total_params = model.get_num_params(non_embedding=False)
            >>> non_embed_params = model.get_num_params(non_embedding=True)
        """
        total_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract token and position embedding parameters
            embed_params = self.embed_tokens.weight.numel()  # vocab_size * d_model
            embed_params += self.embed_positions.weight.numel()  # max_seq_len * d_model

            # If weights are NOT tied, also subtract lm_head
            if not self.config.tie_weights:
                embed_params += self.lm_head.weight.numel()  # vocab_size * d_model
            # else: lm_head shares weights with embed_tokens, don't double-subtract

            return total_params - embed_params

        return total_params

    def reset_memory(self):
        """Reset memory bank to zero state (if enabled)."""
        if self.memory_bank is not None:
            self.memory_bank.reset_memory()

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """
        Get memory bank statistics (if enabled).

        Returns:
            Dictionary with memory stats, or None if memory disabled
        """
        if self.memory_bank is not None:
            return self.memory_bank.get_memory_stats()
        return None

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory efficiency.

        Trades ~30% compute for ~50% memory savings during training.
        """
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
