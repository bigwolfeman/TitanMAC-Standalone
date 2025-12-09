"""
TitanMAC wrapper for HuggingFace compatibility.

This wrapper adapts TitanMAC to the HuggingFace Transformers interface,
specifically returning CausalLMOutput for compatibility with Trainer.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_outputs import CausalLMOutput

from .titanmac import TitanMAC


class TitanMACForDistillation(nn.Module):
    """
    HuggingFace-compatible wrapper for TitanMAC.

    Adapts TitanMAC's dictionary output to CausalLMOutput for
    compatibility with HuggingFace Trainer and distillation pipeline.

    Args:
        model: TitanMAC model instance

    Shape:
        Input: input_ids [B, T]
        Output: CausalLMOutput with .logits and .loss

    Example:
        >>> config = TitanMACConfig(...)
        >>> base_model = TitanMAC(config)
        >>> model = TitanMACForDistillation(base_model)
        >>> input_ids = torch.randint(0, 1000, (2, 64))
        >>> output = model(input_ids, labels=input_ids)
        >>> output.logits.shape
        torch.Size([2, 64, 1000])
        >>> output.loss
        tensor(...)
    """

    def __init__(self, model: TitanMAC):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs  # Ignore other HuggingFace trainer kwargs
    ) -> CausalLMOutput:
        """
        Forward pass with HuggingFace-compatible output.

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T] (optional)
            labels: Target token IDs [B, T] (optional)
            **kwargs: Additional kwargs (ignored)

        Returns:
            CausalLMOutput with:
                - loss: scalar (if labels provided, includes memory_loss if enabled)
                - logits: [B, T, vocab_size]
                - hidden_states: None
                - attentions: None

        Note:
            If NeuralMemory is enabled, the returned loss includes both
            CE loss and memory_loss. Access result dict directly for
            individual loss components: result["ce_loss"], result["memory_loss"]
        """
        # Call underlying TitanMAC model
        result = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

        # Convert to CausalLMOutput
        # Note: result["loss"] already includes memory_loss if enabled
        return CausalLMOutput(
            loss=result.get("loss"),
            logits=result["logits"],
            hidden_states=None,
            attentions=None,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Delegate to underlying model."""
        return self.model.get_num_params(non_embedding=non_embedding)

    def reset_memory(self):
        """Delegate to underlying model."""
        self.model.reset_memory()

    def get_memory_stats(self):
        """Delegate to underlying model."""
        return self.model.get_memory_stats()

    @property
    def config(self):
        """Access underlying model config."""
        return self.model.config

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing in underlying model."""
        self.model.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing in underlying model."""
        self.model.disable_gradient_checkpointing()
