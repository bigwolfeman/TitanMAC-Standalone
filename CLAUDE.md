# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TitanMAC-Standalone is a production-grade implementation of memory-augmented transformers combining:
- **Neural Long-Term Memory** with gradient-based surprise updates (Titans paper - arXiv:2501.00663)
- **Nested Learning** optimization with learned momentum (NeurIPS 2025)
- **Efficient Windowed Attention** with persistent tokens (O(T*w) memory vs O(T²))

## Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev tools
pip install -e ".[dev]"

# Install with training tools (wandb, bitsandbytes)
pip install -e ".[train]"

# Run tests
pytest tests/

# Run single test
pytest tests/unit/test_deep_nested_optimizer.py -v

# Format code
black titans_core/ --line-length 100

# Lint
ruff check titans_core/

# Train on math problems
python examples/train_math.py --steps 1000 --batch-size 4 --device cuda

# Train with neural memory
python examples/train_math.py --use-neural-memory --memory-capacity 256
```

## Architecture

```
titans_core/
├── config.py                    # TitanMACConfig - single dataclass for all params
├── models/
│   ├── titanmac.py             # Core TitanMAC model
│   └── titanmac_wrapper.py     # HuggingFace integration
├── blocks/
│   ├── titan_block.py          # Pre-norm attention+MLP block
│   ├── mlp.py                  # Feed-forward networks
│   └── norms.py                # RMSNorm
├── attn/
│   ├── windowed_attention.py   # O(T²) local attention (legacy)
│   └── block_sparse_attention.py # O(T*w) paper-faithful (recommended)
├── memory/
│   ├── neural_memory.py        # Gradient-based memory with forget/decay gates
│   └── memory_bank.py          # Key-value retrieval for MAC
└── opt/                         # Nested Learning optimizers
    ├── deep_nested_optimizer.py # Main optimizer (recommended)
    ├── dmgd.py                  # Deep Momentum Gradient Descent
    ├── cms.py                   # Continuum Memory System
    ├── nested_controller.py     # LR multiplier network
    └── param_groups.py          # Parameter grouping utilities
```

## Key Concepts

### Model Forward Contract
```python
model(input_ids, labels=None, attention_mask=None)
→ {"logits": [B, T, vocab_size], "loss": Optional[scalar]}
```

### Neural Memory Update (Paper-Faithful)
Memory M is a **deep MLP**, not an embedding table:
```
Associative Loss: l(M; x) = ||M(k) - v||²
Momentum: S_t = η_t * S_{t-1} - θ_t * ∇l(M; x)
Memory Update: M_t = (1 - α_t) * M_{t-1} + S_t
```

### DeepNestedOptimizer Training Loop
```python
optimizer.zero_grad()
loss = model(x, y)['loss']
loss.backward()
optimizer.step(loss.item())  # Pass loss value for meta-learning
```

### Winning Configurations

**TitanMAC** (from grid search):
- `momentum_num_layers=4`, `controller_num_layers=5`

**MoE models**:
- `momentum_num_layers=3`, `controller_num_layers=2`

## Critical Patterns

### torch.compile + Neural Memory
Neural memory uses `torch.autograd.grad(..., retain_graph=True)` which conflicts with compile's donated buffers:
```python
# neural_memory.py: Decorate update() with @torch._dynamo.disable
# Training script: Set torch._functorch.config.donated_buffer = False
# Do NOT use fullgraph=True
```

### CUDA Graph with Gradient Clipping
Gradient clipping must be **outside** the CUDA graph (gradient tensor addresses change between backward passes).

### In-Place Operation Safety
Meta-learning backward can conflict with optimizer.step(). Key functions must:
- Wrap stats computation in `torch.no_grad()`
- Return `stats.detach()` from `_compute_group_stats()`
- Detach `grad`, `prev_momentum`, `ema_target` in `_compute_mlp_proxy_loss()`

## Documentation

Key docs in `/docs`:
- `OPTIMIZER_QUICK_REFERENCE.md` - DeepNestedOptimizer API and troubleshooting
- `PERFORMANCE_OPTIMIZATIONS.md` - Fused operations, CUDA graphs, torch.compile
- `GPU_PROFILING_REPORT.md` - Profiling analysis and bottlenecks
- `PAPER_FAITHFUL_REFACTOR.md` - Memory architecture alignment with papers

## Variants

- **MAC**: Memory-Augmented Context (memory after attention)
- **MAG**: Memory-Augmented Gate (gated memory integration)
- **MAL**: Memory-Augmented Layer (memory as separate layer)

Set via `TitanMACConfig(titans_variant="MAC")`

## Active Technologies
- Python 3.10+ + PyTorch ≥2.1.0, Triton (existing in project), transformers ≥4.35.0 (001-cms-block-sparse)
- In-memory tensors (Block-ELL format), checkpointing via PyTorch state_dict (001-cms-block-sparse)

## Recent Changes
- 001-cms-block-sparse: Added Python 3.10+ + PyTorch ≥2.1.0, Triton (existing in project), transformers ≥4.35.0
