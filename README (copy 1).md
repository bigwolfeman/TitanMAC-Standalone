# TitanMAC: Memory-Augmented Transformer

Standalone implementation of the TitanMAC architecture based on:
- **Titans paper** (arxiv 2501.00663): Neural Long-Term Memory
- **Nested Learning** (NeurIPS 2025): DMGD and CMS

## Features

- **Windowed Attention**: O(T*w) memory instead of O(T²)
- **Persistent Tokens**: Global context via bidirectional attention
- **Neural Long-Term Memory**: Gradient-based memory with surprise updates
- **MAC/MAG/MAL Variants**: Three memory integration strategies
- **Deep Momentum Gradient Descent** (DMGD): Learned momentum
- **Continuum Memory System** (CMS): Multi-frequency updates

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from titans_core import TitanMACConfig, TitanMAC

# Create config
config = TitanMACConfig(
    vocab_size=50000,
    d_model=512,
    n_heads=8,
    n_layers=12,
    window_size=256,
    n_persistent=16,
)

# Create model
model = TitanMAC(config)

# Forward pass
input_ids = torch.randint(0, 50000, (2, 512))
outputs = model(input_ids=input_ids, labels=input_ids)
loss = outputs["loss"]
```

## Training Example

Train on generated math problems:

```bash
python examples/train_math.py --steps 1000 --batch-size 4 --device cuda
```

With neural memory enabled:

```bash
python examples/train_math.py --steps 1000 --use-neural-memory --memory-capacity 256
```

## Architecture

```
TitanMAC
├── Token Embeddings + Positional Embeddings
├── Persistent Tokens (learnable, bidirectional attention)
├── N × TitanBlock
│   ├── RMSNorm → WindowedAttention → Residual
│   └── RMSNorm → MLP → Residual
├── Optional: NeuralMemory (gradient-based)
├── Final RMSNorm
└── LM Head (optionally tied to embeddings)
```

## Configuration

Key parameters in `TitanMACConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 640 | Model dimension |
| `n_heads` | 10 | Attention heads |
| `n_layers` | 16 | Transformer layers |
| `window_size` | 512 | Local attention window |
| `n_persistent` | 16 | Global persistent tokens |
| `use_neural_memory` | False | Enable gradient-based memory |
| `titans_variant` | "MAC" | Memory integration (MAC/MAG/MAL) |

## Package Structure

```
titans_core/
├── config.py              # TitanMACConfig dataclass
├── blocks/
│   ├── norms.py          # RMSNorm
│   ├── mlp.py            # MLPBlock
│   └── titan_block.py    # TitanBlock
├── attn/
│   └── windowed_attention.py
├── memory/
│   ├── memory_bank.py    # Key-value retrieval
│   └── neural_memory.py  # Gradient-based memory
├── models/
│   ├── titanmac.py       # Main model
│   └── titanmac_wrapper.py  # HuggingFace wrapper
└── opt/
    ├── continuum_optimizer.py  # Nested optimizer
    ├── nested_controller.py    # LR modulation MLP
    ├── param_groups.py         # Parameter grouping
    ├── dmgd.py                 # Deep Momentum GD
    └── cms.py                  # Continuum Memory System
```

## Dependencies

- PyTorch >= 2.1
- Transformers >= 4.35 (for wrapper only)

## License

MIT
