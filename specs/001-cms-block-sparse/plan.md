# Implementation Plan: CMS Dynamic Block Sparse Linear Layer

**Branch**: `001-cms-block-sparse` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-cms-block-sparse/spec.md`

## Summary

Implement a block-sparse linear layer (`CMSBlockLinear`) that stores weights in Block-ELL format and makes dynamic topology decisions via CMS Level 2 updates. The goal is to reduce catastrophic forgetting by allowing the network to route around established pathways when learning new tasks, while achieving real computational speedup through hardware-aligned block sparsity (16x16 tiles matching GPU cache lines).

**Primary Requirements**:
- Drop-in replacement for `nn.Linear` with block-sparse storage
- Dynamic topology (prune/grow blocks) at CMS Level 2 frequency (~100 steps)
- Magnitude-based scoring heuristics with epsilon-greedy exploration
- ≥1.3x forward pass speedup at 50% density
- <30% relative forgetting on sequential task training

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch ≥2.1.0, Triton (existing in project), transformers ≥4.35.0
**Storage**: In-memory tensors (Block-ELL format), checkpointing via PyTorch state_dict
**Testing**: pytest ≥7.0 (existing `tests/unit/` structure)
**Target Platform**: Linux with NVIDIA GPU (Volta+ for Tensor Core acceleration)
**Project Type**: Single Python package extending existing `titans_core`
**Performance Goals**:
- Forward pass: 1.3x speedup at 50% density
- Training: 1.2x end-to-end speedup
- Topology decision overhead: <5% of training time
**Constraints**:
- Memory proportional to density (50% density = ~50% weight memory)
- Training stability: no loss spikes >2x running average at topology boundaries
- DDP compatibility: identical topology across all workers
**Scale/Scope**:
- Target model: TitanMAC 237M parameters
- MLP layers: 16 blocks, fc1 (640→2560), fc2 (2560→640)
- Block size: 16x16 = 256 parameters per block

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution is a template (not yet customized). Applying general best practices:

| Principle | Status | Notes |
|-----------|--------|-------|
| Library-First | ✅ PASS | CMSBlockLinear is a standalone layer module |
| Test-First | ✅ PASS | Plan includes unit tests before implementation |
| Integration Testing | ✅ PASS | CMS integration tests planned |
| Observability | ✅ PASS | Topology metrics logging in requirements |
| Simplicity | ✅ PASS | Starting with heuristics, learned controller deferred |

**No violations requiring justification.**

## Project Structure

### Documentation (this feature)

```text
specs/001-cms-block-sparse/
├── plan.md              # This file
├── research.md          # Phase 0 output - technical research
├── data-model.md        # Phase 1 output - entity definitions
├── quickstart.md        # Phase 1 output - usage guide
├── contracts/           # Phase 1 output - API contracts
│   └── cms_block_linear.py  # Type stubs / interface contract
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
titans_core/
├── layers/                      # NEW: Block-sparse layer module
│   ├── __init__.py
│   ├── block_sparse.py          # CMSBlockLinear implementation
│   └── block_ell.py             # Block-ELL format utilities
├── kernels/
│   ├── memory_mlp.py            # EXISTING: Triton kernels
│   ├── block_ell_forward.py     # NEW: Triton forward kernel
│   └── block_ell_backward.py    # NEW: Triton backward kernels
├── opt/
│   ├── cms.py                   # EXISTING: ContinuumMemorySystem
│   ├── deep_nested_optimizer.py # MODIFY: Add topology step integration
│   └── topology_scorer.py       # NEW: Magnitude-based scoring
├── blocks/
│   └── mlp.py                   # MODIFY: Support CMSBlockLinear option
└── config.py                    # MODIFY: Add block_sparse config options

tests/
├── unit/
│   ├── test_block_ell_format.py       # NEW: Block-ELL utilities
│   ├── test_cms_block_linear.py       # NEW: Layer correctness
│   ├── test_topology_scorer.py        # NEW: Scoring heuristics
│   └── test_deep_nested_optimizer.py  # EXISTING: Extend for topology
└── integration/
    ├── test_cms_topology_integration.py  # NEW: CMS Level 2 integration
    └── test_ddp_topology_sync.py         # NEW: DDP synchronization
```

**Structure Decision**: Extends existing `titans_core` package structure. New `layers/` directory for block-sparse components. Triton kernels go in existing `kernels/` directory following `memory_mlp.py` patterns.

## Complexity Tracking

> No constitution violations. Table not required.

## Phase 0: Research Tasks

Based on Technical Context unknowns and dependencies:

### Research Topics

1. **Block-ELL Format Best Practices**
   - Optimal memory layout for GPU cache efficiency
   - Index representation (int32 vs int64, sentinel values)
   - Comparison with CSR/COO for block sparsity

2. **Triton Block-Sparse Kernels**
   - Existing Triton block-sparse implementations (references)
   - Optimal tile sizes for A100/H100 Tensor Cores
   - Backward pass patterns for block-sparse matmul

3. **Dynamic Topology Scoring**
   - RigL gradient-based growth criteria
   - Magnitude pruning pitfalls and alternatives
   - Exploration/exploitation balance (epsilon-greedy tuning)

4. **DDP Topology Synchronization**
   - When to sync (every step vs Level 2 only)
   - What to sync (scores vs decisions)
   - All-reduce vs broadcast patterns

5. **Catastrophic Forgetting Baselines**
   - Dense model forgetting rates on math→NLP task sequence
   - EWC/L2 regularization baselines for comparison
   - Pathway overlap measurement techniques

**Output**: `research.md` with decisions, rationale, and alternatives considered.

## Phase 1: Design Artifacts

### Data Model (`data-model.md`)

Key entities from spec:
- **Block**: 16x16 weight tile with position, age, score
- **Topology**: Block-ELL indices mapping output rows to input columns
- **BlockScore**: EMA of gradient norms per block
- **TopologyDecision**: Prune/grow event with before/after topology

### API Contracts (`contracts/`)

From functional requirements:
- `CMSBlockLinear.__init__(in_features, out_features, tile_size, density)`
- `CMSBlockLinear.forward(x) → Tensor`
- `CMSBlockLinear.accumulate_scores()` - called after backward
- `CMSBlockLinear.topology_step(generator) → num_swaps` - called at Level 2
- `CMSBlockLinear.get_topology_stats() → Dict` - for logging

### Quickstart (`quickstart.md`)

Usage example:
```python
from titans_core.layers import CMSBlockLinear
from titans_core.config import TitanMACConfig

# Configure sparse layers
config = TitanMACConfig(
    use_block_sparse=True,
    block_sparse_tile_size=16,
    block_sparse_density=0.5,
    block_sparse_layers=(8, 9, 10, 11, 12),
)

# Create model (MLP layers 8-12 use CMSBlockLinear)
model = create_titanmac_model(config)

# Train with DeepNestedOptimizer (topology steps automatic)
optimizer = DeepNestedOptimizer(model, ...)
```

## Implementation Phases (Preview)

*Detailed tasks generated by `/speckit.tasks`*

### Phase A: Foundation
- Block-ELL format utilities with tests
- PyTorch reference forward/backward (no Triton yet)
- Unit tests for correctness vs dense

### Phase B: Triton Kernels
- Forward kernel with benchmarks
- Backward kernels (grad_values, grad_input)
- Performance validation vs dense

### Phase C: CMS Integration
- Topology scorer (magnitude + epsilon-greedy)
- Level 2 topology_step integration
- DDP synchronization

### Phase D: Model Integration
- Config options for TitanMACConfig
- MLP layer replacement logic
- End-to-end training test

### Phase E: Validation
- Forgetting measurement experiments
- Speedup benchmarks
- Topology dynamics analysis
