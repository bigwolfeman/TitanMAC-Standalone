# Paper-Faithful TitanMAC Refactor

**Status**: COMPLETE
**Started**: 2025-12-14
**Completed**: 2025-12-14
**VLT Thread**: `paper_faithful_refactor`

## Overview

Refactoring TitanMAC implementation to match the Titans paper (arxiv 2501.00663) and Nested Learning paper (NeurIPS 2025).

## Audit Summary

The implementation had ~55% fidelity to papers. Critical issues:

| Issue | Severity | Status |
|-------|----------|--------|
| Memory is embedding table, not deep MLP | CRITICAL | Phase 1 |
| MAC memory tokens unreachable (T positions away) | CRITICAL | Phase 2 |
| Windowed attention is O(T²), not O(T*w) | HIGH | Phase 3 |
| Missing 1D conv after QKV | MEDIUM | Phase 4 |
| Persistent tokens can't see sequence | MEDIUM | Phase 5 |
| MAG/MAL don't match paper | MEDIUM | Phase 6 |

## Phase 1: Deep MLP Memory

### Paper Specification (Titans §3.1)

The paper's key insight is that **memory M is a deep MLP** (L_M >= 2 layers).

**Equations**:
```
Loss:     l(M; x) = ||M(k_t) - v_t||²    (Eq. 12)
Retrieve: y_t = M*(q_t)                   (Eq. 15)
Update:   S_t = η_t * S_{t-1} - θ_t * ∇l  (Eq. 13)
          M_t = (1 - α_t) * M_{t-1} + S_t (Eq. 14)
```

Where `M_t` refers to **all MLP weights** at time t. The gradient `∇l` is w.r.t. those weights.

**Key Insight**: This is **test-time learning** — gradient descent on MLP weights during inference.

### Current Implementation (WRONG)

```python
# neural_memory.py:160
self.memory = nn.Parameter(torch.zeros(capacity, self.d_memory))  # Embedding table!

# Uses softmax attention over slots - NOT paper's MLP
scores = torch.matmul(k, self.memory.T)
weights = F.softmax(scores / math.sqrt(self.d_memory), dim=-1)
retrieved = torch.matmul(weights, self.memory)
```

### Target Implementation

```python
class DeepMemoryMLP(nn.Module):
    def __init__(self, d_model, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_model, d_model))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)  # M(k) = simple MLP forward pass
```

### Changes Required

- [ ] Create `DeepMemoryMLP` class
- [ ] Replace `nn.Parameter(zeros)` with MLP
- [ ] Change `compute_loss` to use MLP forward, not softmax attention
- [ ] Update memory via gradient descent on MLP weights
- [ ] Keep momentum buffer for MLP weight updates
- [ ] Update config to support `n_memory_layers`

---

## Phase 2: Segment-wise MAC

### Paper Specification (Titans §4.1)

MAC processes sequences in **segments** with fixed N_l memory tokens per segment:

```
For each segment S^(i):
    h_t = M*_{t-1}(q_t)                    # Retrieve from PREVIOUS memory state
    S̃ = [p_1...p_np] || h_t || S^(i)      # Concat: persistent, N_l memory, segment
    y = Attn(S̃)                            # Attention over combined
    M_t = M_{t-1}(y)                       # Update memory with output
```

### Current Implementation (WRONG)

```python
# titanmac.py:217
x_combined = torch.cat([persistent, h_proj, x], dim=1)
# h_proj has T tokens, x has T tokens
# Memory at positions 0..T, input at T..2T
# With window_size << T, input can't reach its memory!
```

### Target Implementation

- Process in segments of size `segment_size` (e.g., 512)
- For each segment:
  1. Retrieve N_l memory tokens (fixed, small number like 16-64)
  2. Concat: `[persistent | N_l memory | segment]`
  3. Full attention within segment (or window >= segment)
  4. Update memory with segment output
- Memory state persists across segments

---

## Phase 3: Block-Sparse Attention

### Current Implementation (WRONG)

Uses `F.scaled_dot_product_attention` with mask — computes full O(T²) matrix then masks.

### Target Implementation

True O(T*w) block-sparse attention:
- Never materialize full T×T matrix
- Use block-local patterns
- Options: custom CUDA kernel, Triton, or chunked computation

---

## Progress Log

### 2025-12-14

- Initial audit identified critical issues
- Verified claims with code inspection
- Created refactor plan
- Starting Phase 1: Deep MLP Memory

---

## Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `titans_core/memory/neural_memory.py` | 1 | Replace embedding with DeepMemoryMLP |
| `titans_core/models/titanmac.py` | 2 | Implement segment-wise MAC |
| `titans_core/attn/windowed_attention.py` | 3 | Block-sparse attention |
| `titans_core/config.py` | 1,2,3 | Add new config params |
| `examples/train_math.py` | All | Test with toy trainer |
