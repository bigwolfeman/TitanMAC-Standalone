# Benchmarking & Data Logging Specification

**Feature Branch**: `001-cms-block-sparse`
**Created**: 2025-12-25
**Purpose**: Define benchmark phases, metrics, and logging schemas for CMS dynamic block-sparse validation

---

## Overview

This document specifies the benchmarking methodology and data logging infrastructure for validating the CMS Dynamic Block Sparse Linear Layer. The benchmark suite is organized in phases that progressively validate correctness, performance, and anti-forgetting capabilities.

---

## Phase 0: Dense Baselines (Before Sparse Implementation)

Establish ground-truth measurements before implementing sparse kernels. These baselines are required for meaningful comparison.

| Experiment | Description | Steps | Key Outputs |
|------------|-------------|-------|-------------|
| **Dense Task A Only** | Train on math problems until convergence | 10K | `accuracy_A_baseline`, final loss, training time |
| **Dense Task A->B** | Sequential training without sparsity | 10K + 10K | `forgetting_rate_dense` = (acc_A_before - acc_A_after) / acc_A_before |
| **Dense with EWC** | State-of-art continual learning comparison | 10K + 10K | `forgetting_rate_ewc`, compute overhead |
| **Static Sparse 50%** | Random fixed topology (lower bound) | 10K + 10K | `forgetting_rate_static`, speedup vs dense |

**Expected Baselines (from literature)**:
- Dense model: 40-60% forgetting
- EWC: 25-40% forgetting
- Static sparse: ~same as dense or worse
- Dynamic block-sparse target: <30% forgetting

---

## Phase 1: Block-Sparse Correctness

Validate that block-sparse implementation produces numerically correct results.

### 1.1 Sparse vs Dense Numerical Match

```python
# Forward pass validation
dense_output = dense_layer(input)
sparse_output = sparse_layer(input)

assert torch.allclose(dense_output, sparse_output, atol=1e-5, rtol=1e-4), \
    f"Forward mismatch: max_diff={torch.abs(dense_output - sparse_output).max()}"
```

| Test | Tolerance | Pass Criteria |
|------|-----------|---------------|
| Forward pass | atol=1e-5 | Max absolute difference < 1e-5 |
| Backward (grad_weight) | atol=1e-4 | Gradient diff < 1e-4 |
| Backward (grad_input) | atol=1e-4 | Gradient diff < 1e-4 (atomics may add noise) |

### 1.2 Gradient Correctness

```python
from torch.autograd import gradcheck

# Test block-ELL matmul kernel gradients
sparse_fn = lambda x: block_sparse_matmul(x, values, col_indices)
assert gradcheck(sparse_fn, (input,), eps=1e-6, atol=1e-4, rtol=1e-3)
```

### 1.3 State Dict Round-Trip

```python
# Save checkpoint
original_topology = sparse_layer.col_indices.clone()
original_scores = sparse_layer.block_scores.clone()
torch.save(sparse_layer.state_dict(), "checkpoint.pt")

# Load checkpoint
new_layer = BlockSparseLinear(...)
new_layer.load_state_dict(torch.load("checkpoint.pt"))

# Verify topology preserved
assert torch.equal(original_topology, new_layer.col_indices)
assert torch.allclose(original_scores, new_layer.block_scores, atol=1e-6)
```

---

## Phase 2: Performance Benchmarks

Measure computational efficiency gains from block-sparse implementation.

### 2.1 Forward Latency

| Batch Size | Dense Baseline (ms) | Block-Sparse 50% (ms) | Speedup Target |
|------------|--------------------|-----------------------|----------------|
| 1 | TBD | TBD | >= 1.3x |
| 4 | TBD | TBD | >= 1.3x |
| 16 | TBD | TBD | >= 1.4x |
| 64 | TBD | TBD | >= 1.5x |

```python
# Benchmark script
def benchmark_forward(layer, input, warmup=10, trials=100):
    for _ in range(warmup):
        _ = layer(input)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(trials):
        _ = layer(input)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / trials
```

### 2.2 Backward Latency

| Component | Expected Overhead | Notes |
|-----------|------------------|-------|
| grad_values | None | Unique write locations per (r,k) |
| grad_input | 10-30% | Atomic contention from scatter-add |

### 2.3 End-to-End Step Time

Full optimizer step including topology updates:

```python
# Measure full training step
def benchmark_step(model, optimizer, batch, warmup=5, trials=20):
    for _ in range(warmup):
        loss = model(batch)['loss']
        loss.backward()
        optimizer.step(loss.item())
        optimizer.zero_grad()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        loss = model(batch)['loss']
        loss.backward()
        optimizer.step(loss.item())
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)
```

### 2.4 Memory Footprint

| Metric | Measurement Method |
|--------|-------------------|
| Peak memory | `torch.cuda.max_memory_allocated()` |
| Weight storage | Direct calculation: `values.numel() * 4` bytes |
| Index overhead | `col_indices.numel() * 4` bytes |
| Score storage | `block_scores.numel() * 4` bytes |

**Expected at 50% density**: ~50% weight memory reduction (plus ~6% index overhead)

### 2.5 Throughput

```python
tokens_per_sec = (batch_size * seq_len * trials) / elapsed_time
```

| Configuration | Target Throughput Improvement |
|---------------|------------------------------|
| 50% density | >= 1.2x tokens/sec |
| 25% density | >= 1.5x tokens/sec |

---

## Phase 3: Topology Dynamics

Validate that topology evolution exhibits healthy dynamics.

### 3.1 Swap Rate Over Time

```python
# Track swaps across training
swap_rate = num_swaps / total_blocks  # Per topology decision
```

| Metric | Healthy Range | Interpretation |
|--------|--------------|----------------|
| Swap rate | 1-10% | Too low: stagnation. Too high: instability |
| Swap rate trend | Decreasing | Should stabilize as training progresses |

### 3.2 Block Age Distribution

```python
# Histogram of block ages at each topology decision
block_ages = global_step - block_creation_step
```

| Metric | Healthy Signal |
|--------|---------------|
| Mean age | Increasing early, stable later |
| Age variance | Non-zero (diverse ages) |
| Max age | Should see some blocks survive entire training |

### 3.3 Column Entropy

Measures diversity of input column usage:

```python
# H(columns) = -sum(p_i * log(p_i))
column_counts = torch.bincount(col_indices.flatten())
probs = column_counts / column_counts.sum()
entropy = -torch.sum(probs * torch.log(probs + 1e-10))
max_entropy = torch.log(torch.tensor(num_columns, dtype=torch.float))
normalized_entropy = entropy / max_entropy
```

| Metric | Healthy Range | Interpretation |
|--------|--------------|----------------|
| Normalized entropy | > 0.5 | Topology uses diverse input columns |
| Entropy trend | Stable or increasing | Not collapsing to narrow subset |

### 3.4 Score Correlation with Forgetting

```python
# Hypothesis: high-scoring blocks correlate with low forgetting
correlation = torch.corrcoef(
    torch.stack([block_scores, per_block_forgetting])
)[0, 1]
```

Expected: Negative correlation (higher scores = less forgetting contribution)

---

## Phase 4: Forgetting Experiments

Core validation of anti-forgetting capabilities.

### 4.1 Accuracy Retention

```python
# Forgetting measurement protocol
accuracy_A_before = evaluate(model, task_A_eval)  # After Task A training
train(model, task_B, steps=10000)                  # Train on Task B
accuracy_A_after = evaluate(model, task_A_eval)    # Re-evaluate Task A

forgetting_pct = 100 * (accuracy_A_before - accuracy_A_after) / accuracy_A_before
retention_pct = 100 * accuracy_A_after / accuracy_A_before
```

| Target | Dense Baseline | Block-Sparse Target | Success Criterion |
|--------|---------------|---------------------|-------------------|
| Forgetting | 40-60% | <30% | >= 30% relative improvement |
| Retention | 40-60% | >70% | Accuracy remains usable |

### 4.2 Pathway Overlap

```python
# Track which blocks are active per task
blocks_active_A = set(blocks with high_activation during Task A)
blocks_active_B = set(blocks with high_activation during Task B)

overlap = len(blocks_active_A & blocks_active_B) / len(blocks_active_A | blocks_active_B)
```

| Target | Interpretation |
|--------|---------------|
| Overlap < 70% | Distinct pathways for distinct tasks |
| Overlap > 90% | Tasks using same pathways (forgetting likely) |

### 4.3 Task-Specific Block Sets

Log per-task block activation patterns for post-hoc analysis:

```python
task_block_log = {
    "task_A": {
        "layer_0": [active_col_indices],
        "layer_1": [active_col_indices],
        ...
    },
    "task_B": {...}
}
```

---

## Data Logging Schemas

### Per-Step (Level 0) - Every Training Step

Logged every step. High-frequency core metrics.

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Global training step |
| `loss` | float | Total loss value |
| `ce_loss` | float | Cross-entropy component |
| `aux_loss` | float | Auxiliary losses (router, etc.) |
| `grad_norm` | float | Global gradient norm (post-clip) |
| `lr_multipliers` | List[float] | Per-group LR multipliers from controller |
| `tokens_per_sec` | float | Training throughput |

```python
# Example log entry
{
    "step": 1000,
    "loss": 2.34,
    "ce_loss": 2.30,
    "aux_loss": 0.04,
    "grad_norm": 0.85,
    "lr_multipliers": [1.02, 0.98],
    "tokens_per_sec": 45000.0
}
```

### Per-Score-Step (Level 1) - Every 10 Steps

Score accumulation metrics. Aligned with CMS Level 1 frequency.

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Global training step |
| `block_score_ema_mean` | float | Mean of block score EMAs across all layers |
| `block_score_ema_std` | float | Std of block score EMAs |
| `activation_norm_mean` | float | Mean activation norm (for growth candidates) |
| `error_norm_mean` | float | Mean error norm (for growth candidates) |
| `block_age_mean` | float | Mean age of active blocks |

```python
# Example log entry
{
    "step": 100,
    "block_score_ema_mean": 0.0234,
    "block_score_ema_std": 0.0089,
    "activation_norm_mean": 1.45,
    "error_norm_mean": 0.023,
    "block_age_mean": 45.2
}
```

### Per-Topology-Step (Level 2) - Every 100 Steps

Topology decision metrics. Aligned with CMS Level 2 frequency.

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Global training step |
| `num_swaps` | int | Total blocks swapped this decision |
| `swap_rate` | float | Fraction of blocks swapped |
| `density` | float | Current active density (should be ~target) |
| `column_entropy` | float | Normalized entropy of column usage |
| `avg_block_age` | float | Average age of active blocks |
| `pruned_positions` | List[Tuple[int,int]] | (row, col) of pruned blocks |
| `grown_columns` | List[int] | Column indices of newly grown blocks |
| `exploration_triggered` | bool | Whether epsilon-greedy exploration fired |

```python
# Example log entry
{
    "step": 1000,
    "num_swaps": 12,
    "swap_rate": 0.047,
    "density": 0.502,
    "column_entropy": 0.78,
    "avg_block_age": 234.5,
    "pruned_positions": [(3, 5), (7, 12), ...],
    "grown_columns": [8, 15, 22],
    "exploration_triggered": False
}
```

### Per-Layer Topology Snapshot

Full topology state for visualization and debugging.

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | str | Layer identifier (e.g., "block_3.mlp.fc1") |
| `col_indices` | Tensor[R, K] | Active column indices |
| `block_scores` | Tensor[R, K] | Current score EMAs |
| `block_ages` | Tensor[R, K] | Steps since block creation |

```python
# Example snapshot (saved to file, not wandb)
{
    "layer_name": "block_3.mlp.fc1",
    "col_indices": [[0, 5, 12], [1, 8, 15], ...],  # [R, K]
    "block_scores": [[0.034, 0.089, 0.012], ...],
    "block_ages": [[500, 100, 234], ...]
}
```

### Per-Eval

Evaluation metrics logged after each evaluation run.

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Global training step |
| `task` | str | Task identifier ("task_A", "task_B") |
| `accuracy` | float | Evaluation accuracy |
| `perplexity` | float | Evaluation perplexity |
| `forgetting_pct` | float | Percent forgetting vs peak (if applicable) |
| `pathway_overlap` | float | Block overlap with other task (if applicable) |

```python
# Example log entry
{
    "step": 20000,
    "task": "task_A",
    "accuracy": 0.72,
    "perplexity": 15.4,
    "forgetting_pct": 18.2,
    "pathway_overlap": 0.45
}
```

### Per-Run Summary

Aggregated metrics logged at end of training run.

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | str | Unique run identifier |
| `config` | Dict | Full configuration dict |
| `final_accuracy_A` | float | Final Task A accuracy |
| `final_accuracy_B` | float | Final Task B accuracy |
| `forgetting_rate` | float | Overall forgetting percentage |
| `speedup_forward` | float | Forward pass speedup vs dense |
| `speedup_e2e` | float | End-to-end training speedup |
| `memory_reduction` | float | Peak memory reduction percentage |
| `topology_stability` | float | Final swap rate (lower = more stable) |
| `total_training_time` | float | Wall-clock time in seconds |

```python
# Example summary
{
    "experiment_id": "cms_block_sparse_v1_20251225_143022",
    "config": {"density": 0.5, "block_size": 16, ...},
    "final_accuracy_A": 0.72,
    "final_accuracy_B": 0.84,
    "forgetting_rate": 18.2,
    "speedup_forward": 1.45,
    "speedup_e2e": 1.22,
    "memory_reduction": 0.48,
    "topology_stability": 0.02,
    "total_training_time": 3600.5
}
```

---

## WandB Integration

### Logging Grouping Convention

Use prefixes to organize metrics in WandB dashboard:

| Prefix | Category | Example Metrics |
|--------|----------|-----------------|
| `train/` | Core training | `train/loss`, `train/ce_loss`, `train/tokens_per_sec` |
| `optimizer/` | Optimizer state | `optimizer/lr_mult_core`, `optimizer/grad_norm`, `optimizer/ema_loss` |
| `topology/` | Topology dynamics | `topology/swap_rate`, `topology/column_entropy`, `topology/avg_block_age` |
| `forgetting/` | Forgetting metrics | `forgetting/accuracy_A`, `forgetting/retention_pct`, `forgetting/pathway_overlap` |
| `perf/` | Performance | `perf/forward_ms`, `perf/step_ms`, `perf/memory_gb` |

### Example wandb.log() Calls

```python
import wandb

# Per-step logging (Level 0)
wandb.log({
    "train/loss": loss.item(),
    "train/ce_loss": ce_loss.item(),
    "train/aux_loss": aux_loss.item(),
    "train/tokens_per_sec": tokens_per_sec,
    "optimizer/grad_norm": grad_norm.item(),
    "optimizer/lr_mult_core": lr_multipliers[0].item(),
    "optimizer/lr_mult_embed": lr_multipliers[1].item(),
}, step=global_step)

# Per-score-step logging (Level 1, every 10 steps)
if global_step % 10 == 0:
    wandb.log({
        "topology/block_score_ema_mean": block_scores.mean().item(),
        "topology/block_score_ema_std": block_scores.std().item(),
        "topology/activation_norm_mean": activation_norms.mean().item(),
        "topology/error_norm_mean": error_norms.mean().item(),
        "topology/block_age_mean": block_ages.float().mean().item(),
    }, step=global_step)

# Per-topology-step logging (Level 2, every 100 steps)
if global_step % 100 == 0:
    wandb.log({
        "topology/num_swaps": num_swaps,
        "topology/swap_rate": swap_rate,
        "topology/density": current_density,
        "topology/column_entropy": column_entropy.item(),
        "topology/avg_block_age": avg_block_age.item(),
        "topology/exploration_triggered": int(exploration_triggered),
    }, step=global_step)

# Per-eval logging
wandb.log({
    f"forgetting/accuracy_{task}": accuracy,
    f"forgetting/perplexity_{task}": perplexity,
    f"forgetting/forgetting_pct_{task}": forgetting_pct,
    f"forgetting/pathway_overlap": pathway_overlap,
}, step=global_step)

# Performance benchmarks
wandb.log({
    "perf/forward_ms": forward_latency_ms,
    "perf/backward_ms": backward_latency_ms,
    "perf/step_ms": step_latency_ms,
    "perf/memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    "perf/tokens_per_sec": tokens_per_sec,
}, step=global_step)
```

### Custom WandB Tables for Topology Snapshots

```python
# Log topology snapshot as wandb.Table
topology_table = wandb.Table(
    columns=["layer", "row", "col_idx", "score", "age"]
)
for layer_name, (col_indices, scores, ages) in topology_snapshots.items():
    for r in range(col_indices.shape[0]):
        for k in range(col_indices.shape[1]):
            topology_table.add_data(
                layer_name,
                r,
                col_indices[r, k].item(),
                scores[r, k].item(),
                ages[r, k].item()
            )

wandb.log({"topology/snapshot": topology_table}, step=global_step)
```

### Histogram Logging

```python
# Block score distribution
wandb.log({
    "topology/block_scores_hist": wandb.Histogram(block_scores.flatten().cpu().numpy()),
    "topology/block_ages_hist": wandb.Histogram(block_ages.flatten().cpu().numpy()),
}, step=global_step)
```

---

## Benchmark Execution Order

1. **Phase 0**: Run all dense baselines (one-time, results cached)
2. **Phase 1**: Correctness tests (must pass before proceeding)
3. **Phase 2**: Performance benchmarks on single-batch scenarios
4. **Phase 3**: Topology dynamics during short training runs (1K steps)
5. **Phase 4**: Full forgetting experiments (10K + 10K steps)

---

## Success Criteria Summary

| Phase | Metric | Target | Blocking? |
|-------|--------|--------|-----------|
| 1 | Numerical match | < 1e-5 forward, < 1e-4 grad | Yes |
| 1 | gradcheck pass | All kernels pass | Yes |
| 2 | Forward speedup | >= 1.3x at 50% density | Yes |
| 2 | Memory reduction | ~50% weight storage | No |
| 3 | Swap rate | 1-10% per decision | No |
| 3 | Column entropy | > 0.5 (normalized) | No |
| 4 | Forgetting reduction | >= 30% relative improvement vs dense | Yes |
| 4 | Pathway overlap | < 70% | No |
| 5 | OOD forgetting | < 15% | Yes (easy bar) |
| 5 | ID-Semantic forgetting | < 30% (Silver tier) | No (research goal) |
| 5 | ID-Context forgetting | < 50% | No (stretch goal) |
| 6 | NLP-Synth accuracy | > 80% | No |
| 6 | Cross-domain forgetting | Better than dense baseline | No |

---

## File Outputs

| File | Contents | Format |
|------|----------|--------|
| `baselines.json` | Phase 0 dense baselines | JSON |
| `correctness_report.json` | Phase 1 test results | JSON |
| `perf_benchmarks.csv` | Phase 2 latency/memory data | CSV |
| `topology_snapshots/` | Per-step topology states | Torch .pt files |
| `forgetting_analysis.json` | Phase 4 results | JSON |
| `run_summary.json` | Per-run aggregated metrics | JSON |

---

## Phase 5: In-Distribution vs Out-of-Distribution Forgetting

This phase tests the TRUE difficulty gradient for block-sparse anti-forgetting.

### Understanding the Distinction

| Type | Description | Token Overlap | Challenge Level |
|------|-------------|---------------|-----------------|
| **OOD** | Different vocabulary between tasks | ~0% | Easy - topology can trivially separate |
| **ID-Semantic** | Same tokens, different meaning | 100% | Hard - same inputs, different outputs |
| **ID-Syntactic** | Same tokens, different structure | 100% | Medium - position-dependent meaning |
| **ID-Context** | Same tokens, mode-switched by prefix | 99% | Very Hard - single token controls routing |

### 5.1 OOD Forgetting (Baseline - Expected Easy)

```
Task A: Math tokens {0-9, +, -, *, =}
  "5 + 3 = 8"
  "7 * 2 = 14"

Task B: NLP tokens {the, cat, sat, NOUN, VERB, ...}
  "the cat sat | SUBJ: cat"
  "dogs bark loudly | ADV: loudly"
```

**Hypothesis**: Topology can easily separate because different input columns are activated.

**Success Tier**:
| Forgetting | Interpretation |
|------------|----------------|
| < 10% | Expected - trivial separation |
| 10-20% | Acceptable - some embedding overlap |
| > 20% | Concerning - investigate topology collapse |

### 5.2 ID-Semantic Forgetting (Hard Mode)

```
Task A: Standard Arithmetic (mod ∞)
  "5 + 3 = 8"
  "12 + 9 = 21"

Task B: Modular Arithmetic (mod 7)
  "5 + 3 = 1"    # (5+3=8) mod 7 = 1
  "12 + 9 = 0"   # (12+9=21) mod 7 = 0
```

**The Critical Test**: Same input "5 + 3 =" must produce DIFFERENT outputs depending on task.

**Implementation**:
```python
class ModularMathDataset(IterableDataset):
    def __init__(self, modulus: int = 7):
        self.modulus = modulus

    def _generate_problem(self) -> str:
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        op = random.choice(['+', '-', '*'])

        if op == '+': result = (a + b) % self.modulus
        elif op == '-': result = (a - b) % self.modulus
        else: result = (a * b) % self.modulus

        return f"{a} {op} {b} = {result}"
```

**Success Tier**:
| Forgetting | Interpretation |
|------------|----------------|
| < 15% | 🏆 Gold - topology successfully separates identical inputs |
| 15-30% | 🥈 Silver - partial separation, better than dense |
| 30-50% | 🥉 Bronze - limited benefit |
| > 50% | ❌ Fail - no better than static sparse |

### 5.3 ID-Syntactic Forgetting (Word Order)

```
Task A: Subject-Verb-Object (SVO)
  "the cat chased the mouse | SUBJ: cat"
  "the dog bit the man | SUBJ: dog"

Task B: Object-Verb-Subject (OVS)
  "the mouse chased the cat | SUBJ: cat"  # Same words, cat is now OBJECT
  "the man bit the dog | SUBJ: dog"       # dog is now OBJECT being bitten
```

**Why This Tests Position-Awareness**: Same tokens, but position determines role.

### 5.4 ID-Context Forgetting (Mode-Switched - Hardest)

```
Task A:
  "MODE:STD | 5 + 3 = 8"
  "MODE:STD | 12 * 4 = 48"

Task B:
  "MODE:MOD7 | 5 + 3 = 1"
  "MODE:MOD7 | 12 * 4 = 6"
```

**The Ultimate Test**: A SINGLE prefix token must route to entirely different computation.

**Key Question**: Can topology learn to be mode-aware, or does it require explicit task tokens?

### 5.5 Measurement Protocol for ID Forgetting

```python
class IDForgettingBenchmark:
    def __init__(self):
        # Fixed eval inputs - SAME for both tasks
        self.shared_inputs = generate_arithmetic_inputs(n=500, seed=42)
        # e.g., ["5 + 3 =", "12 * 4 =", "99 - 7 =", ...]

    def eval_standard(self, model) -> float:
        """Accuracy on standard arithmetic."""
        return self._eval_with_targets(model, self.shared_inputs, mode="standard")

    def eval_modular(self, model, mod: int = 7) -> float:
        """Accuracy on modular arithmetic."""
        return self._eval_with_targets(model, self.shared_inputs, mode=f"mod{mod}")

    def compute_topology_divergence(self, topology_A, topology_B) -> float:
        """
        How different is the topology after Task B vs Task A?

        High divergence on SAME inputs = model learned separate pathways (good!)
        Low divergence = blocks are shared (forgetting risk)
        """
        intersection = set(topology_A.flatten()) & set(topology_B.flatten())
        union = set(topology_A.flatten()) | set(topology_B.flatten())
        overlap = len(intersection) / len(union)
        return 1.0 - overlap  # 0 = identical, 1 = completely different
```

### 5.6 Block Divergence Tracking

For ID forgetting, we need to track whether topology CHANGES for same inputs:

```python
# After Task A training
snapshot_A = {
    "inputs": shared_inputs,
    "topology_per_layer": {layer: col_indices.clone() for layer, col_indices in ...},
    "activations": record_block_activations(model, shared_inputs),
}

# After Task B training
snapshot_B = {
    "inputs": shared_inputs,  # SAME inputs
    "topology_per_layer": {layer: col_indices.clone() for layer, col_indices in ...},
    "activations": record_block_activations(model, shared_inputs),
}

# Compute divergence
divergence = compute_topology_divergence(snapshot_A, snapshot_B)
# Target: > 0.3 (at least 30% of blocks should differ)
```

---

## Phase 6: NLP Benchmarks

Efficient NLP tasks that parallel the math benchmarks for cross-domain forgetting validation.

### 6.1 Synthetic NLP Patterns (Recommended for Speed)

Use generated patterns that:
- Share tokenizer with math (extended vocab ~120 tokens)
- Have deterministic correct answers
- Test linguistic structure, not world knowledge

```python
class SyntheticNLPDataset(IterableDataset):
    """
    Generated linguistic patterns parallel to math generation.
    """
    PATTERNS = [
        ("pos", "The quick fox jumps | quick: ADJ"),
        ("antonym", "hot is to cold as big is to: small"),
        ("reorder", "jumped fox the brown | the brown fox jumped"),
        ("cloze", "The ___ barks loudly | dog"),
        ("sequence", "A B A B A: B"),
    ]

    def _generate_pos_example(self) -> str:
        templates = [
            ("The {adj} {noun} {verb}", "adj", "{adj}: ADJ"),
            ("A {noun} {verb} {adv}", "adv", "{adv}: ADV"),
        ]
        # ... fill templates from word banks
```

### 6.2 Extended Vocabulary

Minimal extension to math tokenizer:

```python
NLP_VOCAB = {
    # Math vocab: 0-18 (existing)
    # Core words (~100 tokens)
    'the': 19, 'a': 20, 'is': 21, 'to': 22, 'of': 23,
    'dog': 24, 'cat': 25, 'fox': 26, 'bird': 27, 'mouse': 28,
    'big': 29, 'small': 30, 'hot': 31, 'cold': 32, 'quick': 33,
    'run': 34, 'jump': 35, 'sit': 36, 'bark': 37, 'chase': 38,
    # POS tags
    'NOUN': 39, 'VERB': 40, 'ADJ': 41, 'ADV': 42, 'SUBJ': 43,
    # Special
    ':': 44, '|': 45, '_': 46,
    # Mode tokens for context-switched experiments
    'MODE': 47, 'STD': 48, 'MOD7': 49,
}
# Total: ~50 NLP tokens + 19 math tokens = ~70 total (fits small model)
```

### 6.3 TinyStories Subset (Real Language Validation)

For final validation after synthetic experiments pass:

```python
from datasets import load_dataset

# Load small subset (~50K examples)
dataset = load_dataset("roneneldan/TinyStories", split="train[:50000]")

# Eval: perplexity on held-out stories
val_set = load_dataset("roneneldan/TinyStories", split="validation[:1000]")
```

**Use only if synthetic NLP passes** - real language is slower but validates generalization.

### 6.4 NLP Experiment Matrix

| Experiment | Task A | Task B | Token Overlap | Purpose |
|------------|--------|--------|---------------|---------|
| **OOD-NLP** | Math | NLP-Synth | 0% | Baseline OOD |
| **OOD-Real** | Math | TinyStories | 0% | Real language OOD |
| **ID-Grammar** | SVO patterns | OVS patterns | 100% | Syntactic ID |
| **ID-Semantic-NLP** | "X is Y" | "X is not Y" | 95% | Semantic ID |

---

## Experiment Difficulty Gradient

Complete experiment ordering from easiest to hardest:

| Rank | Experiment | Type | Token Overlap | Expected Forgetting |
|------|------------|------|---------------|---------------------|
| 1 | Math → NLP-Synth | OOD | 0% | < 15% |
| 2 | Math → TinyStories | OOD | 0% | < 20% |
| 3 | SVO → OVS | ID-Syntactic | 100% | 15-35% |
| 4 | Standard → Modular | ID-Semantic | 100% | 20-40% |
| 5 | MODE:STD → MODE:MOD7 | ID-Context | 99% | 25-50% |

**Interpretation Guide**:
- If experiments 1-2 fail: Block-sparse has fundamental issues
- If experiments 1-2 pass, 3-5 fail: Topology lacks context-awareness (expected limitation)
- If experiments 1-4 pass, 5 fails: Need explicit task routing (known hard problem)
- If all pass: Dynamic block-sparse is highly effective

---

## References

- CMS levels aligned with `DeepNestedOptimizer.cms_frequencies = [1, 10, 100]`
- Block-ELL format from `research.md` Section 1
- Forgetting protocol from `research.md` Section 5
- Scoring heuristics from `research.md` Section 3
- ID vs OOD forgetting analysis: Catastrophic Interference literature (McCloskey & Cohen 1989)
