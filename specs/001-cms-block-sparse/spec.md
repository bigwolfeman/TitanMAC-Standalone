# Feature Specification: CMS Dynamic Block Sparse Linear Layer

**Feature Branch**: `001-cms-block-sparse`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "CMS Dynamic Block Sparse Linear Layer - Dynamic topology block-sparse neural network layer with learned sparsity patterns for catastrophic forgetting mitigation"

## Problem Statement

Neural networks suffer from **catastrophic forgetting**: when trained sequentially on multiple tasks, performance on earlier tasks degrades severely as new learning overwrites previously encoded knowledge. Current mitigations have significant limitations:

- **Elastic Weight Consolidation (EWC)**: Importance estimates are static and computed on old tasks
- **Progressive Networks**: Parameter count grows linearly with each new task
- **Replay Buffers**: Requires storing and replaying old training data
- **PackNet**: Fixed capacity allocation per task, no adaptation

Additionally, standard sparse neural network implementations provide no actual computational speedup—they use dense storage with binary masks, meaning memory and compute costs remain unchanged despite "sparsity."

**Our hypothesis**: Dynamic network topology enables knowledge preservation by routing around established pathways rather than overwriting them, while structured block sparsity provides real hardware acceleration.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Model on Sequential Tasks Without Forgetting (Priority: P1)

A machine learning researcher wants to train a TitanMAC model on Task A (mathematics problems), then continue training on Task B (natural language), and have the model retain strong performance on Task A without storing Task A data.

**Why this priority**: This is the core value proposition—enabling continual learning without catastrophic forgetting. If this doesn't work, nothing else matters.

**Independent Test**: Can be fully tested by training on a math dataset, measuring accuracy, training on NLP data, then re-evaluating math accuracy. Delivers the fundamental anti-forgetting capability.

**Acceptance Scenarios**:

1. **Given** a model trained to convergence on Task A with 85% validation accuracy, **When** the model is further trained on Task B for an equivalent number of steps, **Then** Task A validation accuracy remains above 70% (less than 20% relative degradation)

2. **Given** a model with block-sparse layers at 50% density, **When** trained sequentially on two distinct tasks, **Then** the forgetting rate is at least 30% lower than an equivalent dense model

3. **Given** a trained model, **When** analyzing which network pathways (blocks) are active for Task A vs Task B, **Then** there is measurable pathway separation (overlap less than 70%)

---

### User Story 2 - Achieve Real Computational Speedup (Priority: P2)

A researcher running large-scale experiments wants the sparse model to train faster than the dense equivalent, reducing GPU hours and enabling more experiments within compute budget.

**Why this priority**: Without actual speedup, the complexity isn't justified. This is what makes block-level sparsity valuable over element-level approaches.

**Independent Test**: Can be tested by benchmarking forward/backward pass times on sparse vs dense layers with identical input shapes. Delivers concrete resource savings.

**Acceptance Scenarios**:

1. **Given** a block-sparse layer at 50% density, **When** compared to an equivalent dense layer, **Then** the forward pass completes in less than 70% of the dense time (at least 1.4x speedup)

2. **Given** a full training run with block-sparse layers, **When** measuring end-to-end training time, **Then** total training time is at least 20% faster than dense baseline

3. **Given** a block-sparse model, **When** measuring peak GPU memory usage, **Then** memory consumption is reduced proportionally to density (50% density = ~50% memory reduction for weight storage)

---

### User Story 3 - Configure Sparsity for Different Use Cases (Priority: P3)

A practitioner wants to choose which layers use sparsity and at what density level to balance forgetting mitigation, speedup, and model quality for their specific use case.

**Why this priority**: Flexibility enables experimentation and finding optimal configurations. Not required for initial validation but essential for practical adoption.

**Independent Test**: Can be tested by configuring different layers with different density settings and verifying they behave correctly. Delivers customization capability.

**Acceptance Scenarios**:

1. **Given** a model configuration specifying sparse layers 8-12 at 50% density, **When** the model is created, **Then** only those layers use block-sparse computation while others remain dense

2. **Given** a user specifying 25% density, **When** the model trains, **Then** each sparse layer maintains approximately 25% of possible connections (within 5% tolerance)

3. **Given** multiple density configurations (25%, 50%, 75%), **When** running the same task, **Then** higher density achieves better final accuracy while lower density achieves better speedup

---

### User Story 4 - Monitor Topology Evolution (Priority: P4)

A researcher wants to understand how network structure evolves during training to debug issues, validate the routing hypothesis, and gain insights into what the model learned.

**Why this priority**: Observability is crucial for research and debugging. The topology dynamics are what make this approach unique, so visibility into them has scientific value.

**Independent Test**: Can be tested by running training and checking that topology metrics are logged. Delivers transparency into model behavior.

**Acceptance Scenarios**:

1. **Given** a training run with block-sparse layers, **When** topology decisions occur (every ~100 steps), **Then** metrics are logged showing: blocks swapped, average block age, column usage distribution

2. **Given** topology logs over a full training run, **When** analyzed, **Then** the swap rate is between 1-10% per decision (not 0% stagnation, not 50% instability)

3. **Given** a model trained on two tasks sequentially, **When** comparing topology snapshots, **Then** it's possible to identify which blocks were added for Task B vs preserved from Task A

---

### Edge Cases

- What happens when topology decisions try to swap more blocks than available candidates?
- How does the system handle extreme sparsity (e.g., 10% density) where very few blocks exist?
- What happens if all blocks in a row score equally, making selection arbitrary?
- How does the system recover if topology changes cause a sudden loss spike?
- What happens when running distributed training (multiple GPUs) and topology decisions must be synchronized?

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a drop-in replacement for standard linear layers that stores weights in block-sparse format
- **FR-002**: System MUST support configurable block/tile sizes (default 16x16) that match GPU cache line boundaries
- **FR-003**: System MUST support configurable density levels from 10% to 100%
- **FR-004**: System MUST accumulate per-block importance scores during training using gradient statistics
- **FR-005**: System MUST make topology decisions (prune/grow blocks) at configurable intervals (default every 100 training steps)
- **FR-006**: System MUST initialize new blocks with appropriate weight scaling to avoid training instability
- **FR-007**: System MUST synchronize topology decisions across distributed training workers to maintain model consistency
- **FR-008**: System MUST log topology metrics (swaps, ages, column usage) for observability
- **FR-009**: System MUST provide a baseline magnitude-based scoring heuristic for topology decisions
- **FR-010**: System MUST support exploration in topology search (epsilon-greedy or equivalent mechanism)
- **FR-011**: System MUST integrate with the existing CMS (Continuum Memory System) multi-timescale update framework
- **FR-012**: System MUST maintain training stability when topology changes occur (no catastrophic loss spikes)

### Key Entities

- **Block**: A tile of weights (default 16x16 = 256 parameters) that is the atomic unit of sparsity. Blocks are either active (part of the computation) or inactive (pruned).

- **Topology**: The pattern of active blocks in a layer, represented as indices mapping output rows to input columns. Topology evolves during training.

- **Block Score**: An importance metric for each block, accumulated from gradient statistics over multiple training steps. Higher scores indicate blocks that contribute more to learning.

- **Topology Decision**: A periodic event where the system evaluates block scores, selects which blocks to prune (low scores) and which candidates to grow (high potential), and updates the topology.

- **CMS Level**: The multi-timescale framework where Level 0 = every step (weight updates), Level 1 = every ~10 steps (score accumulation), Level 2 = every ~100 steps (topology decisions).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Models trained with block-sparse layers exhibit less than 30% relative forgetting when trained sequentially on two distinct tasks (vs 40-60% baseline for dense models)

- **SC-002**: Block-sparse layers at 50% density complete forward passes at least 1.3x faster than equivalent dense layers on target hardware

- **SC-003**: End-to-end training with block-sparse layers achieves at least 1.2x speedup compared to dense baseline

- **SC-004**: Training remains stable with no loss spikes greater than 2x the running average at topology decision boundaries

- **SC-005**: Topology metrics show healthy dynamics: 1-10% block swap rate per decision, diverse column usage (entropy > 50% of maximum)

- **SC-006**: Model quality at convergence is within 5% of dense baseline accuracy for any given task

- **SC-007**: Peak GPU memory usage for weight storage scales proportionally with density (50% density = ~50% weight memory)

- **SC-008**: Distributed training maintains identical topology across all workers (verified via checksum comparison)

---

## Assumptions

1. **Hardware**: Target deployment is NVIDIA GPUs with Tensor Cores (Volta architecture or newer) where 16x16 tile operations are hardware-accelerated

2. **Model Scale**: Initial validation will be on TitanMAC at 237M parameters; the approach should scale to larger models but this is not guaranteed

3. **Task Similarity**: The sequential tasks used for forgetting evaluation will be sufficiently different that naive training would cause significant forgetting (>30%)

4. **CMS Availability**: The existing CMS (Continuum Memory System) infrastructure in DeepNestedOptimizer is functional and can be extended

5. **Baseline Metrics**: Dense model forgetting rates and training times will be measured as baselines before sparse implementation begins

6. **Block Size Trade-off**: 16x16 blocks are a reasonable default balancing hardware efficiency and representational granularity; smaller blocks (8x8) may be explored if results indicate granularity issues

---

## Scope Boundaries

### In Scope

- Block-sparse linear layers for MLP components
- Magnitude-based topology scoring heuristics
- Integration with CMS multi-timescale updates
- Single-GPU and multi-GPU (DDP) training support
- Configurable density and block size
- Topology logging and metrics

### Out of Scope (Future Work)

- Sparse attention mechanisms (focus is on MLP layers first)
- Sparse neural memory (memory_mlp remains dense in v1)
- Learned topology controller (start with heuristics, add learning if heuristics plateau)
- Inference-time topology adaptation (topology is fixed after training)
- Automatic density selection (user must specify density)

---

## Dependencies

1. **Existing CMS Framework**: Requires functional CMS in DeepNestedOptimizer for multi-timescale scheduling
2. **Triton or Equivalent**: Requires GPU kernel framework for block-sparse matrix operations
3. **Baseline Measurements**: Phase 0 baseline experiments must complete before sparse experiments begin
4. **Test Tasks**: Need two sufficiently different tasks (math and NLP) with established training protocols

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Block-sparse kernels don't achieve expected speedup | Medium | High | Start with simple PyTorch reference, optimize only if correctness verified |
| Topology dynamics don't reduce forgetting | Medium | High | This is falsifiable; if true, document and pivot to alternative approaches |
| Training instability at topology boundaries | Medium | Medium | Conservative swap thresholds (1.5x improvement required), gradual weight initialization |
| DDP synchronization adds significant overhead | Low | Medium | Sync only at topology decisions (every ~100 steps), not every step |
| 16x16 block size too coarse for fine-grained knowledge | Medium | Medium | Ablate with 8x8 and 32x32 to characterize trade-off |
