# Tasks: CMS Dynamic Block Sparse Linear Layer

**Input**: Design documents from `/specs/001-cms-block-sparse/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cms_block_linear.py, quickstart.md
**Branch**: `001-cms-block-sparse`
**Date**: 2025-12-25

---

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- All file paths are relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project structure and foundational modules

- [x] T001 Create `titans_core/layers/__init__.py` with exports for CMSBlockLinear
- [x] T002 [P] Create `titans_core/layers/block_ell.py` with Block-ELL format dataclass and utility stubs
- [x] T003 [P] Create `titans_core/layers/block_sparse.py` with CMSBlockLinear class skeleton matching contract
- [x] T004 [P] Create `titans_core/opt/topology_scorer.py` with TopologyScorer class skeleton
- [x] T005 [P] Create `titans_core/kernels/block_ell_forward.py` with forward kernel stub
- [x] T006 [P] Create `titans_core/kernels/block_ell_backward.py` with backward kernel stubs

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can begin

**CRITICAL**: No user story work can proceed until this phase is complete

### Block-ELL Format Utilities

- [x] T007 Implement `BlockELLFormat` dataclass in `titans_core/layers/block_ell.py` with: R, C, K, B, values[R,K,B,B], col_indices[R,K], validate() method
- [x] T008 Implement `create_random_topology(R, C, K, generator)` in `titans_core/layers/block_ell.py` - creates initial random column indices ensuring uniqueness per row
- [x] T009 [P] Implement `to_dense(values, col_indices, R, C, K, B)` in `titans_core/layers/block_ell.py` - converts block-ELL to dense matrix for testing
- [x] T010 [P] Implement `from_dense(dense, tile_size, density)` in `titans_core/layers/block_ell.py` - magnitude-based pruning to create initial topology

### Block-ELL Tests

- [x] T011 Create `tests/unit/test_block_ell_format.py` with test for BlockELLFormat validation rules
- [x] T012 [P] Add test `test_random_topology_uniqueness` - verify all col_indices[r] values unique per row
- [x] T013 [P] Add test `test_to_dense_from_dense_roundtrip` - verify dense conversion and back preserves values
- [x] T014 [P] Add test `test_from_dense_respects_density` - verify correct number of blocks per row

### CMSBlockLinear Core Implementation

- [x] T015 Implement `CMSBlockLinear.__init__()` in `titans_core/layers/block_sparse.py` - dimension validation, Block-ELL storage creation, Kaiming initialization, register buffers for scoring state
- [x] T016 Implement `CMSBlockLinear.forward()` PyTorch reference in `titans_core/layers/block_sparse.py` - gather blocks from input, matmul with tiles, scatter-add to output (no Triton yet)
- [x] T017 Implement `CMSBlockLinear.to_dense()` in `titans_core/layers/block_sparse.py` - convert current topology to dense weight matrix using block_ell.to_dense()
- [x] T018 Implement `CMSBlockLinear.get_density()` in `titans_core/layers/block_sparse.py` - return K/C

### CMSBlockLinear Core Tests

- [x] T019 Create `tests/unit/test_cms_block_linear.py` with test for dimension validation (ValueError on non-divisible)
- [x] T020 [P] Add test `test_forward_shape_2d` - verify [batch, in_features] -> [batch, out_features]
- [x] T021 [P] Add test `test_forward_shape_3d` - verify [batch, seq, in_features] -> [batch, seq, out_features]
- [x] T022 Add test `test_forward_matches_dense_at_full_density` - at density=1.0, output matches nn.Linear

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Anti-Forgetting (Priority: P1)

**Goal**: Train on sequential tasks without catastrophic forgetting through dynamic topology routing

**Independent Test**: Train on math dataset, measure accuracy, train on NLP, re-evaluate math accuracy. Forgetting should be <30% relative.

### Topology Scoring Implementation [US1]

- [x] T023 [US1] Implement `TopologyScorer.__init__()` in `titans_core/opt/topology_scorer.py` - initialize EMA buffers (gradient_ema, activation_norm_acc, error_norm_acc, block_age)
- [x] T024 [US1] Implement `TopologyScorer.update_gradient_ema()` in `titans_core/opt/topology_scorer.py` - compute per-block Frobenius norm, update EMA with alpha=0.95
- [x] T025 [US1] Implement `compute_gradient_frobenius_norms()` in `titans_core/opt/topology_scorer.py` - compute per-block gradient norms
- [x] T026 [US1] Implement `TopologyScorer.compute_candidate_scores()` in `titans_core/opt/topology_scorer.py` - compute candidate scores as error_norm[r] * activation_norm[c] (outer product)
- [x] T027 [US1] Implement `TopologyScorer.select_top_k()` in `titans_core/opt/topology_scorer.py` - epsilon-greedy selection, return new indices and swap list
- [x] T028 [US1] Implement `TopologyScorer.compute_column_entropy()`, `should_swap()`, `initialize_scores()` in `titans_core/opt/topology_scorer.py`

### CMSBlockLinear Scoring Integration [US1]

- [x] T029 [US1] Add forward hook to `CMSBlockLinear` for input activation capture in `titans_core/layers/block_sparse.py`
- [x] T030 [US1] Add backward hook to `CMSBlockLinear` for output gradient capture in `titans_core/layers/block_sparse.py`
- [x] T031 [US1] Implement `CMSBlockLinear.accumulate_scores()` in `titans_core/layers/block_sparse.py` - delegate to TopologyScorer methods
- [x] T032 [US1] Implement `CMSBlockLinear.score_step()` in `titans_core/layers/block_sparse.py` - normalize accumulators, increment ages, reset step counter

### Topology Step Implementation [US1]

- [x] T033 [US1] Implement `CMSBlockLinear.topology_step(generator)` in `titans_core/layers/block_sparse.py` - call scorer.select_topology(), update col_indices, initialize new blocks with Kaiming * 0.1, reset ages for new blocks
- [x] T034 [US1] Implement new block weight initialization in `topology_step()` - Kaiming uniform scaled by 0.1 to avoid instability
- [x] T035 [US1] Implement `TopologyDecisionResult` population in `topology_step()` - capture num_swaps, swap_rate, pruned_positions, grown_columns

### Topology Scoring Tests [US1]

- [x] T036 [US1] Create `tests/unit/test_topology_scorer.py` with test for gradient EMA decay (verify alpha=0.95 behavior)
- [x] T037 [P] [US1] Add test `test_candidate_scoring` - verify candidates scored by error*activation product
- [x] T038 [P] [US1] Add test `test_epsilon_greedy_exploration` - with epsilon=1.0 all swaps should be random
- [x] T039 [US1] Add test `test_swap_threshold` - verify candidates must be 1.5x better to swap
- [x] T040 [US1] Add test `test_topology_maintains_density` - after topology_step, K blocks per row unchanged

### Integration Tests [US1]

- [x] T041 [US1] Create `tests/integration/test_cms_topology_integration.py` with test for full training loop: forward, backward, accumulate, score_step, topology_step
- [x] T042 [US1] Add test `test_pathway_separation` - train on two synthetic tasks, verify block overlap < 70%
- [x] T043 [US1] Add test `test_no_loss_spike_at_topology_step` - verify loss increase < 2x running average after swap

**Checkpoint**: Anti-forgetting capability complete - can train sequential tasks with topology routing

---

## Phase 4: User Story 2 - Computational Speedup (Priority: P2)

**Goal**: Achieve real forward/backward speedup through Triton block-sparse kernels

**Independent Test**: Benchmark sparse vs dense forward/backward on same input shapes. Sparse at 50% density should be at least 1.3x faster.

### Triton Forward Kernel [US2]

- [x] T044 [US2] Implement `block_ell_forward_kernel` in `titans_core/kernels/block_ell_forward.py` - Grid(batch, R), gather input blocks, matmul with tiles, accumulate output
- [x] T045 [US2] Implement `BlockELLForward.forward()` autograd function in `titans_core/kernels/block_ell_forward.py` - wrap kernel, handle 2D/3D input reshape
- [x] T046 [US2] Add tile size parameter support (16, 32) in `block_ell_forward_kernel` - template for BLOCK_M, BLOCK_K

### Triton Backward Kernels [US2]

- [x] T047 [US2] Implement `block_ell_grad_values_kernel` in `titans_core/kernels/block_ell_backward.py` - Grid(R, K), outer product of grad_output and input blocks
- [x] T048 [US2] Implement `block_ell_grad_input_kernel` in `titans_core/kernels/block_ell_backward.py` - Grid(batch, R), scatter-add via tl.atomic_add (NOTE: uses reference impl due to Triton atomic limitations)
- [x] T049 [US2] Implement `BlockELLForward.backward()` in `titans_core/kernels/block_ell_forward.py` - call both backward kernels, return gradients

### Kernel Integration [US2]

- [x] T050 [US2] Add `_use_triton_kernel` property to `CMSBlockLinear` - detect CUDA availability and tile size compatibility
- [x] T051 [US2] Modify `CMSBlockLinear.forward()` to dispatch to Triton kernel when available in `titans_core/layers/block_sparse.py`
- [x] T052 [US2] Add fallback to PyTorch reference when Triton unavailable (CPU, unsupported tile size)

### Performance Tests [US2]

- [x] T053 [US2] Create `tests/unit/test_block_ell_kernels.py` with test `test_forward_correctness` - Triton matches PyTorch reference within 2e-2 (FP32 accumulation tolerance)
- [x] T054 [P] [US2] Add test `test_backward_grad_values_correctness` - gradient w.r.t. values matches reference
- [x] T055 [P] [US2] Add test `test_backward_grad_input_correctness` - gradient w.r.t. input matches reference
- [x] T056 [US2] Create `benchmarks/bench_block_ell.py` with forward/backward timing comparison (dense vs sparse at 25%, 50%, 75% density)
- [x] T057 [US2] Add memory usage benchmark in `benchmarks/bench_block_ell.py` - measure peak GPU memory during forward/backward

**Checkpoint**: Computational speedup achieved - sparse layers faster than dense at target density

---

## Phase 5: User Story 3 - Configuration (Priority: P3)

**Goal**: Allow users to configure which layers use sparsity and at what density

**Independent Test**: Configure model with different density settings per layer, verify each behaves correctly.

### TitanMACConfig Extensions [US3]

- [x] T058 [US3] Add block sparse config fields to `TitanMACConfig` in `titans_core/config.py`: use_block_sparse, block_sparse_tile_size, block_sparse_density, block_sparse_layers, block_sparse_components
- [x] T059 [US3] Add config validation in `TitanMACConfig.__post_init__()` - density in [0.1, 1.0], tile_size in {8, 16, 32}, layers tuple valid

### MLP Layer Integration [US3]

- [x] T060 [US3] Modify `TitanMLP.__init__()` in `titans_core/blocks/mlp.py` to accept `use_block_sparse` flag
- [x] T061 [US3] Modify `TitanMLP.__init__()` to create `CMSBlockLinear` for fc1/fc2 when flag enabled
- [x] T062 [US3] Add `block_sparse_layers` property to `TitanMAC` model in `titans_core/models/titanmac.py` - returns list of all CMSBlockLinear layers

### Layer Selection Logic [US3]

- [x] T063 [US3] Implement layer selection in model creation - only layers in `block_sparse_layers` tuple use CMSBlockLinear
- [x] T064 [US3] Implement component selection - `block_sparse_components=('mlp',)` applies to MLP only
- [x] T065 [US3] Add per-layer density override support - `block_sparse_density` can be dict mapping layer index to density

### Configuration Tests [US3]

- [x] T066 [US3] Add test `test_config_validation` in `tests/unit/test_config.py` - verify invalid density/tile_size raises ValueError
- [x] T067 [P] [US3] Add test `test_selective_layer_sparsity` - verify only specified layers use CMSBlockLinear
- [x] T068 [P] [US3] Add test `test_per_layer_density` - verify different densities per layer when configured

**Checkpoint**: Configuration complete - users can customize sparsity per layer/component

---

## Phase 6: User Story 4 - Monitoring (Priority: P4)

**Goal**: Enable visibility into topology evolution for debugging and research

**Independent Test**: Run training, verify topology metrics are logged at decision steps.

### Topology Statistics [US4]

- [ ] T069 [US4] Implement `CMSBlockLinear.get_topology_stats()` in `titans_core/layers/block_sparse.py` - return TopologyStats(density, avg_block_score, avg_block_age, column_entropy, num_blocks)
- [ ] T070 [US4] Implement column entropy calculation in `get_topology_stats()` - entropy of column usage distribution normalized to [0, 1]
- [ ] T071 [US4] Add `get_block_age_distribution()` method - return histogram of block ages for analysis

### Logging Integration [US4]

- [ ] T072 [US4] Add topology logging callback to `DeepNestedOptimizer.step()` in `titans_core/opt/deep_nested_optimizer.py` - log at Level 2 steps
- [ ] T073 [US4] Implement `_log_topology_metrics()` in DeepNestedOptimizer - iterate block_sparse_layers, log stats to wandb/tensorboard if available
- [ ] T074 [US4] Add swap rate tracking over time - maintain rolling average of swap rates for stability monitoring

### Checkpointing [US4]

- [ ] T075 [US4] Implement `CMSBlockLinear.state_dict()` in `titans_core/layers/block_sparse.py` - include values, col_indices, bias, all scoring state
- [ ] T076 [US4] Implement `CMSBlockLinear.load_state_dict()` in `titans_core/layers/block_sparse.py` - restore full layer state including topology
- [ ] T077 [US4] Add topology snapshot saving in checkpoints - save before/after col_indices at each decision

### Monitoring Tests [US4]

- [ ] T078 [US4] Add test `test_topology_stats_values` in `tests/unit/test_cms_block_linear.py` - verify stats return correct types and ranges
- [ ] T079 [P] [US4] Add test `test_column_entropy_bounds` - verify entropy in [0, 1] range
- [ ] T080 [P] [US4] Add test `test_state_dict_roundtrip` - save and load state, verify identical behavior
- [ ] T081 [US4] Add test `test_swap_rate_stability` - verify swap rate between 1-10% per decision in normal training

**Checkpoint**: Monitoring complete - full visibility into topology dynamics

---

## Phase 7: DDP Synchronization (Cross-Cutting)

**Purpose**: Ensure distributed training maintains consistent topology across ranks

- [ ] T082 Add `sync_topology_scores()` method to `CMSBlockLinear` - all-reduce block_score_ema, activation_norm_acc, error_norm_acc
- [ ] T083 Modify `topology_step()` to call sync before selection when DDP detected
- [ ] T084 Add deterministic RNG in `topology_step()` - use generator seeded by global_step for identical decisions across ranks
- [ ] T085 Create `tests/integration/test_ddp_topology_sync.py` with test `test_topology_identical_across_ranks` - spawn 2 processes, verify col_indices match
- [ ] T086 [P] Add test `test_ddp_score_averaging` - verify all-reduce correctly averages scores
- [ ] T087 Add topology checksum logging for divergence detection

---

## Phase 8: DeepNestedOptimizer Integration

**Purpose**: Integrate topology lifecycle with existing CMS framework

- [ ] T088 Add `discover_block_sparse_layers()` to `DeepNestedOptimizer` in `titans_core/opt/deep_nested_optimizer.py` - scan model for CMSBlockLinear instances
- [ ] T089 Modify `DeepNestedOptimizer.__init__()` to call discover and store layer references
- [ ] T090 Add score accumulation call in `DeepNestedOptimizer.step()` - call layer.accumulate_scores() after backward
- [ ] T091 Integrate Level 1 score_step - call layer.score_step() at CMS Level 1 frequency (every ~10 steps)
- [ ] T092 Integrate Level 2 topology_step - call layer.topology_step() at CMS Level 2 frequency (every ~100 steps)
- [ ] T093 Add deterministic generator creation in topology_step loop - same seed from global_step for DDP
- [ ] T094 Extend `tests/unit/test_deep_nested_optimizer.py` with test for block sparse layer discovery
- [ ] T095 [P] Add test `test_optimizer_triggers_topology_step` - verify topology_step called at Level 2

---

## Phase 9: Polish & Validation

**Purpose**: Final integration, documentation, and validation

### End-to-End Validation

- [ ] T096 Create `examples/train_block_sparse.py` - minimal training script demonstrating block sparse usage
- [ ] T097 Run forgetting benchmark: train on math, train on NLP, measure math accuracy retention
- [ ] T098 Run speedup benchmark: compare training throughput (samples/sec) sparse vs dense
- [ ] T099 Validate stability: run 10K steps, verify no loss spikes > 2x running average

### Conversion Utilities

- [ ] T100 Implement `CMSBlockLinear.from_dense()` class method - create sparse layer from existing nn.Linear weights
- [ ] T101 Add test `test_from_dense_preserves_important_weights` - verify highest magnitude blocks retained

### Documentation

- [ ] T102 Update `quickstart.md` with any API changes discovered during implementation
- [ ] T103 Add troubleshooting section for common issues (dimension errors, no speedup, instability)
- [ ] T104 Update `CLAUDE.md` with block sparse training patterns if needed

### Cleanup

- [ ] T105 Run `black titans_core/ --line-length 100` formatting
- [ ] T106 Run `ruff check titans_core/` and fix any linting issues
- [ ] T107 Verify all tests pass: `pytest tests/ -v`
- [ ] T108 Remove any debug print statements or temporary code

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ────────────────────────────────────────────────────────────────
    │
    ▼
Phase 2 (Foundational) ─────────────────────── BLOCKS ALL USER STORIES
    │
    ├───────────────┬───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
Phase 3 (US1)   Phase 4 (US2)   Phase 5 (US3)   Phase 6 (US4)
Anti-Forgetting Speedup         Config          Monitoring
    │               │               │               │
    └───────────────┴───────────────┴───────────────┘
                            │
                            ▼
                    Phase 7 (DDP Sync) ← Requires T033, T051
                            │
                            ▼
                    Phase 8 (Optimizer) ← Requires T033, T082
                            │
                            ▼
                    Phase 9 (Polish)
```

### Critical Task Dependencies

| Task | Depends On | Why |
|------|-----------|-----|
| T015 (CMSBlockLinear init) | T007, T008 | Needs Block-ELL format |
| T016 (forward) | T015 | Needs initialized layer |
| T022 (forward test) | T016, T009 | Needs forward and to_dense |
| T031 (accumulate_scores) | T023-T26 | Needs TopologyScorer |
| T033 (topology_step) | T27, T28, T32 | Needs scoring and selection |
| T051 (Triton dispatch) | T044-T49 | Needs kernels |
| T060 (MLP integration) | T015 | Needs CMSBlockLinear |
| T088 (optimizer discover) | T062 | Needs block_sparse_layers property |
| T092 (Level 2 trigger) | T033 | Needs topology_step |

### User Story Independence

Each user story can be tested independently after Phase 2:

| Story | Independent Test Criteria | Required Phases |
|-------|--------------------------|-----------------|
| US1 | Train sequential tasks, measure forgetting | 1, 2, 3 |
| US2 | Benchmark forward/backward speedup | 1, 2, 4 |
| US3 | Configure different densities, verify | 1, 2, 5 |
| US4 | Run training, check logs for topology metrics | 1, 2, 6 |

---

## Parallel Execution Examples

### Phase 1 Parallel Launch
```bash
# All setup tasks can run in parallel (different files):
Task T002: "Create titans_core/layers/block_ell.py"
Task T003: "Create titans_core/layers/block_sparse.py"
Task T004: "Create titans_core/opt/topology_scorer.py"
Task T005: "Create titans_core/kernels/block_ell_forward.py"
Task T006: "Create titans_core/kernels/block_ell_backward.py"
```

### Phase 2 Block-ELL Tests Parallel Launch
```bash
# After T007-T10 complete, tests can run in parallel:
Task T012: "test_random_topology_uniqueness"
Task T013: "test_to_dense_from_dense_roundtrip"
Task T014: "test_from_dense_respects_density"
```

### Phase 3 + Phase 4 Parallel (After Phase 2)
```bash
# User Story 1 and User Story 2 can proceed in parallel:
Developer A: T023-T043 (Anti-Forgetting topology scoring)
Developer B: T044-T057 (Triton kernel speedup)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T022)
3. Complete Phase 3: User Story 1 Anti-Forgetting (T023-T043)
4. **STOP and VALIDATE**: Test with sequential task training
5. Measure forgetting rate - target <30% relative

### Incremental Delivery

1. **MVP**: Setup + Foundational + US1 = Working topology with PyTorch reference
2. **+Speedup**: Add Phase 4 (US2) = Triton kernels for performance
3. **+Config**: Add Phase 5 (US3) = User-configurable sparsity
4. **+Monitoring**: Add Phase 6 (US4) = Full observability
5. **+DDP**: Add Phase 7 = Multi-GPU support
6. **+Integration**: Add Phase 8 = Seamless optimizer integration
7. **+Polish**: Add Phase 9 = Production ready

### Risk Mitigation

- **PyTorch reference first** (T016): Correctness before optimization
- **Tests before implementation** (T011-T014): Catch bugs early
- **Independent user stories**: Can ship partial features
- **Swap threshold (1.5x)**: Prevents unstable topology oscillation
- **DDP sync at Level 2 only**: Minimal overhead for distributed

---

## Notes

- [P] tasks = different files, no dependencies between them
- [US#] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Run `pytest tests/ -v` after completing each phase
