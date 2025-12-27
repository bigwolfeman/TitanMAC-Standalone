"""Unit tests for TopologyScorer (T023-T028).

Tests the core topology scoring functionality for CMS block-sparse layers.

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import pytest
import torch

from titans_core.opt.topology_scorer import (
    TopologyScorer,
    compute_gradient_frobenius_norms,
    initialize_scores,
)


class TestTopologyScorerInit:
    """T023: Test TopologyScorer initialization."""

    def test_basic_init(self):
        """Test basic initialization with valid parameters."""
        scorer = TopologyScorer(R=160, C=40, K=20)
        assert scorer.R == 160
        assert scorer.C == 40
        assert scorer.K == 20
        assert scorer.ema_alpha == 0.95
        assert scorer.exploration_epsilon == 0.05
        assert scorer.swap_threshold == 1.5

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        scorer = TopologyScorer(
            R=100, C=50, K=25, ema_alpha=0.9, exploration_epsilon=0.1, swap_threshold=2.0
        )
        assert scorer.ema_alpha == 0.9
        assert scorer.exploration_epsilon == 0.1
        assert scorer.swap_threshold == 2.0

    def test_validation_k_exceeds_c(self):
        """Test that K > C raises ValueError."""
        with pytest.raises(ValueError, match="K.*cannot exceed C"):
            TopologyScorer(R=10, C=5, K=10)

    def test_validation_ema_alpha_bounds(self):
        """Test ema_alpha bounds validation."""
        with pytest.raises(ValueError, match="ema_alpha"):
            TopologyScorer(R=10, C=10, K=5, ema_alpha=1.5)
        with pytest.raises(ValueError, match="ema_alpha"):
            TopologyScorer(R=10, C=10, K=5, ema_alpha=-0.1)

    def test_validation_epsilon_bounds(self):
        """Test exploration_epsilon bounds validation."""
        with pytest.raises(ValueError, match="exploration_epsilon"):
            TopologyScorer(R=10, C=10, K=5, exploration_epsilon=0.6)
        with pytest.raises(ValueError, match="exploration_epsilon"):
            TopologyScorer(R=10, C=10, K=5, exploration_epsilon=-0.1)


class TestUpdateGradientEMA:
    """T024: Test update_gradient_ema."""

    def test_first_update_from_zeros(self):
        """Test EMA update from zero initial state."""
        scorer = TopologyScorer(R=4, C=8, K=4, ema_alpha=0.95)
        grad_norms = torch.ones(4, 4)
        current_ema = torch.zeros(4, 4)

        new_ema = scorer.update_gradient_ema(grad_norms, current_ema)

        # ema = 0.95 * 1.0 + 0.05 * 0.0 = 0.95
        expected = torch.full((4, 4), 0.95)
        assert torch.allclose(new_ema, expected)

    def test_ema_converges(self):
        """Test that EMA converges toward constant input."""
        scorer = TopologyScorer(R=2, C=4, K=2, ema_alpha=0.5)
        current_ema = torch.zeros(2, 2)
        constant_input = torch.full((2, 2), 2.0)

        # Apply EMA updates multiple times
        for _ in range(10):
            current_ema = scorer.update_gradient_ema(constant_input, current_ema)

        # Should converge close to 2.0
        assert torch.allclose(current_ema, constant_input, atol=0.01)

    def test_ema_shape_preserved(self):
        """Test that output shape matches input."""
        scorer = TopologyScorer(R=16, C=32, K=8)
        grad_norms = torch.randn(16, 8).abs()
        current_ema = torch.randn(16, 8).abs()

        new_ema = scorer.update_gradient_ema(grad_norms, current_ema)

        assert new_ema.shape == (16, 8)


class TestComputeGradientFrobeniusNorms:
    """T025: Test compute_gradient_frobenius_norms."""

    def test_identity_matrix_norm(self):
        """Test Frobenius norm of identity matrix."""
        # Identity matrix has Frobenius norm = sqrt(n) for n x n
        B = 4
        grad = torch.zeros(1, 1, B, B)
        grad[0, 0] = torch.eye(B)

        norms = compute_gradient_frobenius_norms(grad)

        expected = torch.tensor([[2.0]])  # sqrt(4) = 2
        assert torch.allclose(norms, expected)

    def test_all_ones_norm(self):
        """Test Frobenius norm of all-ones matrix."""
        R, K, B = 2, 3, 4
        grad = torch.ones(R, K, B, B)

        norms = compute_gradient_frobenius_norms(grad)

        # ||ones(4,4)||_F = sqrt(16) = 4
        expected = torch.full((R, K), 4.0)
        assert torch.allclose(norms, expected)

    def test_shape_output(self):
        """Test output shape is [R, K]."""
        R, K, B = 10, 5, 16
        grad = torch.randn(R, K, B, B)

        norms = compute_gradient_frobenius_norms(grad)

        assert norms.shape == (R, K)

    def test_non_negative(self):
        """Test that norms are always non-negative."""
        grad = torch.randn(4, 8, 16, 16)
        norms = compute_gradient_frobenius_norms(grad)
        assert (norms >= 0).all()


class TestComputeCandidateScores:
    """T026: Test compute_candidate_scores."""

    def test_outer_product(self):
        """Test that output is outer product of norms."""
        scorer = TopologyScorer(R=4, C=8, K=4)
        activation_norms = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        error_norms = torch.tensor([0.5, 1.0, 1.5, 2.0])

        scores = scorer.compute_candidate_scores(activation_norms, error_norms)

        assert scores.shape == (4, 8)
        # Verify specific values: score[r,c] = error_norms[r] * activation_norms[c]
        assert torch.isclose(scores[0, 0], torch.tensor(0.5 * 1.0))
        assert torch.isclose(scores[1, 2], torch.tensor(1.0 * 3.0))
        assert torch.isclose(scores[3, 7], torch.tensor(2.0 * 8.0))

    def test_shape_output(self):
        """Test output shape is [R, C]."""
        scorer = TopologyScorer(R=10, C=20, K=5)
        activation_norms = torch.randn(20).abs()
        error_norms = torch.randn(10).abs()

        scores = scorer.compute_candidate_scores(activation_norms, error_norms)

        assert scores.shape == (10, 20)


class TestSelectTopK:
    """T027: Test select_top_k."""

    def test_no_swap_when_current_is_best(self):
        """Test that no swaps occur when current blocks have highest scores."""
        scorer = TopologyScorer(R=2, C=4, K=2, swap_threshold=1.5, exploration_epsilon=0.0)

        # Current scores are high
        current_scores = torch.tensor([[10.0, 9.0], [10.0, 9.0]])
        # Current columns are 0,1 for both rows
        col_indices = torch.tensor([[0, 1], [0, 1]])
        # Candidate scores for all positions (including inactive 2,3)
        candidate_scores = torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])

        new_cols, pruned, grown = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        # Should keep columns 0,1 since they're boosted by 1.5x and still beat candidates
        assert len(pruned) == 0
        assert len(grown) == 0

    def test_swap_when_candidate_is_much_better(self):
        """Test that swaps occur when candidate exceeds threshold."""
        scorer = TopologyScorer(R=2, C=4, K=2, swap_threshold=1.5, exploration_epsilon=0.0)

        # Current scores are low
        current_scores = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        col_indices = torch.tensor([[0, 1], [0, 1]])
        # Candidate at column 2 is much better (> 1.5 * 1.0 = 1.5)
        candidate_scores = torch.tensor([[0.5, 0.5, 10.0, 0.5], [0.5, 0.5, 0.5, 10.0]])

        new_cols, pruned, grown = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        # Column 2 should replace one of the current columns in row 0
        # Column 3 should replace one of the current columns in row 1
        assert 2 in new_cols[0].tolist()
        assert 3 in new_cols[1].tolist()
        assert len(pruned) >= 2
        assert len(grown) >= 2

    def test_deterministic_with_generator(self):
        """Test that results are deterministic with seeded generator."""
        scorer = TopologyScorer(R=4, C=8, K=2, exploration_epsilon=0.1)
        current_scores = torch.randn(4, 2).abs()
        candidate_scores = torch.randn(4, 8).abs()
        col_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])

        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        result1 = scorer.select_top_k(current_scores, candidate_scores, col_indices, gen1)
        result2 = scorer.select_top_k(current_scores, candidate_scores, col_indices, gen2)

        assert torch.equal(result1[0], result2[0])

    def test_output_shape(self):
        """Test that output col_indices has correct shape."""
        scorer = TopologyScorer(R=10, C=20, K=5, exploration_epsilon=0.0)
        current_scores = torch.randn(10, 5).abs()
        candidate_scores = torch.randn(10, 20).abs()
        col_indices = torch.arange(5).unsqueeze(0).expand(10, -1).clone()

        new_cols, _, _ = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        assert new_cols.shape == (10, 5)


class TestComputeColumnEntropy:
    """T028: Test compute_column_entropy."""

    def test_uniform_distribution_max_entropy(self):
        """Test that uniform column usage gives entropy close to 1."""
        scorer = TopologyScorer(R=4, C=4, K=1)
        # Each column used exactly once
        col_indices = torch.tensor([[0], [1], [2], [3]])

        entropy = scorer.compute_column_entropy(col_indices)

        assert 0.99 <= entropy <= 1.0

    def test_single_column_min_entropy(self):
        """Test that using only one column gives entropy 0."""
        scorer = TopologyScorer(R=4, C=4, K=1)
        # All rows use same column
        col_indices = torch.tensor([[0], [0], [0], [0]])

        entropy = scorer.compute_column_entropy(col_indices)

        assert entropy == 0.0

    def test_entropy_in_valid_range(self):
        """Test that entropy is always in [0, 1]."""
        scorer = TopologyScorer(R=10, C=20, K=5)
        col_indices = torch.randint(0, 20, (10, 5))

        entropy = scorer.compute_column_entropy(col_indices)

        assert 0.0 <= entropy <= 1.0


class TestShouldSwap:
    """T028b: Test should_swap."""

    def test_swap_when_candidate_exceeds_threshold(self):
        """Test swap when candidate > current * threshold."""
        scorer = TopologyScorer(R=4, C=8, K=2, swap_threshold=1.5)

        # 2.0 > 1.0 * 1.5 = 1.5, should swap
        assert scorer.should_swap(current_score=1.0, candidate_score=2.0, block_age=0)

    def test_no_swap_when_candidate_below_threshold(self):
        """Test no swap when candidate < current * threshold."""
        scorer = TopologyScorer(R=4, C=8, K=2, swap_threshold=1.5)

        # 1.4 < 1.0 * 1.5 = 1.5, should not swap
        assert not scorer.should_swap(current_score=1.0, candidate_score=1.4, block_age=0)

    def test_no_swap_at_threshold_boundary(self):
        """Test no swap when candidate == current * threshold (requires strictly greater)."""
        scorer = TopologyScorer(R=4, C=8, K=2, swap_threshold=1.5)

        # 1.5 is not > 1.0 * 1.5, should not swap
        assert not scorer.should_swap(current_score=1.0, candidate_score=1.5, block_age=0)


class TestInitializeScores:
    """T028c: Test initialize_scores."""

    def test_shapes(self):
        """Test that all tensors have correct shapes."""
        R, C, K = 10, 20, 5
        block_score_ema, activation_norm_acc, error_norm_acc, block_age = initialize_scores(
            R, C, K
        )

        assert block_score_ema.shape == (R, K)
        assert activation_norm_acc.shape == (C,)
        assert error_norm_acc.shape == (R,)
        assert block_age.shape == (R, K)

    def test_zeros_initialized(self):
        """Test that all tensors are initialized to zeros."""
        block_score_ema, activation_norm_acc, error_norm_acc, block_age = initialize_scores(
            4, 8, 2
        )

        assert (block_score_ema == 0).all()
        assert (activation_norm_acc == 0).all()
        assert (error_norm_acc == 0).all()
        assert (block_age == 0).all()

    def test_dtype_specification(self):
        """Test that dtype is respected for float tensors."""
        block_score_ema, activation_norm_acc, error_norm_acc, block_age = initialize_scores(
            4, 8, 2, dtype=torch.float16
        )

        assert block_score_ema.dtype == torch.float16
        assert activation_norm_acc.dtype == torch.float16
        assert error_norm_acc.dtype == torch.float16
        assert block_age.dtype == torch.int32  # Always int32

    def test_device_specification(self):
        """Test that device is respected."""
        device = torch.device("cpu")
        block_score_ema, activation_norm_acc, error_norm_acc, block_age = initialize_scores(
            4, 8, 2, device=device
        )

        assert block_score_ema.device == device
        assert activation_norm_acc.device == device
        assert error_norm_acc.device == device
        assert block_age.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDACompatibility:
    """Test that all operations work on CUDA tensors."""

    def test_update_gradient_ema_cuda(self):
        """Test EMA update on CUDA."""
        scorer = TopologyScorer(R=4, C=8, K=4)
        grad_norms = torch.randn(4, 4, device="cuda").abs()
        current_ema = torch.zeros(4, 4, device="cuda")

        new_ema = scorer.update_gradient_ema(grad_norms, current_ema)

        assert new_ema.device.type == "cuda"

    def test_compute_gradient_frobenius_norms_cuda(self):
        """Test Frobenius norms on CUDA."""
        grad = torch.randn(4, 8, 16, 16, device="cuda")
        norms = compute_gradient_frobenius_norms(grad)
        assert norms.device.type == "cuda"

    def test_select_top_k_cuda(self):
        """Test top-K selection on CUDA."""
        scorer = TopologyScorer(R=4, C=8, K=2)
        current_scores = torch.randn(4, 2, device="cuda").abs()
        candidate_scores = torch.randn(4, 8, device="cuda").abs()
        col_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device="cuda")

        new_cols, _, _ = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        assert new_cols.device.type == "cuda"

    def test_initialize_scores_cuda(self):
        """Test initialization on CUDA."""
        device = torch.device("cuda")
        tensors = initialize_scores(4, 8, 2, device=device)

        for t in tensors:
            assert t.device.type == "cuda"


class TestGradientEMADecayRate:
    """T036: Test gradient EMA decay rate (alpha=0.95 behavior)."""

    def test_gradient_ema_decay_rate(self):
        """Verify EMA decays with alpha=0.95.

        When current_ema = 1.0 and grad_norms = 0.0:
        new_ema = 0.95 * 0.0 + 0.05 * 1.0 = 0.05
        """
        scorer = TopologyScorer(R=4, C=8, K=2, ema_alpha=0.95)
        # Start with EMA = 1.0, apply zero gradients
        current_ema = torch.ones(4, 2)
        grad_norms = torch.zeros(4, 2)

        new_ema = scorer.update_gradient_ema(grad_norms, current_ema)

        # EMA formula: alpha * new + (1 - alpha) * old
        # = 0.95 * 0 + 0.05 * 1.0 = 0.05
        expected = torch.full((4, 2), 0.05)
        assert torch.allclose(new_ema, expected)

    def test_gradient_ema_decay_multiple_steps(self):
        """Verify EMA decays correctly over multiple steps."""
        scorer = TopologyScorer(R=2, C=4, K=2, ema_alpha=0.95)
        current_ema = torch.ones(2, 2)
        grad_norms = torch.zeros(2, 2)  # Zero gradients to observe decay

        # After multiple zero-gradient steps, EMA should decay toward 0
        for step in range(10):
            current_ema = scorer.update_gradient_ema(grad_norms, current_ema)

        # After 10 steps: (1 - 0.95)^10 = 0.05^10 is tiny
        # More precisely: EMA_n = old * (1-alpha)^n
        expected_value = 0.05 ** 10  # Approximately 0
        assert torch.allclose(current_ema, torch.full((2, 2), expected_value), atol=1e-10)

    def test_gradient_ema_with_constant_input(self):
        """Verify EMA converges to constant input value."""
        scorer = TopologyScorer(R=2, C=4, K=2, ema_alpha=0.95)
        current_ema = torch.zeros(2, 2)
        constant_grad = torch.full((2, 2), 5.0)

        # EMA toward 5.0 should converge
        for _ in range(100):
            current_ema = scorer.update_gradient_ema(constant_grad, current_ema)

        # Should be very close to 5.0 after 100 iterations
        assert torch.allclose(current_ema, constant_grad, atol=0.001)


class TestEpsilonGreedyFullRandom:
    """T038: Test epsilon-greedy exploration with epsilon=1.0 (fully random)."""

    def test_epsilon_greedy_all_random_causes_swaps(self):
        """With epsilon=1.0 (or high epsilon), exploration should cause some swaps."""
        # Note: TopologyScorer caps exploration_epsilon at 0.5, so we test with max allowed
        scorer = TopologyScorer(R=4, C=8, K=2, exploration_epsilon=0.5, swap_threshold=1.5)

        # Current blocks have very high scores - normally no swaps would occur
        current_scores = torch.full((4, 2), 100.0)
        col_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        # Candidate scores are very low - normally would never be selected
        candidate_scores = torch.full((4, 8), 0.1)

        # Run multiple times with different seeds and check if swaps occur
        swaps_occurred = False
        for seed in range(20):
            gen = torch.Generator().manual_seed(seed)
            new_cols, pruned, grown = scorer.select_top_k(
                current_scores, candidate_scores, col_indices, generator=gen
            )
            if len(pruned) > 0:
                swaps_occurred = True
                break

        # With epsilon=0.5, we expect exploration swaps to occur in at least some runs
        assert swaps_occurred, "Epsilon-greedy exploration should cause some swaps with epsilon=0.5"

    def test_zero_epsilon_no_random_swaps(self):
        """With epsilon=0.0, only score-based decisions should occur."""
        scorer = TopologyScorer(R=4, C=8, K=2, exploration_epsilon=0.0, swap_threshold=1.5)

        # Current blocks have high scores
        current_scores = torch.full((4, 2), 10.0)
        col_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        # Candidates are below threshold (10 * 1.5 = 15), so no swaps should occur
        candidate_scores = torch.full((4, 8), 5.0)

        new_cols, pruned, grown = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        # No swaps should occur since candidates don't beat threshold
        assert len(pruned) == 0
        assert len(grown) == 0
        assert torch.equal(new_cols, col_indices)


class TestSwapThresholdBehavior:
    """T039: Test swap threshold - candidates must be 1.5x better to swap."""

    def test_swap_threshold_exact_boundary(self):
        """Verify that exactly 1.5x is NOT enough to swap (requires strictly greater)."""
        scorer = TopologyScorer(R=4, C=8, K=2, swap_threshold=1.5, exploration_epsilon=0.0)

        current_score = 10.0
        candidate_score = 15.0  # Exactly 1.5x

        # Should NOT swap since we require strictly greater
        assert not scorer.should_swap(current_score, candidate_score, block_age=0)

    def test_swap_threshold_above_boundary(self):
        """Verify that just above 1.5x DOES cause swap."""
        scorer = TopologyScorer(R=4, C=8, K=2, swap_threshold=1.5, exploration_epsilon=0.0)

        current_score = 10.0
        candidate_score = 15.01  # Just above 1.5x

        # Should swap since candidate is strictly greater than threshold
        assert scorer.should_swap(current_score, candidate_score, block_age=0)

    def test_swap_threshold_in_select_top_k(self):
        """Verify threshold is applied correctly in select_top_k."""
        scorer = TopologyScorer(R=2, C=4, K=2, swap_threshold=1.5, exploration_epsilon=0.0)

        # Current scores are 10.0 per block
        current_scores = torch.full((2, 2), 10.0)
        col_indices = torch.tensor([[0, 1], [2, 3]])

        # Row 0: candidate at col 2 is 20.0 (> 10 * 1.5 = 15) -> should swap
        # Row 1: candidate at col 0 is 14.0 (< 10 * 1.5 = 15) -> should NOT swap
        candidate_scores = torch.tensor([
            [5.0, 5.0, 20.0, 5.0],  # Row 0: col 2 beats threshold
            [14.0, 5.0, 5.0, 5.0],  # Row 1: col 0 doesn't beat threshold
        ])

        new_cols, pruned, grown = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=None
        )

        # Row 0 should have col 2 in its selection
        assert 2 in new_cols[0].tolist(), "Row 0 should swap in column 2 (score 20 > 10*1.5)"

        # Row 1 should keep columns 2, 3 (no swap since 14 < 15)
        assert set(new_cols[1].tolist()) == {2, 3}, "Row 1 should not change (14 < 10*1.5)"


class TestTopologyMaintainsDensityMock:
    """T040: Test topology maintains density after topology_step.

    Note: CMSBlockLinear.topology_step() is not yet fully implemented.
    This test verifies the TopologyScorer.select_top_k maintains K blocks per row,
    which is the core invariant that topology_step() must preserve.
    """

    def test_select_top_k_maintains_k_per_row(self):
        """After select_top_k, each row still has exactly K unique blocks."""
        scorer = TopologyScorer(R=10, C=20, K=5, swap_threshold=1.5, exploration_epsilon=0.1)

        # Random initial state
        current_scores = torch.rand(10, 5)
        col_indices = torch.stack([
            torch.randperm(20)[:5] for _ in range(10)
        ]).to(torch.int32)
        candidate_scores = torch.rand(10, 20)

        gen = torch.Generator().manual_seed(12345)
        new_cols, _, _ = scorer.select_top_k(
            current_scores, candidate_scores, col_indices, generator=gen
        )

        # Verify each row has exactly K unique columns
        for r in range(10):
            cols = new_cols[r].tolist()
            assert len(cols) == 5, f"Row {r} should have exactly K=5 columns"
            assert len(set(cols)) == 5, f"Row {r} columns should all be unique"
            assert all(0 <= c < 20 for c in cols), f"Row {r} columns should be valid [0, C)"

    def test_multiple_topology_decisions_maintain_k(self):
        """Simulate multiple topology updates and verify K is maintained."""
        scorer = TopologyScorer(R=4, C=8, K=3, swap_threshold=1.5, exploration_epsilon=0.1)

        # Initial topology
        col_indices = torch.tensor([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 0],
            [1, 2, 3],
        ], dtype=torch.int32)

        gen = torch.Generator().manual_seed(42)

        # Simulate 10 topology updates
        for step in range(10):
            current_scores = torch.rand(4, 3)
            candidate_scores = torch.rand(4, 8) * 2  # Higher candidates to encourage swaps

            col_indices, _, _ = scorer.select_top_k(
                current_scores, candidate_scores, col_indices, generator=gen
            )

            # Verify invariant after each step
            for r in range(4):
                cols = col_indices[r].tolist()
                assert len(cols) == 3, f"Step {step}, Row {r}: should have K=3 columns"
                assert len(set(cols)) == 3, f"Step {step}, Row {r}: columns should be unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
