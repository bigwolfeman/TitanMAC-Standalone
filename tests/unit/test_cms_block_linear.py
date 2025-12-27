"""Tests for CMSBlockLinear block-sparse linear layer.

Tests cover:
- T019: Dimension validation (ValueError on non-divisible)
- T020: test_forward_shape_2d - verify [batch, in_features] -> [batch, out_features]
- T021: test_forward_shape_3d - verify [batch, seq, in_features] -> [batch, seq, out_features]
- T022: test_forward_matches_dense_at_full_density - at density=1.0, output matches nn.Linear

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import pytest
import torch
import torch.nn as nn

from titans_core.layers.block_sparse import CMSBlockLinear


class TestDimensionValidation:
    """T019: Test dimension validation raises ValueError on non-divisible dimensions."""

    def test_in_features_not_divisible_raises_error(self):
        """in_features not divisible by tile_size should raise ValueError."""
        with pytest.raises(ValueError, match="in_features .* must be divisible by tile_size"):
            CMSBlockLinear(
                in_features=65,  # Not divisible by 16
                out_features=64,
                tile_size=16,
                density=0.5,
            )

    def test_out_features_not_divisible_raises_error(self):
        """out_features not divisible by tile_size should raise ValueError."""
        with pytest.raises(ValueError, match="out_features .* must be divisible by tile_size"):
            CMSBlockLinear(
                in_features=64,
                out_features=65,  # Not divisible by 16
                tile_size=16,
                density=0.5,
            )

    def test_both_not_divisible_raises_error(self):
        """Both dimensions not divisible should raise ValueError (checks in_features first)."""
        with pytest.raises(ValueError, match="in_features .* must be divisible by tile_size"):
            CMSBlockLinear(
                in_features=33,
                out_features=33,
                tile_size=16,
                density=0.5,
            )

    def test_density_too_low_raises_error(self):
        """Density below 0.1 should raise ValueError."""
        with pytest.raises(ValueError, match="density .* must be in"):
            CMSBlockLinear(
                in_features=64,
                out_features=64,
                tile_size=16,
                density=0.05,
            )

    def test_density_above_one_raises_error(self):
        """Density above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="density .* must be in"):
            CMSBlockLinear(
                in_features=64,
                out_features=64,
                tile_size=16,
                density=1.5,
            )

    def test_valid_dimensions_succeed(self):
        """Valid dimensions should create layer without error."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.tile_size == 16
        assert layer.density == 0.5
        assert layer.R == 128 // 16  # 8
        assert layer.C == 64 // 16   # 4
        assert layer.K == max(1, int(4 * 0.5))  # 2


class TestForwardShape2D:
    """T020: Test forward with 2D input [batch, in_features] -> [batch, out_features]."""

    def test_basic_2d_forward(self):
        """Basic 2D forward should produce correct output shape."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(8, 64)  # [batch=8, in_features=64]
        y = layer(x)

        assert y.shape == (8, 128), f"Expected (8, 128), got {y.shape}"

    def test_2d_forward_single_sample(self):
        """2D forward with batch=1 should work."""
        layer = CMSBlockLinear(
            in_features=32,
            out_features=64,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(1, 32)
        y = layer(x)

        assert y.shape == (1, 64)

    def test_2d_forward_large_batch(self):
        """2D forward with large batch should work."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(256, 64)
        y = layer(x)

        assert y.shape == (256, 128)

    def test_2d_forward_preserves_dtype(self):
        """2D forward should preserve input dtype."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
            dtype=torch.float32,
        )
        x = torch.randn(8, 64, dtype=torch.float32)
        y = layer(x)

        assert y.dtype == torch.float32


class TestForwardShape3D:
    """T021: Test forward with 3D input [batch, seq, in_features] -> [batch, seq, out_features]."""

    def test_basic_3d_forward(self):
        """Basic 3D forward should produce correct output shape."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(8, 16, 64)  # [batch=8, seq=16, in_features=64]
        y = layer(x)

        assert y.shape == (8, 16, 128), f"Expected (8, 16, 128), got {y.shape}"

    def test_3d_forward_single_sequence(self):
        """3D forward with seq=1 should work."""
        layer = CMSBlockLinear(
            in_features=32,
            out_features=64,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(4, 1, 32)
        y = layer(x)

        assert y.shape == (4, 1, 64)

    def test_3d_forward_long_sequence(self):
        """3D forward with long sequence should work."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        x = torch.randn(4, 512, 64)
        y = layer(x)

        assert y.shape == (4, 512, 128)

    def test_3d_forward_preserves_dtype(self):
        """3D forward should preserve input dtype."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
            dtype=torch.float32,
        )
        x = torch.randn(8, 16, 64, dtype=torch.float32)
        y = layer(x)

        assert y.dtype == torch.float32


class TestForwardMatchesDenseAtFullDensity:
    """T022: Test that at density=1.0, CMSBlockLinear output matches nn.Linear."""

    def test_matches_dense_2d_input(self):
        """At full density, sparse layer should match dense for 2D input."""
        torch.manual_seed(42)

        in_features, out_features = 64, 128
        tile_size = 16

        # Create sparse layer at full density
        sparse_layer = CMSBlockLinear(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=1.0,
            bias=True,
        )

        # Create equivalent dense layer with same weights
        dense_layer = nn.Linear(in_features, out_features, bias=True)

        # Copy weights from sparse to dense using to_dense()
        with torch.no_grad():
            dense_weights = sparse_layer.to_dense()
            dense_layer.weight.copy_(dense_weights)
            dense_layer.bias.copy_(sparse_layer.bias)

        # Forward pass with same input
        x = torch.randn(8, in_features)
        sparse_output = sparse_layer(x)
        dense_output = dense_layer(x)

        # Should be very close (within float precision)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-4, atol=1e-5), \
            f"Max diff: {(sparse_output - dense_output).abs().max().item()}"

    def test_matches_dense_3d_input(self):
        """At full density, sparse layer should match dense for 3D input."""
        torch.manual_seed(123)

        in_features, out_features = 64, 128
        tile_size = 16

        # Create sparse layer at full density
        sparse_layer = CMSBlockLinear(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=1.0,
            bias=True,
        )

        # Create equivalent dense layer with same weights
        dense_layer = nn.Linear(in_features, out_features, bias=True)

        # Copy weights from sparse to dense
        with torch.no_grad():
            dense_weights = sparse_layer.to_dense()
            dense_layer.weight.copy_(dense_weights)
            dense_layer.bias.copy_(sparse_layer.bias)

        # Forward pass with same input
        x = torch.randn(4, 16, in_features)
        sparse_output = sparse_layer(x)
        dense_output = dense_layer(x)

        # Should be very close (within float precision)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-4, atol=1e-5), \
            f"Max diff: {(sparse_output - dense_output).abs().max().item()}"

    def test_matches_dense_no_bias(self):
        """At full density without bias, sparse layer should match dense."""
        torch.manual_seed(456)

        in_features, out_features = 32, 64
        tile_size = 16

        # Create sparse layer at full density without bias
        sparse_layer = CMSBlockLinear(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=1.0,
            bias=False,
        )

        # Create equivalent dense layer with same weights
        dense_layer = nn.Linear(in_features, out_features, bias=False)

        # Copy weights from sparse to dense
        with torch.no_grad():
            dense_weights = sparse_layer.to_dense()
            dense_layer.weight.copy_(dense_weights)

        # Forward pass with same input
        x = torch.randn(8, 32)
        sparse_output = sparse_layer(x)
        dense_output = dense_layer(x)

        # Should be very close (within float precision)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-4, atol=1e-5), \
            f"Max diff: {(sparse_output - dense_output).abs().max().item()}"

    def test_gradient_flow_matches_dense(self):
        """Gradients should flow correctly through sparse layer like dense."""
        torch.manual_seed(789)

        in_features, out_features = 64, 128
        tile_size = 16

        # Create sparse layer at full density
        sparse_layer = CMSBlockLinear(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=1.0,
            bias=True,
        )

        # Forward and backward
        x = torch.randn(4, in_features, requires_grad=True)
        y = sparse_layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert sparse_layer.values.grad is not None
        assert sparse_layer.bias.grad is not None

        # Gradient shapes should be correct
        assert x.grad.shape == x.shape
        assert sparse_layer.values.grad.shape == sparse_layer.values.shape


class TestGetDensity:
    """Additional tests for get_density method."""

    def test_get_density_returns_k_over_c(self):
        """get_density should return K / C."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.5,  # K = 4
        )

        expected_density = layer.K / layer.C
        actual_density = layer.get_density()

        assert actual_density == expected_density

    def test_density_at_full(self):
        """At density=1.0, get_density should return 1.0."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=1.0,
        )

        assert layer.get_density() == 1.0

    def test_density_at_minimum(self):
        """At density=0.1 with small C, K should be at least 1."""
        layer = CMSBlockLinear(
            in_features=32,  # C = 2
            out_features=32,
            tile_size=16,
            density=0.1,
        )

        # K = max(1, int(2 * 0.1)) = max(1, 0) = 1
        assert layer.K >= 1
        # Actual density will be K/C = 1/2 = 0.5 (not 0.1 due to minimum)
        assert layer.get_density() == layer.K / layer.C


class TestToDense:
    """Additional tests for to_dense method."""

    def test_to_dense_shape(self):
        """to_dense should return [out_features, in_features] tensor."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        dense = layer.to_dense()

        assert dense.shape == (128, 64)

    def test_to_dense_dtype(self):
        """to_dense should preserve weight dtype."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
            dtype=torch.float32,
        )

        dense = layer.to_dense()

        assert dense.dtype == torch.float32

    def test_to_dense_contains_values(self):
        """to_dense output should contain the sparse values at correct positions."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        dense = layer.to_dense()

        # Check that blocks are placed correctly
        B = layer.tile_size
        for r in range(layer.R):
            for k, c in enumerate(layer.col_indices[r]):
                c = c.item()
                block_in_dense = dense[r * B : (r + 1) * B, c * B : (c + 1) * B]
                block_in_values = layer.values[r, k]
                assert torch.equal(block_in_dense, block_in_values)


class TestIntegration:
    """Integration tests for CMSBlockLinear."""

    def test_layer_is_nn_module(self):
        """CMSBlockLinear should be a proper nn.Module."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        assert isinstance(layer, nn.Module)

    def test_parameters_registered(self):
        """Parameters should be properly registered."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
            bias=True,
        )

        param_names = [name for name, _ in layer.named_parameters()]
        assert "values" in param_names
        assert "bias" in param_names

    def test_buffers_registered(self):
        """Buffers should be properly registered."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        buffer_names = [name for name, _ in layer.named_buffers()]
        assert "col_indices" in buffer_names
        assert "block_score_ema" in buffer_names
        assert "activation_norm_acc" in buffer_names
        assert "error_norm_acc" in buffer_names
        assert "block_age" in buffer_names

    def test_extra_repr(self):
        """extra_repr should provide useful information."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        repr_str = layer.extra_repr()
        assert "in_features=64" in repr_str
        assert "out_features=128" in repr_str
        assert "tile_size=16" in repr_str
        assert "density=0.50" in repr_str


class TestTopologyStatsValues:
    """T078: Test that get_topology_stats returns correct types and ranges."""

    def test_stats_return_correct_types(self):
        """Stats should return float for scores and int for counts."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        stats = layer.get_topology_stats()

        # Check types
        assert isinstance(stats.density, float)
        assert isinstance(stats.avg_block_score, float)
        assert isinstance(stats.avg_block_age, float)
        assert isinstance(stats.column_entropy, float)
        assert isinstance(stats.num_blocks, int)

    def test_density_in_valid_range(self):
        """Density should be in [0, 1] range."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        stats = layer.get_topology_stats()
        assert 0.0 <= stats.density <= 1.0

    def test_entropy_in_valid_range(self):
        """Column entropy should be in [0, 1] range."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        stats = layer.get_topology_stats()
        assert 0.0 <= stats.column_entropy <= 1.0

    def test_num_blocks_equals_r_times_k(self):
        """num_blocks should equal R * K."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        stats = layer.get_topology_stats()
        assert stats.num_blocks == layer.R * layer.K

    def test_avg_block_age_non_negative(self):
        """Average block age should be non-negative."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        stats = layer.get_topology_stats()
        assert stats.avg_block_age >= 0.0


class TestColumnEntropyBounds:
    """T079: Test column entropy bounds with various topologies."""

    def test_entropy_in_range_with_random_topology(self):
        """Entropy should be in [0, 1] with random topology."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.5,  # K = 4
        )

        stats = layer.get_topology_stats()
        assert 0.0 <= stats.column_entropy <= 1.0

    def test_high_entropy_with_uniform_distribution(self):
        """Uniform column usage should give high entropy."""
        layer = CMSBlockLinear(
            in_features=64,  # C = 4
            out_features=64,  # R = 4
            tile_size=16,
            density=1.0,  # K = 4 (all columns used in every row)
        )

        stats = layer.get_topology_stats()
        # At full density, all columns are used equally -> high entropy
        assert stats.column_entropy > 0.9

    def test_low_entropy_with_concentrated_distribution(self):
        """Concentrated column usage should give lower entropy."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.125,  # K = 1 (only 1 column per row)
        )

        # Force all rows to use the same column
        with torch.no_grad():
            layer.col_indices.fill_(0)  # All rows use column 0

        stats = layer.get_topology_stats()
        # Only 1 column used -> entropy = 0
        assert stats.column_entropy == 0.0

    def test_entropy_increases_with_diversity(self):
        """Entropy should increase as column usage becomes more diverse."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.25,  # K = 2
        )

        # Concentrated: all rows use columns 0 and 1
        with torch.no_grad():
            layer.col_indices[:, 0] = 0
            layer.col_indices[:, 1] = 1

        stats_concentrated = layer.get_topology_stats()

        # Diverse: each row uses different columns
        with torch.no_grad():
            for r in range(layer.R):
                layer.col_indices[r, 0] = r % layer.C
                layer.col_indices[r, 1] = (r + 1) % layer.C

        stats_diverse = layer.get_topology_stats()

        # Diverse should have higher or equal entropy
        assert stats_diverse.column_entropy >= stats_concentrated.column_entropy


class TestStateDictRoundtrip:
    """T080: Test state_dict save/load roundtrip."""

    def test_state_dict_contains_all_buffers(self):
        """state_dict should contain all necessary buffers."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        state = layer.state_dict()

        # Check all expected keys are present
        assert "values" in state
        assert "col_indices" in state
        assert "block_score_ema" in state
        assert "activation_norm_acc" in state
        assert "error_norm_acc" in state
        assert "block_age" in state
        assert "_acc_steps" in state
        assert "_swap_rate_history" in state

    def test_roundtrip_preserves_forward_output(self):
        """Save/load roundtrip should preserve forward outputs."""
        torch.manual_seed(42)

        layer1 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Run some forward passes to populate state
        x = torch.randn(4, 64)
        output1 = layer1(x)

        # Save state
        state = layer1.state_dict()

        # Create new layer with same config
        layer2 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Load state (need to copy to avoid modifying original)
        state_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
        layer2.load_state_dict(state_copy)

        # Forward should produce identical output
        output2 = layer2(x)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-6), \
            f"Max diff: {(output1 - output2).abs().max().item()}"

    def test_roundtrip_preserves_all_buffers(self):
        """Save/load roundtrip should preserve all buffer values."""
        layer1 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Modify some state
        layer1._acc_steps = 5
        layer1._swap_rate_history = [0.1, 0.2, 0.3]
        layer1.block_age.fill_(3)

        # Save and load
        state = layer1.state_dict()
        state_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}

        layer2 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        layer2.load_state_dict(state_copy)

        # Verify all state matches
        assert layer2._acc_steps == 5
        assert layer2._swap_rate_history == [0.1, 0.2, 0.3]
        assert torch.equal(layer2.block_age, layer1.block_age)
        assert torch.equal(layer2.col_indices, layer1.col_indices)
        assert torch.equal(layer2.values, layer1.values)


class TestSwapRateStability:
    """T081: Test swap rate stability over multiple topology steps."""

    def test_swap_rates_are_tracked(self):
        """Swap rates should be tracked after topology steps."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        # Run topology steps
        for _ in range(5):
            layer.topology_step()

        assert len(layer._swap_rate_history) == 5

    def test_swap_rates_in_valid_range(self):
        """Swap rates should be in [0, 1] range."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        # Run topology steps
        for _ in range(10):
            result = layer.topology_step()
            assert 0.0 <= result.swap_rate <= 1.0

        # Check avg is also in range
        avg_rate = layer.get_avg_swap_rate()
        assert 0.0 <= avg_rate <= 1.0

    def test_swap_rates_not_always_zero(self):
        """With exploration, swap rates should not always be zero."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.25,  # K = 2
        )

        # Populate scoring state to enable swaps
        layer.train()
        x = torch.randn(8, 128, requires_grad=True)
        for _ in range(5):
            y = layer(x)
            y.sum().backward()
            layer.accumulate_scores()

        # Run several topology steps
        swap_rates = []
        for _ in range(10):
            result = layer.topology_step()
            swap_rates.append(result.swap_rate)
            # Repopulate scores
            y = layer(x)
            y.sum().backward()
            layer.accumulate_scores()

        # At least some swaps should occur due to exploration
        assert sum(swap_rates) > 0, "Expected some swaps to occur"

    def test_swap_rates_not_always_one(self):
        """Swap rates should not always be 100%."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        # Run topology steps
        swap_rates = []
        for _ in range(10):
            result = layer.topology_step()
            swap_rates.append(result.swap_rate)

        # Not all swaps should be 100%
        assert not all(r == 1.0 for r in swap_rates), \
            "Expected not all swap rates to be 100%"

    def test_avg_swap_rate_returns_zero_with_no_history(self):
        """get_avg_swap_rate should return 0.0 when no history."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        assert layer.get_avg_swap_rate() == 0.0


class TestBlockAgeDistribution:
    """Additional tests for get_block_age_distribution."""

    def test_initial_ages_are_zero(self):
        """Initially all block ages should be 0."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        dist = layer.get_block_age_distribution()
        assert dist == {0: layer.R * layer.K}

    def test_ages_increment_on_score_step(self):
        """Block ages should increment on score_step."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Call score_step which increments ages
        layer.score_step()

        dist = layer.get_block_age_distribution()
        assert dist == {1: layer.R * layer.K}

    def test_ages_reset_on_topology_step_for_new_blocks(self):
        """New blocks from topology_step should have age 0."""
        layer = CMSBlockLinear(
            in_features=128,  # C = 8
            out_features=64,  # R = 4
            tile_size=16,
            density=0.25,  # K = 2
        )

        # Increment ages first
        layer.score_step()
        layer.score_step()

        # Do topology step (may swap some blocks)
        result = layer.topology_step()

        if result.num_swaps > 0:
            # Some blocks should have age 0
            dist = layer.get_block_age_distribution()
            assert 0 in dist
            assert dist[0] >= result.num_swaps


class TestTopologyHistory:
    """Tests for topology snapshot history."""

    def test_history_is_saved_on_topology_step(self):
        """Topology history should be saved on topology_step."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        layer.topology_step()

        history = layer.get_topology_history()
        assert len(history) == 1
        assert len(history[0]) == 2  # (before, after) tuple

    def test_history_limited_to_max_size(self):
        """History should be limited to max size."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )
        layer._topology_history_max_size = 5

        for _ in range(10):
            layer.topology_step()

        history = layer.get_topology_history()
        assert len(history) == 5

    def test_history_not_saved_when_disabled(self):
        """History should not be saved when save_snapshot=False."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=64,
            tile_size=16,
            density=0.5,
        )

        layer.topology_step(save_snapshot=False)

        history = layer.get_topology_history()
        assert len(history) == 0


class TestFromDense:
    """T101: Tests for CMSBlockLinear.from_dense() class method."""

    def test_from_dense_basic(self):
        """from_dense should create a sparse layer from a dense layer."""
        torch.manual_seed(42)

        dense = nn.Linear(128, 256, bias=True)
        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)

        assert sparse.in_features == 128
        assert sparse.out_features == 256
        assert sparse.tile_size == 16
        assert sparse.bias is not None

    def test_from_dense_preserves_important_weights(self):
        """from_dense should preserve high-magnitude blocks.

        This test creates a dense weight matrix with known high-magnitude
        regions and verifies that from_dense selects those regions.
        """
        torch.manual_seed(42)

        # Create dense layer with known high-magnitude regions
        dense = nn.Linear(64, 64, bias=True)
        B = 16  # tile_size
        R = 64 // B  # 4 rows
        C = 64 // B  # 4 columns

        # Initialize all weights to small values
        with torch.no_grad():
            dense.weight.fill_(0.01)

            # Make specific blocks have high magnitude
            # Row 0: column 0 is high
            dense.weight[0:B, 0:B] = 10.0
            # Row 1: column 2 is high
            dense.weight[B:2*B, 2*B:3*B] = 10.0
            # Row 2: column 1 is high
            dense.weight[2*B:3*B, B:2*B] = 10.0
            # Row 3: column 3 is high
            dense.weight[3*B:4*B, 3*B:4*B] = 10.0

        # Convert to sparse at 25% density (1 block per row)
        sparse = CMSBlockLinear.from_dense(dense, tile_size=B, density=0.25)

        # Verify the highest magnitude blocks were selected
        # K should be 1 at 25% density with C=4
        assert sparse.K == 1

        # Check that col_indices point to the high-magnitude columns
        expected_cols = [0, 2, 1, 3]  # High-magnitude column for each row
        for r, expected_col in enumerate(expected_cols):
            actual_col = sparse.col_indices[r, 0].item()
            assert actual_col == expected_col, \
                f"Row {r}: expected col {expected_col}, got {actual_col}"

        # Verify the values match the high-magnitude blocks
        for r in range(R):
            block_val = sparse.values[r, 0, 0, 0].item()
            assert abs(block_val - 10.0) < 0.01, \
                f"Row {r}: expected ~10.0, got {block_val}"

    def test_from_dense_preserves_bias(self):
        """from_dense should copy bias from dense layer."""
        torch.manual_seed(42)

        dense = nn.Linear(64, 128, bias=True)
        with torch.no_grad():
            dense.bias.fill_(3.14)

        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)

        assert sparse.bias is not None
        assert torch.allclose(sparse.bias, torch.full((128,), 3.14))

    def test_from_dense_no_bias(self):
        """from_dense should handle layers without bias."""
        torch.manual_seed(42)

        dense = nn.Linear(64, 128, bias=False)
        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)

        assert sparse.bias is None

    def test_from_dense_raises_on_invalid_dimensions(self):
        """from_dense should raise ValueError for non-divisible dimensions."""
        dense = nn.Linear(65, 128)  # 65 not divisible by 16

        with pytest.raises(ValueError, match="in_features .* must be divisible"):
            CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)

    def test_from_dense_forward_output_similar(self):
        """Sparse layer from_dense should produce similar output to dense.

        At the selected blocks, the output should match. Overall output
        will differ due to pruned blocks, but should be correlated.
        """
        torch.manual_seed(42)

        dense = nn.Linear(64, 128, bias=True)
        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=1.0)

        x = torch.randn(8, 64)
        dense_out = dense(x)
        sparse_out = sparse(x)

        # At full density, outputs should match very closely
        assert torch.allclose(dense_out, sparse_out, rtol=1e-4, atol=1e-5)

    def test_from_dense_preserves_dtype(self):
        """from_dense should preserve the dense layer's dtype."""
        dense = nn.Linear(64, 128).to(torch.float32)
        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)

        assert sparse.values.dtype == torch.float32
        assert sparse.bias.dtype == torch.float32

    def test_from_dense_high_density_selects_all(self):
        """At density=1.0, all blocks should be selected."""
        torch.manual_seed(42)

        dense = nn.Linear(64, 128)  # C = 4, R = 8
        sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=1.0)

        # At full density, K should equal C
        assert sparse.K == sparse.C

        # Each row should have all columns represented
        for r in range(sparse.R):
            cols = sorted(sparse.col_indices[r].tolist())
            expected = list(range(sparse.C))
            assert cols == expected, f"Row {r}: expected {expected}, got {cols}"
