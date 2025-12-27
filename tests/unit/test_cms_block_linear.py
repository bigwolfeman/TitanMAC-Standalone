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
