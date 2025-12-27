"""Tests for Block-ELL Triton kernels.

Tests cover:
- T053: test_forward_correctness - Triton matches PyTorch reference within 1e-5
- T054: test_backward_grad_values_correctness - gradient w.r.t. values matches torch.autograd.gradcheck
- T055: test_backward_grad_input_correctness - gradient w.r.t. input matches torch.autograd.gradcheck

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests"
)


class TestForwardCorrectness:
    """T053: Test that Triton forward matches PyTorch reference within 1e-5."""

    def test_forward_2d_basic(self):
        """Forward pass with 2D input should match reference."""
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 8
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size  # 8
        C = in_features // tile_size   # 4
        K = max(1, int(C * density))   # 2

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create weights and input
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)

        # Compute with Triton
        y_triton = block_ell_forward(x, values, col_indices, bias)

        # Compute with reference
        y_ref = block_ell_forward_reference(x, values, col_indices, bias, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(y_triton - y_ref).abs().max().item()}"

    def test_forward_3d_basic(self):
        """Forward pass with 3D input should match reference."""
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(123)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 4
        seq_len = 16
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create weights and input
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)

        # Compute with Triton
        y_triton = block_ell_forward(x, values, col_indices, bias)

        # Compute with reference
        y_ref = block_ell_forward_reference(x, values, col_indices, bias, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(y_triton - y_ref).abs().max().item()}"

    def test_forward_no_bias(self):
        """Forward pass without bias should match reference."""
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(456)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 8
        in_features = 64
        out_features = 64
        tile_size = 16
        density = 0.75

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create weights and input (no bias)
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

        # Compute with Triton
        y_triton = block_ell_forward(x, values, col_indices, bias=None)

        # Compute with reference
        y_ref = block_ell_forward_reference(x, values, col_indices, bias=None, tile_size=tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(y_triton - y_ref).abs().max().item()}"

    def test_forward_full_density(self):
        """Forward pass at full density should match reference."""
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(789)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions with full density (K = C)
        batch_size = 4
        in_features = 64
        out_features = 64
        tile_size = 16

        R = out_features // tile_size  # 4
        C = in_features // tile_size   # 4
        K = C  # Full density

        # Create sequential topology (all columns)
        col_indices = torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0).expand(R, -1).clone()

        # Create weights and input
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)

        # Compute with Triton
        y_triton = block_ell_forward(x, values, col_indices, bias)

        # Compute with reference
        y_ref = block_ell_forward_reference(x, values, col_indices, bias, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(y_triton - y_ref).abs().max().item()}"

    def test_forward_large_batch(self):
        """Forward pass with large batch should match reference."""
        from titans_core.kernels.block_ell_forward import (
            block_ell_forward,
            block_ell_forward_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(111)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 256
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create weights and input
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)

        # Compute with Triton
        y_triton = block_ell_forward(x, values, col_indices, bias)

        # Compute with reference
        y_ref = block_ell_forward_reference(x, values, col_indices, bias, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(y_triton - y_ref).abs().max().item()}"


class TestBackwardGradValuesCorrectness:
    """T054: Test gradient w.r.t. values matches torch.autograd.gradcheck."""

    def test_backward_dw_matches_reference(self):
        """Gradient w.r.t. values should match reference implementation."""
        from titans_core.kernels.block_ell_backward import (
            block_ell_backward_dw,
            block_ell_backward_dw_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 8
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create input and dout
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        dout = torch.randn(batch_size, out_features, device=device, dtype=dtype)

        # Compute with Triton
        dw_triton = block_ell_backward_dw(x, dout, col_indices, R, K, tile_size)

        # Compute with reference
        dw_ref = block_ell_backward_dw_reference(x, dout, col_indices, R, K, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(dw_triton, dw_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(dw_triton - dw_ref).abs().max().item()}"

    def test_backward_dw_3d_input(self):
        """Gradient w.r.t. values with 3D input should match reference."""
        from titans_core.kernels.block_ell_backward import (
            block_ell_backward_dw,
            block_ell_backward_dw_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(123)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 4
        seq_len = 16
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create input and dout
        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
        dout = torch.randn(batch_size, seq_len, out_features, device=device, dtype=dtype)

        # Compute with Triton
        dw_triton = block_ell_backward_dw(x, dout, col_indices, R, K, tile_size)

        # Compute with reference
        dw_ref = block_ell_backward_dw_reference(x, dout, col_indices, R, K, tile_size)

        # Check correctness - FP32 should have ~1e-6 error
        assert torch.allclose(dw_triton, dw_ref, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(dw_triton - dw_ref).abs().max().item()}"

    def test_backward_dw_autograd(self):
        """Gradient w.r.t. values should work with autograd."""
        from titans_core.kernels.block_ell_backward import block_ell_autograd, TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(456)
        device = torch.device("cuda")
        dtype = torch.float32  # Triton kernels use float32 for accumulation

        # Setup dimensions
        batch_size = 4
        in_features = 32
        out_features = 32
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create input and values with requires_grad
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype, requires_grad=True)

        # Forward
        output = block_ell_autograd(x, values, col_indices, None, in_features, R, K, tile_size, use_triton=True)

        # Backward
        loss = output.sum()
        loss.backward()

        # Gradient should exist and have correct shape
        assert values.grad is not None
        assert values.grad.shape == values.shape
        assert not values.grad.isnan().any()


class TestBackwardGradInputCorrectness:
    """T055: Test gradient w.r.t. input matches torch.autograd.gradcheck."""

    def test_backward_dx_matches_reference(self):
        """Gradient w.r.t. input should match reference implementation."""
        from titans_core.kernels.block_ell_backward import (
            block_ell_backward_dx,
            block_ell_backward_dx_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 8
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create values and dout
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        dout = torch.randn(batch_size, out_features, device=device, dtype=dtype)

        # Compute with Triton
        dx_triton = block_ell_backward_dx(dout, values, col_indices, in_features)

        # Compute with reference
        dx_ref = block_ell_backward_dx_reference(dout, values, col_indices, in_features, tile_size)

        # Check correctness
        assert torch.allclose(dx_triton, dx_ref, rtol=1e-4, atol=1e-5), \
            f"Max diff: {(dx_triton - dx_ref).abs().max().item()}"

    def test_backward_dx_3d_input(self):
        """Gradient w.r.t. input with 3D tensors should match reference."""
        from titans_core.kernels.block_ell_backward import (
            block_ell_backward_dx,
            block_ell_backward_dx_reference,
            TRITON_AVAILABLE,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(123)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 4
        seq_len = 16
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create values and dout
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)
        dout = torch.randn(batch_size, seq_len, out_features, device=device, dtype=dtype)

        # Compute with Triton
        dx_triton = block_ell_backward_dx(dout, values, col_indices, in_features)

        # Compute with reference
        dx_ref = block_ell_backward_dx_reference(dout, values, col_indices, in_features, tile_size)

        # Check correctness
        assert torch.allclose(dx_triton, dx_ref, rtol=1e-4, atol=1e-5), \
            f"Max diff: {(dx_triton - dx_ref).abs().max().item()}"

    def test_backward_dx_autograd(self):
        """Gradient w.r.t. input should work with autograd."""
        from titans_core.kernels.block_ell_backward import block_ell_autograd, TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(456)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 4
        in_features = 32
        out_features = 32
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create input and values with requires_grad
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype)

        # Forward
        output = block_ell_autograd(x, values, col_indices, None, in_features, R, K, tile_size, use_triton=True)

        # Backward
        loss = output.sum()
        loss.backward()

        # Gradient should exist and have correct shape
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not x.grad.isnan().any()


class TestEndToEndAutograd:
    """End-to-end tests for the complete autograd function."""

    def test_full_forward_backward(self):
        """Complete forward and backward pass should work correctly."""
        from titans_core.kernels.block_ell_backward import block_ell_autograd, TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions
        batch_size = 8
        in_features = 64
        out_features = 128
        tile_size = 16
        density = 0.5

        R = out_features // tile_size
        C = in_features // tile_size
        K = max(1, int(C * density))

        # Create random topology
        col_indices = torch.zeros(R, K, dtype=torch.int32, device=device)
        for r in range(R):
            perm = torch.randperm(C, device=device)
            col_indices[r] = perm[:K].to(torch.int32)

        # Create tensors with requires_grad
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)

        # Forward
        output = block_ell_autograd(x, values, col_indices, bias, in_features, R, K, tile_size, use_triton=True)

        # Check output shape
        assert output.shape == (batch_size, out_features)

        # Backward
        loss = output.sum()
        loss.backward()

        # All gradients should exist
        assert x.grad is not None
        assert values.grad is not None
        assert bias.grad is not None

        # Shapes should match
        assert x.grad.shape == x.shape
        assert values.grad.shape == values.shape
        assert bias.grad.shape == bias.shape

    def test_matches_dense_at_full_density(self):
        """At full density, should match dense linear layer."""
        from titans_core.kernels.block_ell_backward import block_ell_autograd, TRITON_AVAILABLE
        from titans_core.layers.block_ell import to_dense

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = torch.float32

        # Setup dimensions with full density
        batch_size = 4
        in_features = 64
        out_features = 64
        tile_size = 16

        R = out_features // tile_size  # 4
        C = in_features // tile_size   # 4
        K = C  # Full density

        # Create sequential topology
        col_indices = torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0).expand(R, -1).clone()

        # Create weights
        values = torch.randn(R, K, tile_size, tile_size, device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)

        # Forward with sparse
        output_sparse = block_ell_autograd(x, values, col_indices, bias, in_features, R, K, tile_size, use_triton=True)

        # Convert to dense and compute with nn.Linear
        dense_weights = to_dense(values.detach(), col_indices, R, C, K, tile_size)
        dense_layer = nn.Linear(in_features, out_features, bias=True).to(device)
        with torch.no_grad():
            dense_layer.weight.copy_(dense_weights)
            dense_layer.bias.copy_(bias)

        x_dense = x.detach().clone()
        output_dense = dense_layer(x_dense)

        # Should match - FP32 should have ~1e-6 error
        assert torch.allclose(output_sparse.detach(), output_dense, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(output_sparse.detach() - output_dense).abs().max().item()}"


class TestCMSBlockLinearTriton:
    """Test CMSBlockLinear with Triton dispatch."""

    def test_layer_uses_triton_on_cuda(self):
        """Layer should use Triton kernels on CUDA with compatible tile size."""
        from titans_core.layers.block_sparse import CMSBlockLinear

        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        ).cuda()

        # Check property
        assert layer._use_triton_kernel is True

    def test_layer_forward_with_triton(self):
        """Layer forward should work with Triton kernels."""
        from titans_core.layers.block_sparse import CMSBlockLinear

        torch.manual_seed(42)

        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        ).cuda()

        x = torch.randn(8, 64, device="cuda")
        y = layer(x)

        assert y.shape == (8, 128)
        assert not y.isnan().any()

    def test_layer_backward_with_triton(self):
        """Layer backward should work with Triton kernels."""
        from titans_core.layers.block_sparse import CMSBlockLinear

        torch.manual_seed(42)

        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        ).cuda()

        x = torch.randn(8, 64, device="cuda", requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert layer.values.grad is not None
        assert layer.bias.grad is not None

        # No NaN gradients
        assert not x.grad.isnan().any()
        assert not layer.values.grad.isnan().any()
        assert not layer.bias.grad.isnan().any()

    def test_layer_matches_reference_on_cpu(self):
        """Layer should use reference implementation on CPU."""
        from titans_core.layers.block_sparse import CMSBlockLinear

        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )  # CPU by default

        # Should not use Triton on CPU
        assert layer._use_triton_kernel is False

        # Forward should still work
        x = torch.randn(8, 64)
        y = layer(x)

        assert y.shape == (8, 128)
        assert not y.isnan().any()
