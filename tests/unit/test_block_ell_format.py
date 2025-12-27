"""Tests for Block-ELL sparse tensor format utilities.

Tests cover:
- T011: BlockELLFormat validation rules
- T012: create_random_topology uniqueness guarantees
- T013: to_dense/from_dense roundtrip preservation
- T014: from_dense respects density parameter

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

import pytest
import torch

from titans_core.layers.block_ell import (
    BlockELLFormat,
    create_random_topology,
    to_dense,
    from_dense,
)


class TestBlockELLFormatValidation:
    """T011: Test BlockELLFormat validation rules."""

    def test_valid_format_passes_validation(self):
        """A properly constructed BlockELLFormat should validate."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        col_indices = create_random_topology(R=R, C=C, K=K)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is True
        assert error is None

    def test_k_exceeds_c_fails(self):
        """K > C should fail validation (can't have more blocks than columns)."""
        R, C, K, B = 4, 3, 5, 16  # K=5 > C=3
        values = torch.randn(R, K, B, B)
        col_indices = torch.zeros(R, K, dtype=torch.long)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "K (5) cannot exceed C (3)" in error

    def test_wrong_values_shape_fails(self):
        """Values tensor with wrong shape should fail validation."""
        R, C, K, B = 4, 8, 3, 16
        wrong_values = torch.randn(R, K + 1, B, B)  # Wrong K dimension
        col_indices = create_random_topology(R=R, C=C, K=K)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=wrong_values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "values shape" in error
        assert "doesn't match expected" in error

    def test_wrong_col_indices_shape_fails(self):
        """Col_indices tensor with wrong shape should fail validation."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        wrong_indices = torch.zeros(R, K + 1, dtype=torch.long)  # Wrong K dimension

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=wrong_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "col_indices shape" in error

    def test_out_of_range_col_indices_fails(self):
        """Column indices >= C should fail validation."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        # All indices are C (out of range, should be < C)
        col_indices = torch.full((R, K), C, dtype=torch.long)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "out-of-range" in error

    def test_negative_col_indices_fails(self):
        """Negative column indices should fail validation."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        col_indices = torch.full((R, K), -1, dtype=torch.long)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "negative" in error

    def test_duplicate_col_indices_in_row_fails(self):
        """Duplicate column indices within a row should fail validation."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        # All indices in each row are the same (duplicates)
        col_indices = torch.zeros(R, K, dtype=torch.long)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is False
        assert "duplicate" in error.lower()

    def test_zero_dimensions_fail(self):
        """Zero or negative dimensions should fail validation."""
        # R = 0
        fmt = BlockELLFormat(
            R=0, C=8, K=3, B=16,
            values=torch.randn(0, 3, 16, 16),
            col_indices=torch.zeros(0, 3, dtype=torch.long)
        )
        is_valid, error = fmt.validate()
        assert is_valid is False
        assert "R (0) must be at least 1" in error

    def test_properties_compute_correctly(self):
        """Verify computed properties match expected values."""
        R, C, K, B = 4, 8, 3, 16
        values = torch.randn(R, K, B, B)
        col_indices = create_random_topology(R=R, C=C, K=K)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)

        assert fmt.out_features == R * B == 64
        assert fmt.in_features == C * B == 128
        assert fmt.total_blocks == R * K == 12
        assert fmt.total_parameters == R * K * B * B == 3072
        assert fmt.density == K / C == 0.375


class TestRandomTopologyUniqueness:
    """T012: Test create_random_topology uniqueness guarantees."""

    def test_all_col_indices_unique_per_row(self):
        """Verify all col_indices[r] values are unique within each row."""
        R, C, K = 10, 20, 8

        col_indices = create_random_topology(R=R, C=C, K=K)

        for r in range(R):
            row_indices = col_indices[r]
            unique_count = row_indices.unique().numel()
            assert unique_count == K, f"Row {r} has {unique_count} unique values, expected {K}"

    def test_all_values_in_valid_range(self):
        """All column indices should be in [0, C)."""
        R, C, K = 10, 20, 8

        col_indices = create_random_topology(R=R, C=C, K=K)

        assert col_indices.min() >= 0, f"Min index {col_indices.min()} < 0"
        assert col_indices.max() < C, f"Max index {col_indices.max()} >= C ({C})"

    def test_reproducibility_with_generator(self):
        """Same generator seed should produce same topology."""
        R, C, K = 5, 10, 4

        gen1 = torch.Generator().manual_seed(12345)
        indices1 = create_random_topology(R=R, C=C, K=K, generator=gen1)

        gen2 = torch.Generator().manual_seed(12345)
        indices2 = create_random_topology(R=R, C=C, K=K, generator=gen2)

        assert torch.equal(indices1, indices2), "Generator should produce reproducible results"

    def test_different_seeds_produce_different_topologies(self):
        """Different generator seeds should (usually) produce different topologies."""
        R, C, K = 5, 10, 4

        gen1 = torch.Generator().manual_seed(111)
        indices1 = create_random_topology(R=R, C=C, K=K, generator=gen1)

        gen2 = torch.Generator().manual_seed(222)
        indices2 = create_random_topology(R=R, C=C, K=K, generator=gen2)

        # Very unlikely to be equal with different seeds
        assert not torch.equal(indices1, indices2), "Different seeds should produce different results"

    def test_k_equals_c_uses_all_columns(self):
        """When K=C, each row should contain all column indices."""
        R, C = 4, 8
        K = C  # All columns

        col_indices = create_random_topology(R=R, C=C, K=K)

        for r in range(R):
            row_sorted = col_indices[r].sort().values
            expected = torch.arange(C)
            assert torch.equal(row_sorted, expected), f"Row {r} doesn't contain all columns"

    def test_k_greater_than_c_raises_error(self):
        """K > C should raise ValueError."""
        with pytest.raises(ValueError, match="K .* cannot exceed C"):
            create_random_topology(R=4, C=5, K=10)


class TestToDenseFromDenseRoundtrip:
    """T013: Test to_dense/from_dense roundtrip preserves values."""

    def test_roundtrip_preserves_selected_blocks(self):
        """Converting dense -> sparse -> dense should preserve selected blocks."""
        out_features, in_features = 64, 128
        B = 16
        density = 0.5

        # Create random dense matrix
        original = torch.randn(out_features, in_features)

        # Convert to sparse
        values, col_indices, R, C, K, B_out = from_dense(original, tile_size=B, density=density)

        # Convert back to dense
        reconstructed = to_dense(values, col_indices, R=R, C=C, K=K, B=B_out)

        # Verify selected blocks match
        for r in range(R):
            for k_idx, c in enumerate(col_indices[r]):
                c = c.item()
                original_block = original[r * B : (r + 1) * B, c * B : (c + 1) * B]
                reconstructed_block = reconstructed[r * B : (r + 1) * B, c * B : (c + 1) * B]
                assert torch.allclose(original_block, reconstructed_block), \
                    f"Block ({r}, {c}) values differ after roundtrip"

    def test_to_dense_places_blocks_correctly(self):
        """to_dense should place blocks at correct positions."""
        R, C, K, B = 2, 4, 2, 4

        # Create known values
        values = torch.arange(R * K * B * B, dtype=torch.float).view(R, K, B, B)
        col_indices = torch.tensor([[0, 2], [1, 3]])  # Row 0: cols 0,2; Row 1: cols 1,3

        dense = to_dense(values, col_indices, R=R, C=C, K=K, B=B)

        # Verify blocks are at correct positions
        assert torch.equal(dense[0:B, 0:B], values[0, 0])
        assert torch.equal(dense[0:B, 2 * B : 3 * B], values[0, 1])
        assert torch.equal(dense[B : 2 * B, B : 2 * B], values[1, 0])
        assert torch.equal(dense[B : 2 * B, 3 * B : 4 * B], values[1, 1])

    def test_to_dense_zeros_unselected_blocks(self):
        """to_dense should have zeros where no blocks are placed."""
        R, C, K, B = 2, 4, 2, 4

        values = torch.ones(R, K, B, B)
        col_indices = torch.tensor([[0, 2], [1, 3]])

        dense = to_dense(values, col_indices, R=R, C=C, K=K, B=B)

        # Row 0, col 1 should be zeros (not selected)
        assert torch.equal(dense[0:B, B : 2 * B], torch.zeros(B, B))
        # Row 0, col 3 should be zeros
        assert torch.equal(dense[0:B, 3 * B : 4 * B], torch.zeros(B, B))
        # Row 1, col 0 should be zeros
        assert torch.equal(dense[B : 2 * B, 0:B], torch.zeros(B, B))

    def test_output_shapes_are_correct(self):
        """Verify output tensor shapes match expected dimensions."""
        out_features, in_features = 64, 128
        B = 16
        density = 0.25

        dense = torch.randn(out_features, in_features)
        values, col_indices, R, C, K, B_out = from_dense(dense, tile_size=B, density=density)

        expected_R = out_features // B
        expected_C = in_features // B
        expected_K = max(1, int(expected_C * density))

        assert R == expected_R
        assert C == expected_C
        assert K == expected_K
        assert B_out == B
        assert values.shape == (R, K, B, B)
        assert col_indices.shape == (R, K)


class TestFromDenseRespectsDensity:
    """T014: Test from_dense respects density parameter."""

    @pytest.mark.parametrize("density", [0.25, 0.5, 0.75, 1.0])
    def test_correct_number_of_blocks_per_row(self, density: float):
        """Number of blocks per row should match K = int(C * density)."""
        out_features, in_features = 64, 128
        B = 16
        C = in_features // B

        dense = torch.randn(out_features, in_features)
        values, col_indices, R, _, K, _ = from_dense(dense, tile_size=B, density=density)

        expected_K = max(1, int(C * density))
        assert K == expected_K, f"density={density}: K={K}, expected {expected_K}"
        assert values.shape[1] == K
        assert col_indices.shape[1] == K

    def test_selects_highest_magnitude_blocks(self):
        """from_dense should select blocks with highest Frobenius norm."""
        R, C, B = 1, 4, 4
        out_features, in_features = R * B, C * B

        # Create dense matrix with known block magnitudes
        dense = torch.zeros(out_features, in_features)
        # Block at col 2 has largest magnitude (norm = 16*10 = 160)
        dense[0:B, 2 * B : 3 * B] = 10.0
        # Block at col 0 has second largest (norm = 16*5 = 80)
        dense[0:B, 0:B] = 5.0
        # Block at col 3 has third largest (norm = 16*2 = 32)
        dense[0:B, 3 * B : 4 * B] = 2.0
        # Block at col 1 has smallest (norm = 16*1 = 16)
        dense[0:B, B : 2 * B] = 1.0

        # With density=0.5, K=2, should select cols 0 and 2
        _, col_indices, _, _, K, _ = from_dense(dense, tile_size=B, density=0.5)

        assert K == 2
        selected = set(col_indices[0].tolist())
        assert selected == {0, 2}, f"Expected {{0, 2}} (highest magnitude), got {selected}"

    def test_minimum_one_block_per_row(self):
        """Even with very low density, at least 1 block per row."""
        out_features, in_features = 64, 128
        B = 16

        dense = torch.randn(out_features, in_features)
        _, col_indices, _, _, K, _ = from_dense(dense, tile_size=B, density=0.01)

        # K should be at least 1
        assert K >= 1
        assert col_indices.shape[1] >= 1

    def test_density_one_selects_all_blocks(self):
        """density=1.0 should select all blocks."""
        out_features, in_features = 32, 64
        B = 16
        C = in_features // B

        dense = torch.randn(out_features, in_features)
        _, col_indices, _, _, K, _ = from_dense(dense, tile_size=B, density=1.0)

        assert K == C, f"density=1.0 should select all {C} blocks, got K={K}"

    def test_invalid_dimensions_raise_error(self):
        """Dimensions not divisible by tile_size should raise ValueError."""
        # out_features not divisible
        with pytest.raises(ValueError, match="out_features .* must be divisible"):
            from_dense(torch.randn(33, 64), tile_size=16)

        # in_features not divisible
        with pytest.raises(ValueError, match="in_features .* must be divisible"):
            from_dense(torch.randn(32, 65), tile_size=16)

    def test_col_indices_are_unique_per_row(self):
        """from_dense should produce unique column indices per row."""
        out_features, in_features = 64, 128
        B = 16
        R = out_features // B

        dense = torch.randn(out_features, in_features)
        _, col_indices, _, _, K, _ = from_dense(dense, tile_size=B, density=0.5)

        for r in range(R):
            unique_count = col_indices[r].unique().numel()
            assert unique_count == K, f"Row {r} has duplicate column indices"


class TestIntegration:
    """Integration tests combining multiple Block-ELL operations."""

    def test_create_topology_and_validate(self):
        """Creating topology with create_random_topology should pass validation."""
        R, C, K, B = 8, 16, 6, 16

        col_indices = create_random_topology(R=R, C=C, K=K)
        values = torch.randn(R, K, B, B)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is True, f"Validation failed: {error}"

    def test_from_dense_produces_valid_format(self):
        """from_dense output should pass BlockELLFormat validation."""
        out_features, in_features = 64, 128
        B = 16
        density = 0.5

        dense = torch.randn(out_features, in_features)
        values, col_indices, R, C, K, B_out = from_dense(dense, tile_size=B, density=density)

        fmt = BlockELLFormat(R=R, C=C, K=K, B=B_out, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()

        assert is_valid is True, f"from_dense output failed validation: {error}"

    def test_full_pipeline_dense_to_sparse_to_dense(self):
        """Full pipeline: dense -> sparse (from_dense) -> dense (to_dense)."""
        torch.manual_seed(42)
        out_features, in_features = 128, 256
        B = 16
        density = 0.5

        # Create original dense matrix
        original = torch.randn(out_features, in_features)

        # Convert to Block-ELL sparse format
        values, col_indices, R, C, K, B_out = from_dense(original, tile_size=B, density=density)

        # Validate the sparse format
        fmt = BlockELLFormat(R=R, C=C, K=K, B=B_out, values=values, col_indices=col_indices)
        is_valid, error = fmt.validate()
        assert is_valid, f"Sparse format invalid: {error}"

        # Convert back to dense
        reconstructed = to_dense(values, col_indices, R=R, C=C, K=K, B=B_out)

        # Verify dimensions
        assert reconstructed.shape == original.shape

        # Verify selected blocks match original
        total_blocks_checked = 0
        for r in range(R):
            for k_idx, c in enumerate(col_indices[r]):
                c = c.item()
                original_block = original[r * B : (r + 1) * B, c * B : (c + 1) * B]
                reconstructed_block = reconstructed[r * B : (r + 1) * B, c * B : (c + 1) * B]
                assert torch.allclose(original_block, reconstructed_block)
                total_blocks_checked += 1

        assert total_blocks_checked == R * K
