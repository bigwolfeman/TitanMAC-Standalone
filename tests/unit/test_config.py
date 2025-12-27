"""Tests for TitanMACConfig, particularly block-sparse configuration.

Tests:
    - T066: test_config_validation - Verify invalid density/tile_size raises ValueError
    - T067: test_selective_layer_sparsity - Verify only specified layers use CMSBlockLinear
    - T068: test_per_layer_density - Verify different densities per layer when configured
"""

import pytest
import torch
import torch.nn as nn

from titans_core.config import TitanMACConfig


class TestBlockSparseConfigValidation:
    """T066: Test validation of block-sparse configuration fields."""

    def test_valid_config_minimal(self):
        """Test that minimal valid block-sparse config passes validation."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_tile_size=16,
            mlp_block_sparse_density=0.5,
        )
        config.validate()  # Should not raise

    def test_invalid_density_too_low(self):
        """Test that density below 0.1 raises ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=0.05,  # Below 0.1
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_density.*\\[0.1, 1.0\\]"):
            config.validate()

    def test_invalid_density_too_high(self):
        """Test that density above 1.0 raises ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=1.5,  # Above 1.0
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_density.*\\[0.1, 1.0\\]"):
            config.validate()

    def test_valid_density_boundary_low(self):
        """Test that density=0.1 is valid."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=0.1,
        )
        config.validate()  # Should not raise

    def test_valid_density_boundary_high(self):
        """Test that density=1.0 is valid."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=1.0,
        )
        config.validate()  # Should not raise

    def test_invalid_tile_size(self):
        """Test that invalid tile sizes raise ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_tile_size=24,  # Not in {8, 16, 32}
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_tile_size.*\\{8, 16, 32\\}"):
            config.validate()

    def test_valid_tile_sizes(self):
        """Test that valid tile sizes (8, 16, 32) pass validation."""
        for tile_size in [8, 16, 32]:
            config = TitanMACConfig(
                d_model=64,
                d_ff=256,
                n_heads=4,
                n_layers=4,
                mlp_use_block_sparse=True,
                mlp_block_sparse_tile_size=tile_size,
            )
            config.validate()  # Should not raise

    def test_invalid_layers_not_tuple(self):
        """Test that non-tuple layers raises ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=[0, 1],  # List, not tuple
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_layers must be a tuple"):
            config.validate()

    def test_invalid_layers_negative_index(self):
        """Test that negative layer indices raise ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(-1, 0),  # Negative index
        )
        with pytest.raises(ValueError, match="non-negative ints"):
            config.validate()

    def test_invalid_layers_exceeds_n_layers(self):
        """Test that layer indices >= n_layers raise ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(0, 1, 5),  # 5 >= n_layers (4)
        )
        with pytest.raises(ValueError, match="exceeds n_layers"):
            config.validate()

    def test_valid_layers_tuple(self):
        """Test that valid layers tuple passes validation."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(0, 2, 3),
        )
        config.validate()  # Should not raise

    def test_invalid_component(self):
        """Test that invalid components raise ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_components=('mlp', 'invalid_component'),
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_components.*'invalid_component'"):
            config.validate()

    def test_valid_components(self):
        """Test that valid components pass validation."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_components=('mlp', 'attention'),
        )
        config.validate()  # Should not raise

    def test_per_layer_density_dict_invalid_layer_index(self):
        """Test that per-layer density with negative layer index raises ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density={-1: 0.5, 0: 0.3},  # Negative layer index
        )
        with pytest.raises(ValueError, match="layer index must be non-negative"):
            config.validate()

    def test_per_layer_density_dict_invalid_density_value(self):
        """Test that per-layer density with invalid density value raises ValueError."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density={0: 0.05, 1: 0.5},  # 0.05 is invalid
        )
        with pytest.raises(ValueError, match="mlp_block_sparse_density for layer 0.*\\[0.1, 1.0\\]"):
            config.validate()

    def test_per_layer_density_dict_valid(self):
        """Test that valid per-layer density dict passes validation."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density={0: 0.3, 1: 0.5, 2: 0.7},
        )
        config.validate()  # Should not raise

    def test_disabled_block_sparse_skips_validation(self):
        """Test that validation is skipped when mlp_use_block_sparse=False."""
        # These would be invalid if validation ran
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=False,  # Disabled
            mlp_block_sparse_tile_size=99,  # Invalid
            mlp_block_sparse_density=5.0,  # Invalid
        )
        # Should not raise because block-sparse is disabled
        config.validate()


class TestSelectiveLayerSparsity:
    """T067: Test that only specified layers use CMSBlockLinear."""

    def test_all_layers_sparse_when_no_layers_specified(self):
        """When mlp_block_sparse_layers is None, all layers should be sparse."""
        config = TitanMACConfig(
            d_model=64,  # Divisible by 16
            d_ff=256,  # Divisible by 16
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=None,  # All layers
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC
        from titans_core.layers.block_sparse import CMSBlockLinear

        model = TitanMAC(config)
        sparse_layers = model.block_sparse_layers

        # Should have 4 layers * 2 (fc1 + fc2) = 8 sparse layers
        assert len(sparse_layers) == 8
        for layer in sparse_layers:
            assert isinstance(layer, CMSBlockLinear)

    def test_only_specified_layers_sparse(self):
        """Only layers in mlp_block_sparse_layers should use CMSBlockLinear."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(0, 2),  # Only layers 0 and 2
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC
        from titans_core.layers.block_sparse import CMSBlockLinear

        model = TitanMAC(config)
        sparse_layers = model.block_sparse_layers

        # Should have 2 layers * 2 (fc1 + fc2) = 4 sparse layers
        assert len(sparse_layers) == 4

        # Verify which layers are sparse
        for idx, layer in enumerate(model.layers):
            if idx in (0, 2):
                assert isinstance(layer.mlp.fc1, CMSBlockLinear)
                assert isinstance(layer.mlp.fc2, CMSBlockLinear)
            else:
                assert isinstance(layer.mlp.fc1, nn.Linear)
                assert isinstance(layer.mlp.fc2, nn.Linear)

    def test_empty_layers_tuple_means_no_sparse(self):
        """Empty layers tuple should result in no sparse layers."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(),  # Empty - no layers
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)
        sparse_layers = model.block_sparse_layers

        # Should have no sparse layers
        assert len(sparse_layers) == 0

    def test_single_layer_sparse(self):
        """Test that a single layer can be configured as sparse."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(1,),  # Only layer 1
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC
        from titans_core.layers.block_sparse import CMSBlockLinear

        model = TitanMAC(config)
        sparse_layers = model.block_sparse_layers

        # Should have 1 layer * 2 (fc1 + fc2) = 2 sparse layers
        assert len(sparse_layers) == 2

        # Verify only layer 1 is sparse
        for idx, layer in enumerate(model.layers):
            if idx == 1:
                assert isinstance(layer.mlp.fc1, CMSBlockLinear)
                assert isinstance(layer.mlp.fc2, CMSBlockLinear)
            else:
                assert isinstance(layer.mlp.fc1, nn.Linear)
                assert isinstance(layer.mlp.fc2, nn.Linear)


class TestPerLayerDensity:
    """T068: Test that different densities can be configured per layer."""

    def test_uniform_density_float(self):
        """Test that a single float density applies to all layers."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=2,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=0.75,  # Same for all layers
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)
        sparse_layers = model.block_sparse_layers

        # All layers should have density=0.75
        for layer in sparse_layers:
            assert layer.density == 0.75

    def test_per_layer_density_dict(self):
        """Test that per-layer density dict applies correct densities."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=3,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density={0: 0.25, 1: 0.5, 2: 0.75},
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)

        # Check each layer's density
        assert model.layers[0].mlp.fc1.density == 0.25
        assert model.layers[0].mlp.fc2.density == 0.25
        assert model.layers[1].mlp.fc1.density == 0.5
        assert model.layers[1].mlp.fc2.density == 0.5
        assert model.layers[2].mlp.fc1.density == 0.75
        assert model.layers[2].mlp.fc2.density == 0.75

    def test_per_layer_density_partial_dict(self):
        """Test that partial per-layer density dict uses default for missing layers."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density={0: 0.3, 2: 0.7},  # Missing layers 1, 3
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)

        # Layer 0: density 0.3
        assert model.layers[0].mlp.fc1.density == 0.3
        # Layer 1: default density 0.5
        assert model.layers[1].mlp.fc1.density == 0.5
        # Layer 2: density 0.7
        assert model.layers[2].mlp.fc1.density == 0.7
        # Layer 3: default density 0.5
        assert model.layers[3].mlp.fc1.density == 0.5

    def test_combined_layer_selection_and_density(self):
        """Test that layer selection and per-layer density work together."""
        config = TitanMACConfig(
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(0, 2),  # Only layers 0 and 2
            mlp_block_sparse_density={0: 0.25, 2: 0.75},  # Different densities
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC
        from titans_core.layers.block_sparse import CMSBlockLinear

        model = TitanMAC(config)

        # Layer 0: sparse with density 0.25
        assert isinstance(model.layers[0].mlp.fc1, CMSBlockLinear)
        assert model.layers[0].mlp.fc1.density == 0.25

        # Layer 1: dense
        assert isinstance(model.layers[1].mlp.fc1, nn.Linear)

        # Layer 2: sparse with density 0.75
        assert isinstance(model.layers[2].mlp.fc1, CMSBlockLinear)
        assert model.layers[2].mlp.fc1.density == 0.75

        # Layer 3: dense
        assert isinstance(model.layers[3].mlp.fc1, nn.Linear)


class TestBlockSparseForward:
    """Additional tests to verify block-sparse MLP works correctly in the model."""

    def test_forward_with_block_sparse_mlp(self):
        """Test that forward pass works with block-sparse MLP layers."""
        config = TitanMACConfig(
            vocab_size=1000,
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=2,
            max_seq_len=128,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=0.5,
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)
        model.eval()

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            output = model(input_ids)

        assert "logits" in output
        assert output["logits"].shape == (2, 32, 1000)

    def test_forward_with_mixed_sparse_dense(self):
        """Test forward pass with mixed sparse and dense layers."""
        config = TitanMACConfig(
            vocab_size=1000,
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=4,
            max_seq_len=128,
            mlp_use_block_sparse=True,
            mlp_block_sparse_layers=(0, 2),  # Only some layers sparse
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)
        model.eval()

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            output = model(input_ids)

        assert "logits" in output
        assert output["logits"].shape == (2, 32, 1000)

    def test_backward_with_block_sparse_mlp(self):
        """Test that backward pass works with block-sparse MLP layers."""
        config = TitanMACConfig(
            vocab_size=1000,
            d_model=64,
            d_ff=256,
            n_heads=4,
            n_layers=2,
            max_seq_len=128,
            mlp_use_block_sparse=True,
            mlp_block_sparse_density=0.5,
        )
        config.validate()

        from titans_core.models.titanmac import TitanMAC

        model = TitanMAC(config)
        model.train()

        # Test forward + backward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)

        assert "loss" in output
        output["loss"].backward()

        # Check gradients exist for block-sparse layers
        sparse_layers = model.block_sparse_layers
        for layer in sparse_layers:
            assert layer.values.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
