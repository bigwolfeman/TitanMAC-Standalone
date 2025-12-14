"""
Unit tests for DeepNestedOptimizer.

Tests the complete Nested Learning optimizer including:
- Basic optimization step
- LR multiplier adaptation
- Meta-learning updates
- CMS state management
- Checkpointing (state_dict / load_state_dict)
"""

import pytest
import torch
import torch.nn as nn

from titans_core.opt import (
    DeepNestedOptimizer,
    L2RegressionMomentum,
    ContinuumMemoryState,
    UnrolledMetaTrainer,
    SimplifiedMetaTrainer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return {'loss': self.fc2(x).sum(), 'logits': self.fc2(x)}


class TestL2RegressionMomentum:
    """Tests for L2RegressionMomentum module."""

    def test_initialization(self):
        """Test L2RegressionMomentum initializes correctly."""
        mlp = L2RegressionMomentum(hidden_dim=64)
        assert mlp.hidden_dim == 64
        assert mlp.num_layers == 2

    def test_forward_shape(self):
        """Test forward pass produces correct shapes."""
        mlp = L2RegressionMomentum(hidden_dim=32)

        grad = torch.randn(100)  # Flattened gradient
        prev_momentum = torch.randn(100)
        context = torch.tensor([0.1, 0.001, 1.0])  # step, lr, loss

        scale, shift, damping = mlp(grad, prev_momentum, context)

        assert scale.dim() == 0  # Scalar
        assert shift.dim() == 0
        assert damping.dim() == 0

    def test_output_bounds(self):
        """Test outputs are within expected bounds."""
        mlp = L2RegressionMomentum(hidden_dim=32)

        for _ in range(10):
            grad = torch.randn(50)
            prev_momentum = torch.randn(50)
            context = torch.rand(3)

            scale, shift, damping = mlp(grad, prev_momentum, context)

            assert 0.5 <= scale.item() <= 1.5, f"Scale {scale} out of bounds [0.5, 1.5]"
            assert -1.0 <= shift.item() <= 1.0, f"Shift {shift} out of bounds [-1, 1]"
            assert 0.0 <= damping.item() <= 1.0, f"Damping {damping} out of bounds [0, 1]"

    def test_internal_loss(self):
        """Test internal loss computation."""
        mlp = L2RegressionMomentum()
        predicted = torch.randn(10)
        actual = torch.randn(10)

        loss = mlp.compute_internal_loss(predicted, actual)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # MSE is non-negative


class TestContinuumMemoryState:
    """Tests for ContinuumMemoryState (CMS)."""

    def test_initialization(self):
        """Test CMS state initializes correctly."""
        state = ContinuumMemoryState(
            param_shape=(10, 5),
            num_levels=3,
            base_frequency=10,
        )

        assert state.num_levels == 3
        assert len(state.levels) == 3
        assert state.levels[0]['frequency'] == 1
        assert state.levels[1]['frequency'] == 10
        assert state.levels[2]['frequency'] == 100

    def test_should_update(self):
        """Test update scheduling logic."""
        state = ContinuumMemoryState(
            param_shape=(10,),
            num_levels=3,
            base_frequency=10,
        )

        # Level 0 (freq=1) should always update
        assert state.should_update(0, 0) is True
        assert state.should_update(0, 1) is True
        assert state.should_update(0, 99) is True

        # Level 1 (freq=10) should update every 10 steps
        assert state.should_update(1, 0) is True
        assert state.should_update(1, 5) is False
        assert state.should_update(1, 10) is True
        assert state.should_update(1, 20) is True

        # Level 2 (freq=100) should update every 100 steps
        assert state.should_update(2, 0) is True
        assert state.should_update(2, 50) is False
        assert state.should_update(2, 100) is True

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        state = ContinuumMemoryState(param_shape=(5,), num_levels=2)

        grad1 = torch.ones(5)
        grad2 = torch.ones(5) * 2

        state.accumulate_grad(grad1)
        state.accumulate_grad(grad2)

        # Check accumulated in all levels
        for level in state.levels.values():
            assert torch.allclose(level['accumulated_grad'], torch.ones(5) * 3)


class TestDeepNestedOptimizer:
    """Tests for the main DeepNestedOptimizer class."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        return DeepNestedOptimizer(
            model=model,
            base_lr=1e-3,
            mode='simple',
            meta_update_freq=10,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer.base_lr == 1e-3
        assert optimizer.mode == 'simple'
        assert optimizer.n_groups == 2
        assert optimizer.global_step == 0

    def test_step_increments_counter(self, model, optimizer):
        """Test step increments global_step."""
        x = torch.randn(4, 10)
        output = model(x)
        output['loss'].backward()

        result = optimizer.step(output['loss'].item())

        assert result['global_step'] == 1
        assert optimizer.global_step == 1

    def test_lr_multipliers_change(self, model, optimizer):
        """Test LR multipliers change over training."""
        initial_mults = optimizer.get_lr_multipliers().clone()

        # Run several steps
        for _ in range(20):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            output = model(x)
            output['loss'].backward()
            optimizer.step(output['loss'].item())

        final_mults = optimizer.get_lr_multipliers()

        # Multipliers should have changed
        assert not torch.allclose(initial_mults, final_mults)

    def test_effective_lrs(self, optimizer):
        """Test effective LR computation."""
        lrs = optimizer.get_effective_lrs()

        assert len(lrs) == 2
        for lr in lrs:
            assert lr > 0

    def test_state_dict_save_load(self, model, optimizer):
        """Test checkpointing via state_dict."""
        # Run a few steps
        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            output = model(x)
            output['loss'].backward()
            optimizer.step(output['loss'].item())

        # Save state
        state = optimizer.state_dict()

        # Create new optimizer
        new_optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=1e-3,
            mode='simple',
        )

        # Load state
        new_optimizer.load_state_dict(state)

        assert new_optimizer.global_step == optimizer.global_step
        assert torch.allclose(
            new_optimizer.get_lr_multipliers(),
            optimizer.get_lr_multipliers()
        )

    def test_zero_grad(self, model, optimizer):
        """Test zero_grad clears gradients."""
        x = torch.randn(4, 10)
        output = model(x)
        output['loss'].backward()

        # Gradients should exist
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0
                break

        optimizer.zero_grad()

        # Gradients should be None or zero
        for p in model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


class TestSimplifiedMetaTrainer:
    """Tests for SimplifiedMetaTrainer."""

    def test_record_step(self):
        """Test step recording."""
        trainer = SimplifiedMetaTrainer(window_size=5)

        trainer.record_step(1.0, torch.ones(2), 0.5)
        trainer.record_step(0.9, torch.ones(2), 0.4)

        assert len(trainer.loss_history) == 2
        assert len(trainer.multiplier_history) == 2

    def test_window_size_limit(self):
        """Test history is limited to window_size."""
        trainer = SimplifiedMetaTrainer(window_size=3)

        for i in range(5):
            trainer.record_step(float(i), torch.ones(2), 0.5)

        assert len(trainer.loss_history) == 3
        assert trainer.loss_history == [2.0, 3.0, 4.0]

    def test_proxy_loss_with_insufficient_history(self):
        """Test proxy loss returns regularization when not enough history."""
        trainer = SimplifiedMetaTrainer(window_size=20)

        # Only 5 steps (less than 10 required)
        for i in range(5):
            trainer.record_step(float(i), torch.ones(2), 0.5)

        current_mults = torch.tensor([1.1, 0.9])
        loss = trainer.compute_proxy_loss(current_mults, 0.5)

        # Should be (mult - 1.0)^2
        expected = ((current_mults - 1.0) ** 2).mean()
        assert torch.isclose(loss, expected)


class TestUnrolledMetaTrainer:
    """Tests for UnrolledMetaTrainer."""

    def test_initialization(self):
        """Test UnrolledMetaTrainer initializes correctly."""
        trainer = UnrolledMetaTrainer(k_steps=5)
        assert trainer.k_steps == 5
        assert trainer.use_checkpointing is True


class TestIntegration:
    """Integration tests for the full optimizer pipeline."""

    def test_training_reduces_loss(self):
        """Test that training actually reduces loss over time."""
        model = SimpleModel()
        optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=1e-2,  # Higher LR for faster convergence in test
            mode='simple',
            meta_update_freq=5,
        )

        # Target: minimize loss on fixed input
        x = torch.randn(16, 10)
        target = torch.zeros(16, 5)

        initial_loss = None
        final_loss = None

        for step in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output['logits'], target)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step(loss.item())

            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
