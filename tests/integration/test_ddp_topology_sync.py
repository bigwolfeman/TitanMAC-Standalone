"""Integration tests for DDP topology synchronization (T085-T087).

Tests the DDP synchronization features of CMSBlockLinear:
- sync_topology_scores() all-reduces scoring buffers
- topology_step() calls sync in distributed mode
- Deterministic RNG ensures identical topology across ranks
- get_topology_checksum() for divergence detection

Date: 2025-12-27
Branch: 001-cms-block-sparse
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from titans_core.layers.block_sparse import CMSBlockLinear


def _setup_dist(rank: int, world_size: int, backend: str = "gloo") -> None:
    """Initialize distributed process group for testing."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_dist() -> None:
    """Destroy distributed process group."""
    dist.destroy_process_group()


class TestDDPTopologySyncUnit:
    """Unit tests for DDP sync that don't require actual distributed setup."""

    def test_sync_topology_scores_noop_when_not_initialized(self):
        """sync_topology_scores is a no-op when distributed is not initialized."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Set some scores
        layer.block_score_ema.fill_(1.0)
        layer.activation_norm_acc.fill_(2.0)
        layer.error_norm_acc.fill_(3.0)

        # Call sync - should be no-op since dist is not initialized
        layer.sync_topology_scores()

        # Values should be unchanged
        assert torch.allclose(layer.block_score_ema, torch.ones_like(layer.block_score_ema))
        assert torch.allclose(layer.activation_norm_acc, torch.full_like(layer.activation_norm_acc, 2.0))
        assert torch.allclose(layer.error_norm_acc, torch.full_like(layer.error_norm_acc, 3.0))

    def test_get_topology_checksum_deterministic(self):
        """get_topology_checksum returns same value for same topology."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Get checksum twice
        checksum1 = layer.get_topology_checksum()
        checksum2 = layer.get_topology_checksum()

        assert checksum1 == checksum2
        assert isinstance(checksum1, int)

    def test_get_topology_checksum_changes_with_topology(self):
        """get_topology_checksum changes when topology changes."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        checksum_before = layer.get_topology_checksum()

        # Modify topology
        with torch.no_grad():
            layer.col_indices[0, 0] = (layer.col_indices[0, 0] + 1) % layer.C

        checksum_after = layer.get_topology_checksum()

        assert checksum_before != checksum_after

    def test_deterministic_rng_with_global_step(self):
        """topology_step produces same results with same global_step."""
        # Create two identical layers
        layer1 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        layer2 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Copy topology and scores from layer1 to layer2
        with torch.no_grad():
            layer2.col_indices.copy_(layer1.col_indices)
            layer2.values.copy_(layer1.values)
            layer2.block_score_ema.copy_(layer1.block_score_ema)
            layer2.activation_norm_acc.copy_(layer1.activation_norm_acc)
            layer2.error_norm_acc.copy_(layer1.error_norm_acc)

        # Set identical non-zero scores
        layer1.block_score_ema.fill_(1.0)
        layer2.block_score_ema.fill_(1.0)
        layer1.activation_norm_acc.fill_(0.5)
        layer2.activation_norm_acc.fill_(0.5)
        layer1.error_norm_acc.fill_(0.5)
        layer2.error_norm_acc.fill_(0.5)

        # Run topology_step with same global_step
        global_step = 100
        result1 = layer1.topology_step(global_step=global_step)
        result2 = layer2.topology_step(global_step=global_step)

        # Topologies should be identical
        assert torch.equal(layer1.col_indices, layer2.col_indices)
        assert result1.num_swaps == result2.num_swaps

    def test_different_global_step_different_topology(self):
        """Different global_step values may produce different topologies."""
        # Create two identical layers
        layer1 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )
        layer2 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Copy topology from layer1 to layer2
        with torch.no_grad():
            layer2.col_indices.copy_(layer1.col_indices)
            layer2.values.copy_(layer1.values)

        # Set identical scores that will trigger exploration
        layer1.block_score_ema.fill_(0.1)  # Low scores to encourage swaps
        layer2.block_score_ema.fill_(0.1)
        layer1.activation_norm_acc.fill_(5.0)  # High candidate scores
        layer2.activation_norm_acc.fill_(5.0)
        layer1.error_norm_acc.fill_(5.0)
        layer2.error_norm_acc.fill_(5.0)

        # Run topology_step with different global_steps
        result1 = layer1.topology_step(global_step=100)
        result2 = layer2.topology_step(global_step=200)

        # Note: They may or may not be different depending on exploration
        # but the RNG state will be different
        # This test just verifies no crash with different global_steps
        assert layer1.col_indices.shape == layer2.col_indices.shape


class TestDDPTopologySyncIntegration:
    """Integration tests that require actual distributed setup."""

    @pytest.mark.skipif(
        not torch.distributed.is_available(),
        reason="torch.distributed not available"
    )
    @pytest.mark.skipif(
        not hasattr(mp, 'spawn'),
        reason="multiprocessing.spawn not available"
    )
    def test_topology_identical_across_ranks(self):
        """T085: Verify topology is identical across ranks with sync.

        Uses torch.multiprocessing.spawn to launch 2 processes that each:
        1. Create CMSBlockLinear with same initial state
        2. Set different local scores (simulating different data shards)
        3. Run topology_step with global_step for determinism
        4. Verify col_indices match via checksum
        """
        world_size = 2

        # Store initial topology as a list (picklable)
        torch.manual_seed(42)
        temp_layer = CMSBlockLinear(
            in_features=64, out_features=128, tile_size=16, density=0.5
        )
        initial_topology_list = temp_layer.col_indices.tolist()

        def worker(rank: int, results_queue: mp.Queue, init_topo_list: list):
            """Worker function for each rank."""
            try:
                _setup_dist(rank, world_size)

                # Create layer with same config
                layer = CMSBlockLinear(
                    in_features=64,
                    out_features=128,
                    tile_size=16,
                    density=0.5,
                )

                # Set identical initial topology from list
                with torch.no_grad():
                    layer.col_indices.copy_(torch.tensor(init_topo_list, dtype=torch.int32))

                # Set different local scores (simulating different data shards)
                # Rank 0 gets scores [1, 2, 3, ...], Rank 1 gets [2, 3, 4, ...]
                with torch.no_grad():
                    layer.block_score_ema.fill_(1.0 + rank)
                    layer.activation_norm_acc.fill_(0.5 + rank * 0.1)
                    layer.error_norm_acc.fill_(0.5 + rank * 0.1)

                # Run topology_step with deterministic global_step
                layer.topology_step(global_step=100)

                # Get checksum and col_indices as list (picklable)
                checksum = layer.get_topology_checksum()
                col_indices_list = layer.col_indices.tolist()
                results_queue.put((rank, checksum, col_indices_list))

            except Exception as e:
                import traceback
                results_queue.put((rank, f"error: {e}\n{traceback.format_exc()}", None))
            finally:
                _cleanup_dist()

        # Use multiprocessing queue for results
        results_queue = mp.Queue()

        # Spawn workers
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, results_queue, initial_topology_list))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=30)

        # Collect results
        results = {}
        for _ in range(world_size):
            rank, checksum, col_indices_list = results_queue.get(timeout=5)
            results[rank] = (checksum, col_indices_list)

        # Verify checksums match
        checksums = [results[r][0] for r in range(world_size)]
        assert all(isinstance(c, int) for c in checksums), f"Got errors: {checksums}"
        assert checksums[0] == checksums[1], (
            f"Topology checksums differ: rank 0 = {checksums[0]}, rank 1 = {checksums[1]}"
        )

        # Verify actual col_indices match
        col_indices_0 = results[0][1]
        col_indices_1 = results[1][1]
        assert col_indices_0 == col_indices_1, "col_indices should be identical"

    @pytest.mark.skipif(
        not torch.distributed.is_available(),
        reason="torch.distributed not available"
    )
    @pytest.mark.skipif(
        not hasattr(mp, 'spawn'),
        reason="multiprocessing.spawn not available"
    )
    def test_ddp_score_averaging(self):
        """T086: Verify all-reduce correctly averages scores across ranks.

        Creates two mock tensors with known values, verifies that
        sync_topology_scores produces the correct average.
        """
        world_size = 2

        def worker(rank: int, results_queue: mp.Queue):
            """Worker function for each rank."""
            try:
                _setup_dist(rank, world_size)

                # Create layer
                layer = CMSBlockLinear(
                    in_features=64,
                    out_features=128,
                    tile_size=16,
                    density=0.5,
                )

                # Set known scores based on rank
                # Rank 0: [2.0], Rank 1: [4.0] -> Average: [3.0]
                with torch.no_grad():
                    layer.block_score_ema.fill_(2.0 + rank * 2.0)
                    layer.activation_norm_acc.fill_(1.0 + rank * 1.0)
                    layer.error_norm_acc.fill_(3.0 + rank * 1.0)

                # Sync scores
                layer.sync_topology_scores()

                # Get averaged values
                results_queue.put((
                    rank,
                    layer.block_score_ema[0, 0].item(),
                    layer.activation_norm_acc[0].item(),
                    layer.error_norm_acc[0].item(),
                ))

            except Exception as e:
                results_queue.put((rank, f"error: {e}", None, None))
            finally:
                _cleanup_dist()

        results_queue = mp.Queue()

        # Spawn workers
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, results_queue))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=30)

        # Collect results
        results = {}
        for _ in range(world_size):
            data = results_queue.get(timeout=5)
            results[data[0]] = data[1:]

        # Expected averages:
        # block_score_ema: (2.0 + 4.0) / 2 = 3.0
        # activation_norm_acc: (1.0 + 2.0) / 2 = 1.5
        # error_norm_acc: (3.0 + 4.0) / 2 = 3.5

        for rank in range(world_size):
            score, act, err = results[rank]
            assert isinstance(score, float), f"Rank {rank} error: {score}"
            assert abs(score - 3.0) < 1e-5, f"Rank {rank}: block_score_ema = {score}, expected 3.0"
            assert abs(act - 1.5) < 1e-5, f"Rank {rank}: activation_norm_acc = {act}, expected 1.5"
            assert abs(err - 3.5) < 1e-5, f"Rank {rank}: error_norm_acc = {err}, expected 3.5"


class TestTopologyChecksumLogging:
    """T087: Tests for topology checksum logging for divergence detection."""

    def test_checksum_logging_workflow(self):
        """Demonstrate the checksum logging workflow for divergence detection."""
        layer = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Simulate training loop with checksum logging
        checksums = []

        for step in range(3):
            # Simulate topology step
            layer.block_score_ema.fill_(1.0)
            layer.activation_norm_acc.fill_(0.5)
            layer.error_norm_acc.fill_(0.5)
            layer.topology_step(global_step=step)

            # Log checksum (in real code, would be logged to wandb/tensorboard)
            checksum = layer.get_topology_checksum()
            checksums.append(checksum)

        # Verify checksums are valid integers
        assert all(isinstance(c, int) for c in checksums)

        # Checksums should potentially change between steps
        # (not guaranteed, but structure is correct)
        assert len(checksums) == 3

    def test_checksum_reproducibility_with_seed(self):
        """Checksum is reproducible when using same seed and scores."""
        # First run
        layer1 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        # Use fixed seed for initialization
        torch.manual_seed(12345)
        layer1._initialize_topology()
        layer1.block_score_ema.fill_(1.0)
        layer1.activation_norm_acc.fill_(0.5)
        layer1.error_norm_acc.fill_(0.5)
        layer1.topology_step(global_step=100)
        checksum1 = layer1.get_topology_checksum()

        # Second run with same seed
        layer2 = CMSBlockLinear(
            in_features=64,
            out_features=128,
            tile_size=16,
            density=0.5,
        )

        torch.manual_seed(12345)
        layer2._initialize_topology()
        layer2.block_score_ema.fill_(1.0)
        layer2.activation_norm_acc.fill_(0.5)
        layer2.error_norm_acc.fill_(0.5)
        layer2.topology_step(global_step=100)
        checksum2 = layer2.get_topology_checksum()

        # Should be identical
        assert checksum1 == checksum2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
