"""
Continuum Memory System (CMS) from Nested Learning paper.

This module implements multi-frequency parameter updates, allowing
different parameter groups to update at different intervals.

Key concept: Different parts of the model (core weights vs memory
weights) may benefit from different update frequencies.

Example configuration:
    frequencies = {
        0: 1,   # Core params: update every step
        1: 10,  # Memory params: update every 10 steps
        2: 50,  # Persistent tokens: update every 50 steps
    }

Version: 1.0.0
"""

from typing import Any, Dict, List


class ContinuumMemorySystem:
    """
    Multi-frequency parameter update scheduler.

    Tracks which parameter groups should update on each step,
    enabling hierarchical learning with different timescales.

    Args:
        param_groups: List of parameter group dicts (from optimizer)
        frequencies: Mapping of group index to update frequency
            - e.g., {0: 1, 1: 10} means group 0 updates every step,
              group 1 updates every 10 steps

    Raises:
        ValueError: If frequency <= 0 for any group
        KeyError: If group index doesn't exist in param_groups

    Example:
        >>> param_groups = optimizer.param_groups
        >>> cms = ContinuumMemorySystem(
        ...     param_groups=param_groups,
        ...     frequencies={0: 1, 1: 10, 2: 50}
        ... )
        >>> for step in range(100):
        ...     groups_to_update = cms.step()
        ...     # Only update specified groups
    """

    def __init__(
        self,
        param_groups: List[Dict],
        frequencies: Dict[int, int],
    ):
        # Validate frequencies
        for group_idx, freq in frequencies.items():
            if not isinstance(group_idx, int) or group_idx < 0:
                raise ValueError(
                    f"Group index must be non-negative int, got {group_idx}"
                )
            if not isinstance(freq, int) or freq <= 0:
                raise ValueError(
                    f"Frequency must be positive int, got {freq} for group {group_idx}"
                )

        self.param_groups = param_groups
        self.frequencies = frequencies
        self.n_groups = len(param_groups)

        # Initialize step counters for each group
        self.step_counters: Dict[int, int] = {i: 0 for i in range(self.n_groups)}

        # Global step counter
        self.global_step = 0

    def should_update(self, group_idx: int) -> bool:
        """
        Check if a parameter group should update on current step.

        Args:
            group_idx: Index of parameter group

        Returns:
            True if group should update (global_step % frequency == 0)

        Contract:
            - Returns True on step 0 for all groups (initial update)
            - Groups with frequency=1 always return True
            - Groups with frequency=N return True every N steps
        """
        if group_idx < 0 or group_idx >= self.n_groups:
            return False

        # Get frequency for this group (default: 1 = every step)
        freq = self.frequencies.get(group_idx, 1)

        # Check if current step is a multiple of frequency
        return self.global_step % freq == 0

    def step(self) -> List[int]:
        """
        Increment global step and return groups to update.

        Returns:
            List of group indices that should update this step

        Contract:
            - Increments global_step
            - Updates step_counters for groups that update
            - Returns list of group indices to update
        """
        groups_to_update = []

        for idx in range(self.n_groups):
            if self.should_update(idx):
                groups_to_update.append(idx)
                self.step_counters[idx] += 1

        # Increment global step AFTER checking (so step 0 triggers all groups)
        self.global_step += 1

        return groups_to_update

    def get_update_counts(self) -> Dict[int, int]:
        """
        Get cumulative update counts per group.

        Returns:
            Dict mapping group index to total updates performed
        """
        return dict(self.step_counters)

    def get_frequencies(self) -> Dict[int, int]:
        """
        Get frequency configuration.

        Returns:
            Dict mapping group index to update frequency
        """
        return dict(self.frequencies)

    def set_frequency(self, group_idx: int, frequency: int) -> None:
        """
        Update frequency for a group.

        Args:
            group_idx: Index of parameter group
            frequency: New update frequency (must be > 0)

        Raises:
            ValueError: If frequency <= 0 or group_idx invalid
        """
        if group_idx < 0 or group_idx >= self.n_groups:
            raise ValueError(f"Invalid group index {group_idx}, have {self.n_groups} groups")
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")

        self.frequencies[group_idx] = frequency

    def reset(self) -> None:
        """Reset all counters to initial state."""
        self.global_step = 0
        self.step_counters = {i: 0 for i in range(self.n_groups)}

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing.

        State Contents:
            - frequencies: Frequency configuration
            - step_counters: Per-group update counts
            - global_step: Total steps
        """
        return {
            "frequencies": dict(self.frequencies),
            "step_counters": dict(self.step_counters),
            "global_step": self.global_step,
            "n_groups": self.n_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary from state_dict()
        """
        if "frequencies" in state_dict:
            self.frequencies = dict(state_dict["frequencies"])
        if "step_counters" in state_dict:
            self.step_counters = dict(state_dict["step_counters"])
        if "global_step" in state_dict:
            self.global_step = state_dict["global_step"]

    def __repr__(self) -> str:
        """String representation."""
        freqs = ", ".join(f"g{k}:{v}" for k, v in sorted(self.frequencies.items()))
        return f"ContinuumMemorySystem(n_groups={self.n_groups}, frequencies={{{freqs}}})"
