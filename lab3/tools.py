from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SEED_VALUE = 20250214


class HopfieldNetwork:
    def __init__(
        self,
        n_nodes: int = 100,
        max_epochs: int = 20, 
        sequential: bool = False,
        remove_self_connections: bool = False
    ):
        self.n_nodes = n_nodes
        self.epochs = max_epochs
        self.weights = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self.p = self._init_p()
        self.detailed_history = []
        self.step_function = (
            self._sequential_update if sequential else self._batch_update
        )
        self.sequential = sequential
        self.remove_self_connections = remove_self_connections

    def fit(
        self,
        patterns: Optional[np.ndarray] = None,
        pattern_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Fits a Hopefield Network using `patterns` or `patterns_indices`.

        Parameters
        ----------
        patterns (Optional[np.ndarray])
            Patterns for the user to directly provide to the network to memorize.
        patterns_indices (Optional[List[int]])
            List of indices to reference the attribute `p to provide the patterns.
            For example, to fit patterns p1, p2, enter patterns_indices=[1,2]

        """
        patterns = self._load_pattern(patterns, pattern_indices=pattern_indices)

        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

        self.weights /= self.n_nodes
        if self.remove_self_connections: 
            np.fill_diagonal(self.weights, 0)

    def recall(self, inputs: np.ndarray) -> np.ndarray:
        """
        Recall method supporting both sequential and batch updates with continuous history tracking.
        """
        self.detailed_history = []

        previous = inputs.copy()
        current_step = len(self.detailed_history)

        if self.remove_self_connections: 
            np.fill_diagonal(self.weights, 0)


        for epoch in range(self.epochs):
            # Record state at start of epoch
            self._record_state(previous, epoch, current_step)
            current_step += 1

            # Perform update based on mode
            current = self.step_function(previous, current_step)

            # Check for convergence
            if np.array_equal(current, previous):
                break

            previous = current.copy()
            current_step = len(self.detailed_history)

        return previous

    def _batch_update(self, inputs: np.ndarray, current_step: int) -> np.ndarray:
        """
        Batch update with history tracking.
        """
        # Calculate predictions for all units at once
        prediction = inputs @ self.weights
        prediction = np.where(prediction >= 0, 1, -1)

        return prediction

    def _sequential_update(self, inputs: np.ndarray, current_step: int) -> np.ndarray:
        """
        Updates units sequentially in random order using optimized NumPy operations.

        Parameters
        ----------
        inputs : np.ndarray
            Input patterns of shape (n_patterns, n_nodes)

        Returns
        -------
        np.ndarray
            Updated patterns after sequential updates
        """
        prediction = np.copy(inputs)
        n_patterns = inputs.shape[0]

        # For each pattern
        for pattern_idx in range(n_patterns):
            # Generate random order of units to update
            unit_order = np.random.permutation(self.n_nodes)

            # Create a mask to zero out self-connections
            mask = np.ones(self.n_nodes)

            # Update each unit sequentially, but use vectorized operations
            for unit_idx in unit_order:
                # Zero out self-connection for current unit
                mask[unit_idx] = 0

                # Calculate input to the current unit using vectorized operations
                # Multiply current states by weights and mask, then sum
                unit_input = np.sum(
                    self.weights[unit_idx] * prediction[pattern_idx] * mask
                )

                # Update the unit's state
                prediction[pattern_idx, unit_idx] = np.where(unit_input >= 0, 1, -1)

                # Restore mask for next iteration
                mask[unit_idx] = 1

                self.detailed_history.append(
                    {
                        "step": current_step,
                        "epoch": current_step // (self.n_nodes * n_patterns),
                        "pattern_idx": pattern_idx,
                        "update_type": "sequential",
                        "unit_updated": unit_idx,
                        "state": prediction.copy(),
                    }
                )
                current_step += 1

        return prediction

    def _record_state(self, states: np.ndarray, epoch: int, current_step: int) -> None:
        """
        Helper method to record network states.
        """
        for pattern_idx in range(len(states)):
            self.detailed_history.append(
                {
                    "step": current_step,
                    "epoch": epoch,
                    "pattern_idx": pattern_idx,
                    "update_type": "state_recording",
                    "state": states[pattern_idx].copy(),
                }
            )

    def evaluate(
        self,
        patterns_pred: np.ndarray,
        patterns: Optional[np.ndarray] = None,
        pattern_indices: Optional[List[int]] = None,
    ) -> List[bool]:
        """
        Evaluates `pattern predictions` based on `patterns`.

        Parameters
        ----------
        patterns_pred (np.ndarray)
            Predicted patterns.
        patterns (Optional[np.ndarray])
            Actual patterns for the user to directly provide.
        patterns_indices (Optional[List[int]])
            List of indices to reference the attribute `p` to provide the actual patterns.
            For example, to provide patterns p1, p2, enter patterns_indices=[1,2]
        """

        patterns = self._load_pattern(patterns, pattern_indices=pattern_indices)
        assert patterns_pred.shape == patterns.shape

        return [
            np.array_equal(pattern_pred, patterns[i])
            for i, pattern_pred in enumerate(patterns_pred)
        ]

    def fit_incremental(self, patterns: np.ndarray) -> List[int]:
        """
        Designed for question 3.5.2.
        Incrementally adds patterns to the network and checks the stability after each addition.

        Parameters
        ----------
        patterns (np.ndarray)
            Array of patterns to be added sequentially.

        Returns
        -------
        stable_count_per_step (List[int])
            Number of stable patterns after each addition.
        """
        stable_count_per_step = []

        for i, pattern in enumerate(patterns):
            self.weights += np.outer(pattern, pattern)
            if self.remove_self_connections: 
                np.fill_diagonal(self.weights, 0)

            stable_count = sum(self._is_stable(p) for p in patterns[: i + 1])
            stable_count_per_step.append(stable_count)
        self.weights /= self.n_nodes

        return stable_count_per_step

    def visualizex_pattern(
        self, pattern: np.ndarray = None, pattern_index: int = None
    ) -> None:
        """Visualizes a 1024-bit pattern as a 32x32 image."""
        if pattern is None and pattern_index is None:
            raise ValueError(f"Either pattern or pattern_index is required.")
        if pattern is None:
            pattern = self.p[pattern_index]
        image = pattern.reshape(32, 32)

        plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
        plt.title("32x32 Image Visualization")
        plt.colorbar(label="Value (-1 or 1)")
        plt.show()

    def distort_patterns(
        self,
        num_units: int,
        patterns: Optional[np.ndarray] = None,
        pattern_indices: Optional[List[int]] = None,
        seed: Optional[Union[int, None]] = DEFAULT_SEED_VALUE,
    ) -> np.ndarray:
        """
        Creates a moderate distortion in a given pattern by permutating a specified number of units.

        Parameters
        ---------
        num_units (int)
            The number of indices to shuffle in the pattern.
        patterns (Optional[np.ndarray])
            Patterns for the user to directly provide to distort.
        patterns_indices (Optional[List[int]])
            List of indices to reference the attribute `p` to distort.
            For example, to distort patterns p1, p2, enter patterns_indices=[1,2].
        seed (Optional[Union[int, None]])
            Seed for reproducibility. Set to `None` to disable.

        Returns
        -------
        distored_pattern (np.ndarray)
            The distorted pattern with num_units shuffled.
        """
        if seed is not None:
            np.random.seed(seed)

        patterns = self._load_pattern(patterns, pattern_indices=pattern_indices)
        distorted_patterns = []
        for pattern in patterns:
            if num_units > pattern.size:
                raise ValueError(
                    "num_units cannot be greater than the size of the pattern."
                )

            distorted_pattern = pattern.copy()
            indices = np.random.choice(pattern.size, size=num_units, replace=False)
            shuffled_values = distorted_pattern[indices]
            np.random.shuffle(shuffled_values)
            distorted_pattern[indices] = shuffled_values
            distorted_patterns.append(distorted_pattern)

        return np.array(distorted_patterns)

    """
    ======================================
    HELPER FUNCTIONS TO BE USED INTERNALLY
    ======================================
    """

    def _load_pattern(
        self,
        patterns: Union[np.ndarray, None],
        pattern_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Helper function to load the patterns.
        If patterns is provided, return patterns.
        If patterns is not provided, return patterns in p according to pattern_indices.

        Examples
        --------
        >>> _load_pattern(np.ndarray[[1, 2]])
        array([1, 2])
        >>> _load_pattern(None, [1])
        array([-1., -1., -1., ..., -1., -1., -1.]) -> this corresponds to p1.
        """

        if patterns is None and pattern_indices is None:
            raise ValueError(
                "Either 'patterns' or 'patterns_indices' must be provided."
            )
        if patterns is None:
            patterns = np.array([self.p[i] for i in pattern_indices])
        return patterns

    def _init_p(self) -> Dict[int, np.ndarray]:
        data = np.loadtxt("pict.dat", delimiter=",")
        assert len(data) == 1024 * 11, "The data length is not 1024 * 11."
        patterns = np.split(data, 11)
        return {i + 1: patterns[i] for i in range(11)}

    def _is_stable(self, pattern: np.ndarray) -> bool:
        """
        Checks if a pattern is stable (does not change after one iteration).

        Parameters
        ----------
        pattern (np.ndarray)
            The input pattern to check.

        Returns
        -------
        bool
            True if the pattern is stable, False otherwise.
        """
        updated_pattern = np.where(self.weights @ pattern > 0, 1, -1)
        return np.array_equal(updated_pattern, pattern)


"""
======================
OTHER HELPER FUNCTIONS
======================
"""


def generate_random_patterns(
    num_patterns: int,
    pattern_size: int = 1024,
    seed: Optional[Union[int, None]] = DEFAULT_SEED_VALUE,
) -> np.ndarray:
    """
    Generates an array of num_patterns NumPy arrays, each containing 1024 random values of -1 or 1 by default.

    Parameters
    ----------
    num_patterns (int)
        Number of patterns to generate.
    seed (Optional[int])
        Seed for reproducibility. Set to `None` TO DISABLE.

    Returns
    -------
    np.ndarray
        An array of shape (num_patterns, pattern_size) filled with random -1 or 1.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))


# class HopfieldNetwork_6(HopfieldNetwork):
 
class HopfieldNetwork_6(HopfieldNetwork):
    def __init__(
        self, 
        n_nodes = 100, 
        max_epochs = 20, 
        remove_self_connections = False
    ):
        super().__init__(n_nodes, max_epochs, remove_self_connections)

    