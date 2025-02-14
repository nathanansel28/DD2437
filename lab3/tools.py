from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(
        self, 
        n_nodes: int = 100, 
        max_epochs: int = 20
    ):
        self.n_nodes = n_nodes
        self.epochs = max_epochs
        self.weights = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self.p = self._init_p()


    def fit(
        self, 
        patterns: Optional[np.ndarray] = None, 
        pattern_indices: Optional[List[int]] = None
    ) -> None:
        """
        Fits a Hopefield Network using provided patterns or patterns_indices.

        Parameters
        ----------
        patterns (Optional[np.ndarray])
            Patterns for the user to directly provide to the network to memorize.
        patterns_indices (Optional[List[int]])
            List of indices to reference the attribute `p`
            For example, to fit patterns p1, p2, enter patterns_indices=[1,2]

        """
        patterns = self._load_pattern(patterns, pattern_indices)

        assert (
            patterns.shape[1] == self.weights.shape[0]
        ), f"Pattern dimension must be the same as number of nodes. Given {patterns.shape[1]} dimensions, expected {self.n_nodes} dimensions."

        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

        self.weights /= self.n_nodes


    def recall(self, inputs: np.ndarray) -> np.ndarray:
        previous = np.zeros(inputs.shape)

        for epoch in range(self.epochs):
            current = np.sign(inputs @ self.weights)
            current[current == 0] = 1

            if np.array_equal(current, previous):
                previous = current
                break

            previous = current

        return previous


    def visualize(
        self, pattern: np.ndarray = None, pattern_index: int = None
    ) -> None: 
        """Visualizes a 1024-bit pattern as a 32x32 image."""
        if pattern is None and pattern_index is None: 
            raise ValueError(f"Either pattern or pattern_index is required.")
        if pattern is None: 
            pattern = self.p[pattern_index]
        image = pattern.reshape(32, 32)

        plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
        plt.title("32x32 Image Visualization")
        plt.colorbar(label="Value (-1 or 1)")
        plt.show()


    def distort_patterns(
        self, 
        num_units: int,
        patterns: Optional[np.ndarray] = None, 
        pattern_indices: Optional[List[int]] = None,
        seed: Optional[Union[int, None]] = 20250214
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
            
        patterns = self._load_pattern(patterns, pattern_indices)
        distorted_patterns = []
        for pattern in patterns:
            if num_units > pattern.size:
                raise ValueError("num_units cannot be greater than the size of the pattern.")
            
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
        pattern_indices: Optional[List[int]] = None
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
            patterns = np.array([
                self.p[i] for i in pattern_indices
            ])
        return patterns


    def _init_p(self) -> Dict[int, np.ndarray]: 
        data = np.loadtxt('pict.dat', delimiter=',')
        assert len(data) == 1024 * 11, "The data length is not 1024 * 11."
        patterns = np.split(data, 11)
        return {i + 1: patterns[i] for i in range(11)}
    

    