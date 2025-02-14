from typing import Dict 
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


    def fit(self, patterns: np.ndarray) -> None:
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
            raise TypeError(f"Either pattern or pattern_inde is required.")
        if pattern is None: 
            pattern = self.p[pattern_index]
        image = pattern.reshape(32, 32)

        plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
        plt.title("32x32 Image Visualization")
        plt.colorbar(label="Value (-1 or 1)")
        plt.show()


    def _init_p(self) -> Dict[int, np.ndarray]: 
        data = np.loadtxt('pict.dat', delimiter=',')
        assert len(data) == 1024 * 11, "The data length is not 1024 * 11."
        patterns = np.split(data, 11)
        return {i + 1: patterns[i] for i in range(11)}
    

    