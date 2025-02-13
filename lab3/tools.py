import numpy as np


class HopfieldNetwork:
    def __init__(self, n_nodes: int = 100, max_epochs: int = 20):
        self.n_nodes = n_nodes
        self.epochs = max_epochs
        self.weights = np.zeros((n_nodes, n_nodes), dtype=np.float64)

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
