from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.random import default_rng


def manhattan_distance(target_point: np.ndarray, all_points: np.ndarray) -> np.ndarray:
    """
    Calculates the Manhattan distance between a target point and multiple other points.

    Parameters:
        target_point (np.ndarray): The reference point.
        all_points (np.ndarray): An array of points to calculate the distance to.

    Returns:
        np.ndarray: Array of Manhattan distances.
    """
    return np.sum(np.abs(all_points - target_point), axis=1)


def circular_manhattan_distance(
    target_point: np.ndarray, all_points: np.ndarray
) -> np.ndarray:
    """
    Calculates the Manhattan distance with wrap-around (circular) behavior.

    Parameters:
        target_point (np.ndarray): The reference point.
        all_points (np.ndarray): An array of points to calculate the distance to.

    Returns:
        np.ndarray: Array of circular Manhattan distances.
    """
    size_x, size_y = np.max(all_points, axis=0) + 1  # Determine grid dimensions

    dx = np.abs(all_points[:, 0] - target_point[0])  # Horizontal distance
    dy = np.abs(all_points[:, 1] - target_point[1])  # Vertical distance

    # Apply wrap-around distance calculation
    dx = np.minimum(dx, size_x - dx)
    dy = np.minimum(dy, size_y - dy)

    return dx + dy


class KohonenSOM:
    """
    Kohonen Self-Organizing Map (SOM) class for clustering and visualization.

    Parameters:
        n_features (int): Number of features in the input data.
        output_shape (Tuple[int, int]): Dimensions of the SOM grid.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        eta (float, optional): Learning rate. Defaults to 0.2.
        k0 (int, optional): Initial neighborhood size. Defaults to 50.
        circular (bool, optional): Use circular distance metric. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        n_features: int,
        output_shape: Tuple[int, int],
        n_epochs: int = 20,
        eta: float = 0.2,
        k0: int = 50,
        circular: bool = False,
        seed: int = 42,
    ) -> None:
        rng = default_rng(seed)

        self.output_shape = output_shape
        self.n_features = n_features
        self.weights = rng.random(
            size=(output_shape[0] * output_shape[1], n_features)
        )  # Initialize weights randomly between 0 and 1
        self.n_epochs = n_epochs
        self.eta = eta  # Learning rate
        self.k0 = k0  # Initial neighborhood size

        self.grid = self.__get_grid()  # Grid coordinates
        self.grid_distance = (
            circular_manhattan_distance if circular else manhattan_distance
        )

    def __get_grid(self) -> np.ndarray:
        """
        Creates a 2D grid of neuron coordinates.

        Returns:
            np.ndarray: Grid coordinates.
        """
        return np.argwhere(np.ones(shape=self.output_shape)).astype(np.int32)

    def __get_winner_idx(self, x: np.ndarray) -> int:
        """
        Finds the index of the neuron with the closest weight vector to the input.

        Parameters:
            x (np.ndarray): Input data point.

        Returns:
            int: Index of the winning neuron.
        """
        distances = np.sum((self.weights - x) ** 2, axis=1)
        return np.argmin(distances)

    def step(self, x: np.ndarray, t: int) -> None:
        """
        Performs a single training step for a given input.

        Parameters:
            x (np.ndarray): Input data point.
            t (int): Current epoch number.
        """
        winner_idx = self.__get_winner_idx(x)
        winner = self.grid[winner_idx, :]

        tau = self.n_epochs / np.log(self.k0)  # Time constant for neighborhood decay
        sigma = self.k0 * np.exp(-(t**2 / tau))  # Neighborhood radius decay over time

        distances = self.grid_distance(winner, self.grid)
        neighborhood = np.exp(
            -(distances**2 / (2 * sigma**2))
        )  # Influence of the winner neuron

        # Update weights with neighborhood influence
        self.weights += self.eta * neighborhood.reshape(-1, 1) * (x - self.weights)

    def fit(self, X_train: np.ndarray) -> List[Dict[str, Union[int, np.ndarray]]]:
        """
        Trains the SOM on the provided dataset.

        Parameters:
            X_train (np.ndarray): Training data.

        Returns:
            List[Dict[str, Union[int, np.ndarray]]]: Training history including centroids and grid mappings.
        """
        history = []
        for epoch in range(self.n_epochs):
            for x in X_train:
                self.step(x, epoch)  # Perform learning step for each data point

            history.append(
                {
                    "epoch": epoch,
                    "centroids": self.centroids,  # Save current weights
                    "grid": self.predict(
                        X_train
                    ),  # Map of data points to grid positions
                }
            )
        return history

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Maps each new data point to the corresponding winning neuron's grid position.

        Parameters:
            X_new (np.ndarray): New data to predict.

        Returns:
            np.ndarray: Grid positions of the winning neurons.
        """
        return np.array([self.grid[self.__get_winner_idx(x), :] for x in X_new])

    @property
    def centroids(self) -> np.ndarray:
        """
        Returns the current weights (centroids) of the SOM.

        Returns:
            np.ndarray: Weight matrix representing the centroids.
        """
        return self.weights
