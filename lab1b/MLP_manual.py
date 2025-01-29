from typing import List, Union, Dict, Tuple
import numpy as np
from functools import cache


def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Adds a bias term to the input data by appending a row of ones.

    Parameters:
    ----------
    X : np.ndarray
        A 2D NumPy array of shape (n_features, n_samples), where each column
        represents a data point.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array of shape (n_features + 1, n_samples) with an
        additional bias row of ones.
    """
    return np.vstack((X, np.ones((1, X.shape[1]))))


class MultiLayerPerceptronClassifier:
    def __init__(self, n_layers: int = 1, n_nodes: int = 10):
        self.hidden_layers = n_layers
        self.n_nodes = n_nodes
        self.weights: List[np.ndarray] = []
        self.activations: Dict[int, np.ndarray] = {}
        self.delta_weights: Dict[int, np.ndarray] = {}
        self.gradients: Dict[int, np.ndarray] = {}
   

    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray,
        learn_rate: float = 0.25,
        epochs: int = 10,
        batch: bool = True,
    ) -> None:
        if not batch: 
            raise NotImplementedError
        
        inputs = add_bias(X)
        print(inputs.shape)
        labels = y = (
            y.reshape(-1, 1) if y.ndim == 1 else y
        )  # Ensure y is (1, n_samples)

        self.weights = self._initialize_weights(X=inputs, y=labels)

        for epoch_idx in range(epochs):
            print(f"\nEPOCH: {epoch_idx+1}")
            if batch: 
                self.forward_pass(X=inputs)
                self.backward_pass(y=labels)
                self.update_weights(X=inputs, learn_rate=learn_rate)


    # def _initialize_weights(self, X: np.ndarray, y) -> None: 
    #     """
    #     X already includes the bias term.
    #     """
    #     rndg = np.random.default_rng(seed=20250128)
    #     num_classes = len(np.unique(y))
    #     weights = [
    #         rndg.random((
    #             X.shape[0], 1 if num_classes == 2 else num_classes
    #         )) * 0.1 - 0.05
    #     ]
    #     weights += [
    #         (
    #             rndg.random((
    #                 len(X) - 1, self.n_nodes + 1
    #             ))
    #         ) for _ in range(self.hidden_layers)
    #     ]
    #     return weights


    def _initialize_weights(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Initializes weights for all layers of the MLP, including input, hidden, and output layers.
        """
        rndg = np.random.default_rng(seed=20250128)
        weights = []

        # Input layer → First hidden layer
        input_size = X.shape[0]  # Number of input features (including bias)
        hidden_size = self.n_nodes
        weights.append(rndg.uniform(-0.05, 0.05, size=(input_size, hidden_size)))

        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            weights.append(rndg.uniform(-0.05, 0.05, size=(hidden_size + 1, hidden_size)))

        # Last hidden layer → Output layer
        output_size = len(np.unique(y))  # Number of classes (output layer size)
        weights.append(rndg.uniform(-0.05, 0.05, size=(hidden_size + 1, output_size)))

        return weights


    def forward_pass(
        self, X: np.ndarray, current_layer: int = 0
    ) -> np.ndarray:
        """
        Uses a recursive structure to perform forward pass for all the layers. 

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array of shape (n_features + 1, n_samples) which also include the bias term.
        current_layer : int = 0
            The current layer of the calculation. The forward pass starts at 0 which is the default value. 
        """

        print(f"\nLayer: {current_layer}")
        print(f"Weights of the current layer: {np.transpose(self.weights[current_layer]).shape}, \n {self.weights[current_layer]}")
        print(f"Current inputs: {X.shape} \n {X}")
        try: 
            self.activations[current_layer] = self.phi(
                self.weights[current_layer].T @ X
            )
        except Exception as e: 
            print(f"X: {X}")
            raise ValueError

        if current_layer == self.hidden_layers:
            return None # break the recursion
        else: 
            return self.forward_pass(
                X=add_bias(self.activations[current_layer]), current_layer=current_layer+1
            )

    def backward_pass(
        self, y: np.ndarray, current_layer: Union[int, None]=None, batch: bool = True
    ) -> np.ndarray:
        if not batch: 
            raise NotImplementedError
        if not current_layer: 
            current_layer = self.hidden_layers - 1

        self.delta_weights[current_layer] = (
            (self.activations[current_layer] - y) * self.phi_gradient(self.activations[current_layer - 1])
        )
        if current_layer != 0:
            return self.backward_pass(
                y = self.activations[current_layer - 1], current_layer = current_layer - 1
            )
        else: 
            return None # break the recursion


    def update_weights(self, X: np.ndarray, learn_rate: float = 0.25) -> None: 
        for i, weight in enumerate(self.weights): 
            self.weights[i] -= learn_rate * X * self.delta_weights[i]


    # @cache
    def phi(self, X: np.ndarray): 
        return 2 / (1 + np.exp(-X)) - 1
    

    # @cache    
    def phi_gradient(self, X: np.ndarray): 
        # """Current implementation still uses caching via the Python decorator."""
        return (1 + self.phi(X)) * (1 - self.phi(X)) / 2
        # TODO: # return (1 + phi) * (1 - phi) / 2