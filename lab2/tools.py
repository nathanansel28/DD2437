from typing import List, Union, Tuple
import numpy as np
import sys


class AnimalSOM:
    def __init__(
        self, use_seed: bool=True, seed_value: int=20250205
    ):
        with open('data/animalattributes.txt', 'r') as file:
            lines = file.read().splitlines()
            self.animal_attribute_names = {
                index: line.strip("'") for index, line in enumerate(lines)
            }
        with open('data/animalnames.txt', 'r') as file:
            lines = file.read().splitlines()
            self.animal_names = {
                index: line.strip("'\t") for index, line in enumerate(lines)
            }
        with open('data/animals.dat', 'r') as file: 
            for line in file: 
                self.animal_data = np.array(
                    [int(x.strip()) for x in line.split(",")]
                ).reshape(32, 84)

        if use_seed:
            np.random.seed(20250205)
        self.weights = np.random.rand(100, 84)


    def fit(
        self, n_epochs: int=20, step_size: float=0.2
    ) -> None: 
        for epoch in range(n_epochs): 
            for i, animal in self.animal_names.items():
                animal_vector = self.animal_data[i]
                winner_node, winner_distance = self._search_winner_node(animal_vector)
                self._update_weights(
                    animal_vector, winner_node, step_size, epoch
                )
        

    def predict(
        self 
    ) -> dict:
        result = []
        for i, animal in self.animal_names.items():
            animal_vector = self.animal_data[i]
            winner_node, winner_distance = self._search_winner_node(animal_vector)
            result.append(
                (winner_node, animal)
            )

        sorted_result = sorted(result, key=lambda x: x[0])
        result = {}
        for i, animal in sorted_result:
            if i not in result:
                result[i] = []
            result[i].append(animal)

        return result


    def _search_winner_node(
        self, animal_vector: np.ndarray
    ) -> Tuple[int, float]:
        """
        Returns the winner node to update the weights.
        """
        winner_node, winner_distance = None, sys.maxsize

        for node_idx, node_vector in enumerate(self.weights):
            distance = self._calculate_euclidean_distance(
                animal_vector, node_vector
            )
            if distance < winner_distance: 
                winner_node = node_idx
                winner_distance = distance

        return winner_node, winner_distance

    
    def _calculate_euclidean_distance(
        self, vector_x: np.ndarray, vector_w: np.ndarray
    ) -> float: 
        """
        Returns the euclidean distance between vector_x and vector_w
        """
        assert vector_x.shape == vector_w.shape

        return np.matmul(
            (vector_x - vector_w).T, (vector_x - vector_w)
        )
    

    def _update_weights(
        self, 
        animal_vector: np.ndarray, 
        winner_node: int, 
        step_size: float,
        epoch: int
    ) -> None:
        neighbourhood_size = self._determine_neighbourhood_size(epoch)
        if neighbourhood_size == 1: 
            self.weights[winner_node] += step_size * animal_vector
        else:
            start_idx = int(winner_node - neighbourhood_size/2)
            end_idx = int(winner_node + neighbourhood_size/2)
            self.weights[start_idx:end_idx] += step_size * animal_vector


    def _determine_neighbourhood_size(
        self, epoch: int
    ) -> int:
        """
        Placeholder neighbourhood function just to get things to work.
        NOT PERFECT YET
        """
        
        return (
            50 if epoch < 5
            else 25 if epoch < 10
            else 12 if epoch < 15
            else 6 if epoch < 18
            else 3 if epoch < 19
            else 2 if epoch < 20
            else 1
        )

