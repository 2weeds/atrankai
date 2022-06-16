import numpy as np
import random

class SOM:
    def __init__(self, training_data: [], som_dimensions: [], learning_rate=0.1):
        self.training_data = training_data
        self.rows = training_data.shape[0]
        self.cols = training_data.shape[1]
        self.learning_rate = learning_rate
        self.map = np.random.random_sample(size=(som_dimensions[0], som_dimensions[1], self.cols))

    """
    Random input vector from training dataset
    """
    def get_random_input_vector(self):
        id = random.randint(0, self.rows)
        vector, self.training_data = self.training_data[0], \
                                     np.vstack((self.training_data[:id],
                                                self.training_data[id+1:]))
        return vector

    """
    Calculating the Best Matching Unit (BMU)
    """
    def find_best_matching_unit(self, input_vector):
        # bmu - best matching unit
        bmu_row = 0
        bmu_col = 0
        lowest_dist = 9999999999999999999999999999999999
        for i in range(len(self.map)):
            for k in range(len(self.map[i])):
                current_dist = np.linalg.norm(self.map[i][k] - input_vector)
                if current_dist < lowest_dist:
                    bmu_row = i
                    bmu_col = k
                    lowest_dist = current_dist
        return [bmu_row, bmu_col, lowest_dist]

    """
    Calculating the size of the neighborhood around the BMU
    """
    def calculate_size_of_neighborhood(self, winner):
        neighbors_coordinates = []

        if winner[0] == 0:
            if winner[1] == 0:
                neighbors_coordinates.append([winner[0], winner[1] + 1])
                neighbors_coordinates.append([winner[0] + 1, winner[1]])
            elif winner[1] == self.cols - 1:
                neighbors_coordinates.append([0, winner[1] - 2])
                neighbors_coordinates.append([1, winner[1] - 1])
            elif 0 < winner[1] > self.cols:
                neighbors_coordinates.append([0, winner[1] - 1])
                neighbors_coordinates.append([0, winner[1] + 1])
                neighbors_coordinates.append([1, winner[1]])
        elif winner[0] == self.rows - 1:
            if winner[1] == 0:
                neighbors_coordinates.append([winner[0], 1])
                neighbors_coordinates.append([winner[0] + 1, 0])
            elif winner[1] == self.cols - 1:
                neighbors_coordinates.append([winner[0], winner[1] - 1])
                neighbors_coordinates.append([winner[0] + 1, winner[1]])
            elif 0 < winner[1] > self.cols:
                neighbors_coordinates.append([winner[0], winner[1] - 1])
                neighbors_coordinates.append([winner[0], winner[1] + 1])
                neighbors_coordinates.append([winner[0] - 1, winner[1]])
        elif 0 < winner[0] > self.rows:
            if winner[1] == 0:
                neighbors_coordinates.append([winner[0] - 1, winner[1]])
                neighbors_coordinates.append([winner[0] + 1, winner[1]])
                neighbors_coordinates.append([winner[0], winner[1] + 1])
            elif winner[1] == self.cols - 1:
                neighbors_coordinates.append([winner[0] - 1, winner[1]])
                neighbors_coordinates.append([winner[0] + 1, winner[1]])
                neighbors_coordinates.append([winner[0], winner[1] - 1])
            elif 0 < winner[1] > self.cols:
                neighbors_coordinates.append([winner[0] + 1, winner[1]])
                neighbors_coordinates.append([winner[0], winner[1] + 1])
                neighbors_coordinates.append([winner[0] - 1, winner[1]])
                neighbors_coordinates.append([winner[0], winner[1] - 1])

        return neighbors_coordinates

    """
    Updating weights of the BMU and neighborhood nodes
    """
    def update_weights(self, winner, neighbors):
        print()
    """
    The decay of learning rate
    """
    def decay_of_learning_rate(self, epochs: int, epoch: int):
        return self.learning_rate * (1 - epoch / epochs)
    """
    Execute SOM training
    """
    def train(self, epochs: int = 1000):
        random_input_vector = self.get_random_input_vector()
        for epoch in epochs:
            for k in range(len(self.training_data)):

                winner = self.find_best_matching_unit(random_input_vector)
                neighbors = self.calculate_size_of_neighborhood(winner)
                self.update_weights(random_input_vector, winner, neighbors)

                self.learning_rate = self.decay_of_learning_rate(epochs, epoch)