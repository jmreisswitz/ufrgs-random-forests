import numpy as np


class Bootstrap:
    def __init__(self, features: np.array, labels: np.array):
        self.labels = labels
        self.features = features

    def get_subset(self):
        return self.features, self.labels  # TODO implementar o bootstrap real

