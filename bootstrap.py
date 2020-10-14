import pandas as pd
import numpy as np
from random import choice


class Bootstrap:
    def __init__(self, features: np.array, labels: np.array):
        self.labels = labels
        self.features = features
        self.number_of_rows = len(features)
        self.rows_indexes = [i for i in range(self.number_of_rows)]
        self.subset_size = 1.5

    def get_subset(self):
        features_subset = []
        labels_subset = []
        for _ in range(int(self.number_of_rows * self.subset_size)):
            chosen_index = choice(self.rows_indexes)
            features_subset.append(self.features[chosen_index])
            labels_subset.append(self.labels[chosen_index])
        return np.array(features_subset), np.array(labels_subset)


if __name__ == '__main__':
    benchmark_data = pd.read_csv('datasets/benchmark.csv', sep=';')
    labels = np.array(benchmark_data['Joga'])
    features = np.array(benchmark_data.drop('Joga', axis=1))
    bootstrap = Bootstrap(features, labels)
    print('subset1: ', bootstrap.get_subset())
    print('subset2: ', bootstrap.get_subset())
