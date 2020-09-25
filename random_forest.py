import random
from math import sqrt

import numpy as np

from bootstrap import Bootstrap


class RandomForest:
    def __init__(self, ntree: int):
        self.ntree = ntree
        self.random_trees = self.init_trees()

    def fit(self, train_features: np.array, train_labels: np.array):
        bootstrap = Bootstrap(train_features, train_labels)
        for random_tree in self.random_trees:
            random_tree.fit(bootstrap.get_subset())

    def predict(self, test_features) -> np.array:
        tests_predictions = []
        for test_feature in test_features:
            random_trees_predictions = self._get_trees_predictions(test_feature)
            tests_predictions.append(self._get_prediction_from_voting(random_trees_predictions))
        return np.array(tests_predictions)

    def init_trees(self) -> set:
        random_trees = set()
        for _ in range(self.ntree):
            random_trees.add(RandomTree())
        return random_trees

    def _get_trees_predictions(self, test_feature):
        predictions = []
        for random_tree in self.random_trees:
            predictions.append(random_tree.predict(test_feature))
        return predictions

    @staticmethod
    def _get_prediction_from_voting(random_trees_predictions):
        return max(set(random_trees_predictions), key=random_trees_predictions.count())


class RandomTree:
    def __init__(self):
        self.attributes_to_use = None
        self.number_of_attributes_to_use = None

    def fit(self, train_features: np.array, train_labels: np.array):
        if self.attributes_to_use is None:
            self._init_attributes_to_use(train_features)
        # TODO resto do treinamento

    def predict(self, test_feature):
        pass  # TODO

    def _init_attributes_to_use(self, train_features: np.array):
        number_of_attributes = len(train_features[0])
        self.number_of_attributes_to_use = int(sqrt(number_of_attributes))
        self.attributes_to_use = random.sample([i for i in range(number_of_attributes)], self.number_of_attributes_to_use)
