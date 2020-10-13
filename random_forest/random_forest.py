import numpy as np

from bootstrap import Bootstrap
from random_forest.random_tree import RandomTree


class RandomForest:
    def __init__(self, ntree: int):
        self.ntree = ntree
        self.random_trees = self.init_trees()

    def fit(self, train_features: np.array, train_labels: np.array):
        bootstrap = Bootstrap(train_features, train_labels)
        for random_tree in self.random_trees:
            random_tree.fit(*bootstrap.get_subset())
            random_tree.print_tree(None)

    def predict_test_data(self, test_features) -> np.array:
        tests_predictions = []
        for test_feature in test_features:
            tests_predictions.append(self.predict_value(test_feature))
        return np.array(tests_predictions)

    def predict_value(self, features):
        random_trees_predictions = self._get_trees_predictions(features)
        return self._get_prediction_from_voting(random_trees_predictions)

    def init_trees(self) -> set:
        random_trees = set()
        for _ in range(self.ntree):
            random_tree = RandomTree()
            random_trees.add(random_tree)
        return random_trees

    def _get_trees_predictions(self, test_feature):
        predictions = []
        for random_tree in self.random_trees:
            predictions.append(random_tree.predict(test_feature))
        return predictions

    @staticmethod
    def _get_prediction_from_voting(random_trees_predictions):
        return max(set(random_trees_predictions), key=random_trees_predictions.count)



