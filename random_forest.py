import numpy as np


class RandomForest:
    def __init__(self, ntree: int):
        self.ntree = ntree
        self.random_trees = set()
        self.init_trees()

    def fit(self, train_features: np.array, train_labels: np.array):
        for random_tree in self.random_trees:
            random_tree.fit(train_features, train_labels)

    def predict(self, test_features) -> np.array:
        tests_predictions = []
        for test_feature in test_features:
            random_trees_predictions = self._get_trees_predictions(test_feature)
            tests_predictions.append(self._get_prediction_from_voting(random_trees_predictions))
        return np.array(tests_predictions)

    def init_trees(self):
        for _ in range(self.ntree):
            self.random_trees.add(RandomTree())

    def _get_trees_predictions(self, test_feature):
        predictions = []
        for random_tree in self.random_trees:
            predictions.append(random_tree.predict(test_feature))
        return predictions

    @staticmethod
    def _get_prediction_from_voting(random_trees_predictions):
        return max(set(random_trees_predictions), key=random_trees_predictions.count())


class RandomTree:
    def fit(self, train_features: np.array, train_labels: np.array):
        pass  # TODO

    def predict(self, test_feature):
        pass  # TODO
