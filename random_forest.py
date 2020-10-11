import numbers
import random
from math import sqrt

import logging
import pandas as pd
import numpy as np

from bootstrap import Bootstrap
from gain_info import GainInfoService


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


class TreeNode:
    def __init__(self, train_features: np.array, train_labels: np.array):
        self.train_features = train_features
        self.train_labels = train_labels
        self.is_leaf = self.is_leaf_node()
        self.is_categorical = None
        self.children = []
        self.gain_info_service = GainInfoService(self.features_and_labels_to_dataframe())

    def features_and_labels_to_dataframe(self):
        return pd.DataFrame([self.train_features], columns=self.train_labels)

    def is_leaf_node(self):
        # if all trains_labels are the same or if there are no more features to evaluate
        return len(set(self.train_labels)) == 1 or len(self.train_features) == 0

    def build(self):
        if self.is_leaf:
            return
        information_score, best_column = self.get_best_feature()
        logger.info(f'Got score {information_score} from {best_column}')
        if self.is_categorical_data(best_column):
            self.generate_categorical_children(best_column)
        else:
            self.generate_numerical_children(best_column)

    def get_best_feature(self):
        return self.gain_info_service.build_tree()

    def is_categorical_data(self, column):
        return not isinstance(self.train_features[column][0], numbers.Number)

    def generate_categorical_children(self, column):
        pass

    def generate_numerical_children(self, column):
        cutting_point = self.get_cutting_point_on_numerical_data(column)

    def get_cutting_point_on_numerical_data(self, column):
        return self.train_features[column].average()


class RandomTree:
    def __init__(self):
        self.attributes_to_use = None
        self.number_of_attributes_to_use = None
        self.starting_node = None

    def fit(self, train_features: np.array, train_labels: np.array):
        if self.attributes_to_use is None:
            self._init_attributes_to_use(train_features)
        self.starting_node = TreeNode(train_features, train_labels)
        self.starting_node.build()

    def predict(self, test_feature):
        pass  # TODO

    def _init_attributes_to_use(self, train_features: np.array):
        number_of_attributes = len(train_features[0])
        self.number_of_attributes_to_use = int(sqrt(number_of_attributes))
        self.attributes_to_use = random.sample([i for i in range(number_of_attributes)], self.number_of_attributes_to_use)
