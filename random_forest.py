import numbers
import random
from abc import ABC, abstractmethod
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


class TreeNode(ABC):
    @abstractmethod
    def predict_value(self, value):
        pass


class NumericalNode(TreeNode):
    def __init__(self, cutting_point, left_child: TreeNode, right_child: TreeNode):
        super().__init__()
        self.cutting_point = cutting_point
        self.right_child = right_child
        self.left_child = left_child

    def predict_value(self, value):
        if self.cutting_point > value:
            return self.left_child.predict_value(value)
        return self.right_child.predict_value(value)


class LeafNode(TreeNode):
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class

    def predict_value(self, value):
        return self.predicted_class


class CategoricalNode(TreeNode):
    def __init__(self, children, children_labels):
        self.children_labels = children_labels
        self.children = children

    def predict_value(self, value):
        label_index = self.children_labels.index(value)
        return self.children[label_index]


class TreeBuilder:
    def __init__(self, train_features: np.array, train_labels: np.array):
        self.train_features = train_features
        self.train_labels = train_labels
        self.gain_info_service = GainInfoService(self.features_and_labels_to_dataframe())

    def features_and_labels_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.train_features], columns=self.train_labels)

    def is_leaf_node(self) -> bool:
        # if all trains_labels are the same or if there are no more features to evaluate
        return len(set(self.train_labels)) == 1 or len(self.train_features) == 0

    def build_node(self) -> TreeNode:
        if self.is_leaf_node():
            return LeafNode(self.get_prediction_label())
        information_score, best_column = self.get_best_feature()
        logger.info(f'Got score {information_score} from {best_column}')
        if self.is_categorical_data(best_column):
            self.generate_categorical_children(best_column)
        else:
            self.generate_numerical_children(best_column)

    def get_best_feature(self):
        return self.gain_info_service.build_tree()

    def is_categorical_data(self, column) -> bool:
        return not isinstance(self.train_features[column][0], numbers.Number)

    def generate_categorical_children(self, column) -> CategoricalNode:
        children_list = []
        children_labels = []
        for possible_value in self.train_features[column].unique():
            new_train_features, new_train_labels = self.remove_categorical_data_from(column)
            builder = TreeBuilder(new_train_features, new_train_labels)
            children_list.append(builder.build_node())
            children_labels.append(possible_value)
        return CategoricalNode(children_list, children_labels)

    def generate_numerical_children(self, column) -> NumericalNode:
        cutting_point = self.get_cutting_point_on_numerical_data(column)
        left_features, left_labels, right_features, right_labels = self.divide_numerical_dataset(column, cutting_point)
        left_builder = TreeBuilder(left_features, left_labels)
        right_builder = TreeBuilder(right_features, right_labels)
        return NumericalNode(
            cutting_point,
            left_builder.build_node(),
            right_builder.build_node()
        )

    def get_cutting_point_on_numerical_data(self, column):
        return self.train_features[column].average()

    def get_prediction_label(self):
        return np.bincount(self.train_labels).argmax()

    def divide_numerical_dataset(self, column, cutting_point):
        bellow_cutting_point_indexes = [i for i in self.train_features[column]
                                        if self.train_features[column] < cutting_point]
        return np.array([self.train_features[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_features[i] for i in range(len(self.train_features)) if
                        self.train_features[i] not in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in range(len(self.train_labels)) if
                        self.train_labels[i] not in bellow_cutting_point_indexes])

    def remove_categorical_data_from(self, label):
        indexes_of_other_labels = [i for i in self.train_features
                                   if self.train_features[i] != label]
        return np.array([self.train_features[i] for i in indexes_of_other_labels]), \
                np.array([self.train_features[i] for i in indexes_of_other_labels])


class RandomTree:
    def __init__(self):
        self.attributes_to_use = None
        self.number_of_attributes_to_use = None
        self.starting_node = None

    def fit(self, train_features: np.array, train_labels: np.array):
        if self.attributes_to_use is None:
            self._init_attributes_to_use(train_features)
        tree_builder = TreeBuilder(train_features, train_labels)
        self.starting_node = tree_builder.build_node()

    def predict(self, test_feature):
        return self.starting_node.predict_value(test_feature)

    def _init_attributes_to_use(self, train_features: np.array):
        number_of_attributes = len(train_features[0])
        self.number_of_attributes_to_use = int(sqrt(number_of_attributes))
        self.attributes_to_use = random.sample([i for i in range(number_of_attributes)], self.number_of_attributes_to_use)
