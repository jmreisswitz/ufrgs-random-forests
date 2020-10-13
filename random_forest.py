import numbers
import random
from abc import ABC, abstractmethod
from math import sqrt

import logging
import pandas as pd
import numpy as np

from bootstrap import Bootstrap
from gain_info2 import buildtree

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
    def __init__(self, column, condition):
        self.condition = condition
        self.children = []
        self.column = column

    @abstractmethod
    def predict_value(self, features):
        pass

    @staticmethod
    def get_tabs(depth):
        return '\t' * depth

    def print_node(self, columns_names, depth):
        print(f"{self.get_tabs(depth)}{self.condition} -> {self.__class__.__name__} {columns_names[self.column]}:")
        for child in self.children:
            child.print_node(columns_names, depth + 1)

    def __repr__(self):
        return_string = f'{self.condition} -> {self.__class__.__name__} {self.column}:'
        for child in self.children:
            return_string += f'\t{child}\n'
        return return_string[:-1]


class NumericalNode(TreeNode):
    def __init__(self, column, cutting_point, left_child: TreeNode, right_child: TreeNode):
        super().__init__(column, cutting_point)
        self.cutting_point = cutting_point
        self.children = [left_child, right_child]

    def predict_value(self, features):
        if self.cutting_point > features:
            return self.children[0].predict_features(features)
        return self.children[1].predict_value(features)


class LeafNode(TreeNode):
    def __init__(self, condition, predicted_class):
        super().__init__(None, condition)
        self.predicted_class = predicted_class

    def print_node(self, columns_names, depth):
        print(f"{self.get_tabs(depth)}{self.condition} -> {self.__class__.__name__}: {self.predicted_class}")

    def predict_value(self, features):
        return self.predicted_class

    def __repr__(self):
        return f'{self.condition} -> {self.__class__.__name__}: {self.predicted_class}'


class CategoricalNode(TreeNode):
    def __init__(self, condition, column, children, children_labels):
        super().__init__(column, condition)
        self.children_labels = children_labels
        self.children = children

    def predict_value(self, features):
        label_index = self.children_labels.index(features[self.column])
        return self.children[label_index].predict_value(features)


class TreeBuilder:
    def __init__(self, condition_value, train_features: np.array, train_labels: np.array, already_used_columns: set):
        self.condition_value = condition_value
        self.train_features = train_features
        self.train_labels = train_labels
        self.already_used_columns = already_used_columns

    def features_and_labels_to_gain_info_entry(self):
        features_dataframe = pd.DataFrame(self.train_features)
        return np.array(
            pd.concat([features_dataframe, pd.Series(self.train_labels)], axis=1, sort=False)).tolist()

    def is_leaf_node(self) -> bool:
        # if all trains_labels are the same or if there are no more features to evaluate
        return len(set(self.train_labels)) == 1 or len(self.train_features) == 0

    def build_node(self) -> TreeNode:
        if self.is_leaf_node():
            return LeafNode(self.condition_value, self.get_prediction_label())
        information_score, best_column, _ = self.get_best_feature()
        print(f'Got score {information_score} from {best_column}')
        if information_score == 0:
            return LeafNode(self.condition_value, self.get_prediction_label())
        if self.is_categorical_data(best_column):
            return self.generate_categorical_children(best_column)
        else:
            return self.generate_numerical_children(best_column)

    def get_best_feature(self):
        return buildtree(self.features_and_labels_to_gain_info_entry(), self.already_used_columns)

    def is_categorical_data(self, column) -> bool:
        return not isinstance(self.train_features[0][column], numbers.Number)

    def get_new_already_used_columns(self, column) -> set:
        self.already_used_columns.add(column)
        return self.already_used_columns

    def generate_categorical_children(self, column) -> CategoricalNode:
        children_list = []
        children_labels = []
        possible_values = set([feature_row[column] for feature_row in self.train_features])  # self.train_features[column])
        logger.info(possible_values)
        for possible_value in possible_values:
            new_train_features, new_train_labels = self.remove_categorical_data_from(possible_value, column)
            builder = TreeBuilder(possible_value, new_train_features, new_train_labels, self.get_new_already_used_columns(column))
            children_list.append(builder.build_node())
            children_labels.append(possible_value)
        return CategoricalNode(self.condition_value, column, children_list, children_labels)

    def generate_numerical_children(self, column) -> NumericalNode:
        cutting_point = self.get_cutting_point_on_numerical_data(column)
        left_features, left_labels, right_features, right_labels = self.divide_numerical_dataset(column, cutting_point)
        left_builder = TreeBuilder('<', left_features, left_labels, self.get_new_already_used_columns(column))
        right_builder = TreeBuilder('>', right_features, right_labels, self.get_new_already_used_columns(column))
        return NumericalNode(
            column,
            cutting_point,
            left_builder.build_node(),
            right_builder.build_node()
        )

    def get_cutting_point_on_numerical_data(self, column):
        return self.train_features[column].average()

    def get_prediction_label(self):
        train_labels_as_list = self.train_labels.tolist()
        return max(set(train_labels_as_list), key=train_labels_as_list.count)

    def divide_numerical_dataset(self, column, cutting_point):
        bellow_cutting_point_indexes = [i for i in self.train_features[column]
                                        if self.train_features[column] < cutting_point]
        return np.array([self.train_features[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_features[i] for i in range(len(self.train_features)) if
                        self.train_features[i] not in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in range(len(self.train_labels)) if
                        self.train_labels[i] not in bellow_cutting_point_indexes])

    def remove_categorical_data_from(self, value, column):
        value_indexes = [i for i in range(len(self.train_features))
                                   if self.train_features[i][column] == value]
        return np.array([self.train_features[i] for i in value_indexes]), \
                np.array([self.train_labels[i] for i in value_indexes])


class RandomTree:
    def __init__(self):
        self.attributes_to_use = None
        self.number_of_attributes_to_use = None
        self.starting_node = None

    def fit(self, train_features: np.array, train_labels: np.array):
        if self.attributes_to_use is None:
            self._init_attributes_to_use(train_features)
        already_used_columns = set()
        tree_builder = TreeBuilder('start', train_features, train_labels, already_used_columns)
        self.starting_node = tree_builder.build_node()

    def predict(self, test_feature):
        return self.starting_node.predict_value(test_feature)

    def _init_attributes_to_use(self, train_features: np.array):
        number_of_attributes = len(train_features[0])
        self.number_of_attributes_to_use = int(sqrt(number_of_attributes))
        self.attributes_to_use = random.sample([i for i in range(number_of_attributes)], self.number_of_attributes_to_use)

    def print_tree(self, columns_names: dict):
        self.starting_node.print_node(columns_names, 0)

    def __repr__(self):
        return str(self.starting_node)
