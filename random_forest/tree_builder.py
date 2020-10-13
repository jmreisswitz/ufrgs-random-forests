import numbers

import pandas as pd
import numpy as np

from gain_info import buildtree
from random_forest.tree_nodes import TreeNode, LeafNode, CategoricalNode, NumericalNode


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
        # if all trains_labels are the same
        if len(set(self.train_labels)) == 1:
            return True
        # or if there are no more features to evaluate
        if len(self.train_features) == 0:
            return True
        # or if all columns are already used
        if len(self.already_used_columns) == len(self.train_features[0]):
            return True
        return False

    def build_node(self) -> TreeNode:
        if self.is_leaf_node():
            return LeafNode(self.condition_value, self.get_prediction_label())
        information_score, best_column, _ = self.get_best_feature()
        print(f'Got score {information_score} from {best_column}')
        # if information_score == 0:
        #     return LeafNode(self.condition_value, self.get_prediction_label())
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
        for possible_value in possible_values:
            new_train_features, new_train_labels = self.remove_categorical_data_from(possible_value, column)
            builder = TreeBuilder(possible_value, new_train_features, new_train_labels, self.get_new_already_used_columns(column))
            children_list.append(builder.build_node())
            children_labels.append(possible_value)
        return CategoricalNode(self.condition_value, column, children_list, children_labels)

    def generate_numerical_children(self, column) -> NumericalNode:
        cutting_point = self.get_cutting_point_on_numerical_data(column)
        left_features, left_labels, right_features, right_labels = self.divide_numerical_dataset(column, cutting_point)
        left_builder = TreeBuilder(f'< {cutting_point}', left_features, left_labels, self.get_new_already_used_columns(column))
        right_builder = TreeBuilder(f'> {cutting_point}', right_features, right_labels, self.get_new_already_used_columns(column))
        return NumericalNode(
            column,
            self.condition_value,
            cutting_point,
            left_builder.build_node(),
            right_builder.build_node()
        )

    def get_cutting_point_on_numerical_data(self, column):
        column_data = [self.train_features[i][column] for i in range(len(self.train_features))]
        return np.average(column_data)

    def get_prediction_label(self):
        train_labels_as_list = self.train_labels.tolist()
        return max(set(train_labels_as_list), key=train_labels_as_list.count)

    def divide_numerical_dataset(self, column, cutting_point):
        bellow_cutting_point_indexes = [i for i in range(len(self.train_features))
                                        if self.train_features[i][column] < cutting_point]
        return np.array([self.train_features[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in bellow_cutting_point_indexes]), \
               np.array([self.train_features[i] for i in range(len(self.train_features)) if
                        i not in bellow_cutting_point_indexes]), \
               np.array([self.train_labels[i] for i in range(len(self.train_labels)) if
                        i not in bellow_cutting_point_indexes])

    def remove_categorical_data_from(self, value, column):
        value_indexes = [i for i in range(len(self.train_features))
                                   if self.train_features[i][column] == value]
        return np.array([self.train_features[i] for i in value_indexes]), \
                np.array([self.train_labels[i] for i in value_indexes])
