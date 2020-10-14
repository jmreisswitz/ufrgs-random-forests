from math import sqrt
import random

import numpy as np

from random_forest.tree_builder import TreeBuilder


class RandomTree:
    def __init__(self, possible_values_for_features: dict, use_all_attribs=False, verbose=False):
        self.verbose = verbose
        self.use_all_attribs = use_all_attribs
        self.possible_values_for_features = possible_values_for_features
        self.attributes_to_use = None
        self.number_of_attributes_to_use = None
        self.starting_node = None

    def fit(self, train_features: np.array, train_labels: np.array):
        already_used_columns = set()
        if self.attributes_to_use is None and not self.use_all_attribs:
            self._init_attributes_to_use(train_features)
            already_used_columns = self.get_not_usable_columns(len(train_features))
        tree_builder = TreeBuilder('start', train_features, train_labels, already_used_columns, self.possible_values_for_features, verbose=self.verbose)
        self.starting_node = tree_builder.build_node()

    def predict(self, test_feature):
        return self.starting_node.predict_value(test_feature)

    def _init_attributes_to_use(self, train_features: np.array):
        number_of_attributes = len(train_features[0])
        self.number_of_attributes_to_use = int(sqrt(number_of_attributes))
        self.attributes_to_use = random.sample([i for i in range(number_of_attributes)], self.number_of_attributes_to_use)

    def print_tree(self, columns_names: dict):
        self.starting_node.print_node(columns_names, 0)

    def get_not_usable_columns(self, number_of_features):
        all_attributes_set = set([i for i in range(number_of_features)])
        return all_attributes_set - set(self.attributes_to_use)
