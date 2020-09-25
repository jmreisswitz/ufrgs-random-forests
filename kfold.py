from functools import reduce
from random import shuffle
from typing import Tuple

import numpy as np


class Kfold:
    def __init__(self, dataset: np.DataFrame, target_column: str, folds_num: int = 5):
        self.folds_num = folds_num
        self.labels = np.array(dataset[target_column])
        self.features = np.array(dataset.drop(target_column, axis=1))
        self.folds = self._generate_folds()
        self.current_fold = 0

    def get_next_fold(self):
        # returns train_features, test_features, train_labels, test_labels
        i = self.current_fold
        self.current_fold += 1
        test_indexes = self.folds[i]
        train_indexes = [self.folds[j] for j in range(len(self.folds)) if j != i]
        train_indexes = reduce(lambda x, y: x + y, train_indexes)
        train_features = [self.features[i] for i in train_indexes]
        train_labels = [self.labels[i] for i in train_indexes]
        test_features = [self.features[i] for i in test_indexes]
        test_labels = [self.labels[i] for i in test_indexes]
        return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)

    def separate_labels(self) -> Tuple[list, list]:
        positive_labels = []
        negative_labels = []
        for i in range(len(self.labels)):
            if self.labels[i] == 1:
                positive_labels.append(i)
            else:
                negative_labels.append(i)
        return positive_labels, negative_labels

    def _generate_folds(self):
        positive_indexes, negative_indexes = self.separate_labels()
        shuffle(positive_indexes)
        shuffle(negative_indexes)
        fold_negative_indexes_size = int(len(negative_indexes)/self.folds_num)
        fold_positive_indexes_size = int(len(positive_indexes) / self.folds_num)
        fold_positive_begin = 0
        fold_negative_begin = 0
        folds = []
        for _ in range(self.folds_num):
            folds.append(
                positive_indexes[fold_positive_begin: fold_positive_begin + fold_positive_indexes_size]
                + negative_indexes[fold_negative_begin: fold_negative_begin + fold_negative_indexes_size]
            )
            fold_positive_begin += fold_positive_indexes_size
            fold_negative_begin += fold_negative_indexes_size
        return folds
