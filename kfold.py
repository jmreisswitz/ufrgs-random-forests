from functools import reduce
from random import shuffle

import numpy as np
import pandas as pd

"""
Created on Mon Oct 12 15:14:30 2020

@author: edu & mario
"""


class Kfold:
    def __init__(self, dataset: pd.DataFrame, target_column: str, folds_num: int = 5):
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

    def uniquecounts(self):
        results = {}
        for row in self.labels:
            if row not in results:
                results[row] = 0
            results[row] += 1
        return results

    def separate_labels(self):
        class_labels = []
        final_class = []
        results = self.uniquecounts()
        for r in results.keys():
            for i in range(len(self.labels)):
                if self.labels[i] == r:
                    class_labels.append(i)
            final_class.append([col for col in class_labels])
            class_labels.clear()
        return final_class

    def _generate_folds(self):
        indexes = self.separate_labels()
        for row in indexes:
            shuffle(row)
        results = [int(len(row) / self.folds_num) for row in indexes]
        fold_begin = [0 for row in indexes]
        folds = []
        for _ in range(self.folds_num):
            n_fold = []
            for i in range(0, len(indexes)):
                n_fold = n_fold + indexes[i][fold_begin[i]: fold_begin[i] + results[i]]
                fold_begin[i] = fold_begin[i] + results[i]
            folds.append(n_fold)
        return folds


if __name__ == '__main__':
    my_data = pd.read_csv('datasets/wine-recognition.tsv', sep='\t')
    folds_num = 10
    kfolds = Kfold(my_data, 'target')
    print(kfolds._generate_folds())
    for fold in kfolds.get_next_fold():
        print(fold)
