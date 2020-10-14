from typing import Tuple
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype

from evaluation_service import EvaluationService
from kfold import Kfold
from random_forest.random_forest import RandomForest


class Train:
    def __init__(self, dataset: pd.DataFrame, target_column: str, ntrees: int):
        self.target_column = target_column
        self.dataset = dataset
        self.check_for_numerical_categorical_rows()
        self.model = RandomForest(ntrees, self.get_possible_features())
        self.number_of_folds = 5
        self.kfold = Kfold(dataset, target_column, folds_num=self.number_of_folds)

    def check_for_numerical_categorical_rows(self):
        for column in self.dataset.columns:
            if len(self.dataset[column].unique()) < 10:  # its a categorical column
                self.dataset[column] = self.dataset[column].apply(str)

    def execute(self):
        evaluations = []
        for i in range(self.number_of_folds):
            train_features, test_features, train_labels, test_labels = self._separate_dataset()
            self._train(train_features, train_labels)
            test_predictions = self._predict(test_features)
            evaluations.append((i+1, self._evaluate_model(test_predictions, test_labels)))
        return evaluations

    def _train(self, train_features: np.array, train_labels: np.array) -> None:
        self.model.fit(train_features, train_labels)

    def _predict(self, test_features: np.array) -> np.array:
        return self.model.predict_test_data(test_features)

    def _separate_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        return self.kfold.get_next_fold()

    @staticmethod
    def _evaluate_model(test_predictions, test_labels):
        return EvaluationService.accuracy_score(test_predictions, test_labels)

    def get_possible_features(self) -> dict:
        possibilities_dict = dict()
        counter = 0
        for column in self.dataset.columns:
            if not is_numeric_dtype(self.dataset[column]):
                possibilities_dict[counter] = set(self.dataset[column].unique())
            counter += 1
        return possibilities_dict
