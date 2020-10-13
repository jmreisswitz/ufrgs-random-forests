from typing import Tuple
import pandas as pd
import numpy as np

from evaluation_service import EvaluationService
from kfold import Kfold
from random_forest.random_forest import RandomForest


class Train:
    def __init__(self, model: RandomForest, dataset: pd.DataFrame, target_column: str):
        self.target_column = target_column
        self.dataset = dataset
        self.model = model
        self.number_of_folds = 5
        self.kfold = Kfold(dataset, target_column, folds_num=self.number_of_folds)

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
