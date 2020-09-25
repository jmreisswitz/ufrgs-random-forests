import numpy as np


class EvaluationService:
    @staticmethod
    def accuracy_score(test_predictions: np.array, test_labels: np.array) -> float:
        correct_predictions = 0
        for i in range(len(test_predictions)):
            if test_predictions[i] == test_labels[i]:
                correct_predictions += 1
        return correct_predictions/len(test_predictions)
