from abc import ABC, abstractmethod
from typing import Any
from schuyler.experimenter.result import Result

class Metric(ABC):
    def __call__(self, true_labels, pred_labels, **kwargs) -> float:
        return self.score(true_labels, pred_labels)#
    
    @abstractmethod
    def score(self, true_labels: Result, pred_labels: Result) -> float:
        ...

class F1Score():
    def __call__(self, precision, recall) -> float:
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    def __str__(self):
        return "F1Score"