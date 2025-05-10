"""
Evaluation metrics for assessing model performance on letter count and position tasks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    details: Dict[str, Any]

class BaseMetric(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def compute(self, predictions: List[str], targets: List[str]) -> MetricResult:
        """Compute the metric value."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the metric name."""
        pass

class LetterCountAccuracy(BaseMetric):
    """Accuracy metric for letter count predictions."""

    def get_name(self) -> str:
        return "letter_count_accuracy"

    def _extract_number(self, text: str) -> Tuple[bool, int]:
        """Extract the first number from text."""
        try:
            # Find first sequence of digits
            number = int(''.join(c for c in text if c.isdigit()))
            return True, number
        except ValueError:
            return False, 0

    def compute(self, predictions: List[str], targets: List[str]) -> MetricResult:
        """Compute accuracy of letter count predictions."""
        correct = 0
        errors = []

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            success, pred_num = self._extract_number(pred)
            if not success:
                errors.append(f"Could not extract number from prediction: {pred}")
                continue

            try:
                target_num = int(target)
                if pred_num == target_num:
                    correct += 1
                else:
                    errors.append(f"Prediction {pred_num} does not match target {target_num}")
            except ValueError:
                errors.append(f"Could not convert target to number: {target}")

        accuracy = correct / len(predictions) if predictions else 0
        return MetricResult(
            value=accuracy,
            details={"errors": errors, "correct": correct, "total": len(predictions)}
        )

class LetterPositionAccuracy(BaseMetric):
    """Accuracy metric for letter position predictions."""

    def get_name(self) -> str:
        return "letter_position_accuracy"

    def compute(self, predictions: List[str], targets: List[str]) -> MetricResult:
        """Compute accuracy of letter position predictions."""
        correct = 0
        errors = []

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Get first character of prediction and target
            pred_char = pred.strip()[0].lower() if pred.strip() else ''
            target_char = target.strip()[0].lower() if target.strip() else ''

            if not pred_char:
                errors.append(f"Empty prediction or no characters found")
                continue

            if not target_char:
                errors.append(f"Empty target or no characters found")
                continue

            if pred_char == target_char:
                correct += 1
            else:
                errors.append(f"Prediction '{pred_char}' does not match target '{target_char}'")

        accuracy = correct / len(predictions) if predictions else 0
        return MetricResult(
            value=accuracy,
            details={"errors": errors, "correct": correct, "total": len(predictions)}
        )

class MetricsRegistry:
    """Registry for available metrics."""

    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}

        # Register default metrics
        self.register_metric(LetterCountAccuracy())
        self.register_metric(LetterPositionAccuracy())

    def register_metric(self, metric: BaseMetric) -> None:
        """Register a new metric."""
        self._metrics[metric.get_name()] = metric

    def get_metric(self, name: str) -> BaseMetric:
        """Get a metric by name."""
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found in registry")
        return self._metrics[name]

    def list_metrics(self) -> List[str]:
        """List all registered metrics."""
        return list(self._metrics.keys())

# Create global metrics registry
metrics_registry = MetricsRegistry()
