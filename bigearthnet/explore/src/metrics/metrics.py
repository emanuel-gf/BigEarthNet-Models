import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics import Metric

class MultiClasses(Metric):
    """Comprehensive metrics (Accuracy, F1-Score, Precision, Recall) for multiclass problems.
    
    Handles both segmentation [B, C, H, W] and classification [B, C] outputs.
    """
    
    def __init__(self, num_classes=20, average='macro'):
        """
        Initialize the multi-class metrics.
        
        Args:
            num_classes (int): Number of classes (default: 20)
            task (str): Task type - 'segmentation' or 'classification'
            average (str): Averaging method - 'macro', 'micro', 'weighted', or None
        """
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        
        # Initialize torchmetrics
        metric_task = "multiclass"
        
        # Core metrics
        self.accuracy = Accuracy(
            task=metric_task, 
            num_classes=num_classes,
            average='micro'  # Overall accuracy
        )
        
        self.f1_score = F1Score(
            task=metric_task,
            num_classes=num_classes,
            average=average
        )
        
        self.precision = Precision(
            task=metric_task,
            num_classes=num_classes,
            average=average
        )
        
        self.recall = Recall(
            task=metric_task,
            num_classes=num_classes,
            average=average
        )
        
        # Per-class metrics (no averaging)
        self.accuracy_per_class = Accuracy(
            task=metric_task,
            num_classes=num_classes,
            average=None
        )
        
        self.f1_per_class = F1Score(
            task=metric_task,
            num_classes=num_classes,
            average=None
        )
        
        # Custom state tracking
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.
        
        Args:
            preds: Predictions tensor
                   - Segmentation: [B, C, H, W] (logits) or [B, H, W] (class indices)
                   - Classification: [B, C] (logits) or [B] (class indices)
            target: Target tensor
                    - Segmentation: [B, H, W] (class indices)
                    - Classification: [B] (class indices)
        """
        

        preds_processed, target_processed = self._process_classification_inputs(preds, target)
        
        # Ensure tensors are on the same device and have correct dtype
        preds_processed = preds_processed.to(target_processed.device)
        target_processed = target_processed.long()
        
        # Update all metrics
        self.accuracy.update(preds_processed, target_processed)
        self.f1_score.update(preds_processed, target_processed)
        self.precision.update(preds_processed, target_processed)
        self.recall.update(preds_processed, target_processed)
        self.accuracy_per_class.update(preds_processed, target_processed)
        self.f1_per_class.update(preds_processed, target_processed)
        
        # Update custom states
        correct = torch.sum(preds_processed == target_processed)
        total = target_processed.numel()
        
        self.correct += correct
        self.total += total
        
        if self.task == 'segmentation':
            self.correct_pixels += correct
            self.total_pixels += total

    def _process_classification_inputs(self, preds, target):
        """Process classification inputs for metrics calculation."""
        
        # Handle predictions
        if preds.dim() == 2:  # [B, C] - logits
            preds = torch.argmax(preds, dim=1)  # [B]
        elif preds.dim() != 1:  # Should be [B]
            raise ValueError(f"For classification, preds should be [B, C] or [B], got {preds.shape}")
        
        # Handle targets
        if target.dim() != 1:  # Should be [B]
            raise ValueError(f"For classification, target should be [B], got {target.shape}")
        
        return preds, target

    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        results = {
            # Overall metrics
            'accuracy': self.accuracy.compute(),
            'f1_score': self.f1_score.compute(),
            'precision': self.precision.compute(),
            'recall': self.recall.compute(),
            
            # Per-class metrics
            'accuracy_per_class': self.accuracy_per_class.compute(),
            'f1_per_class': self.f1_per_class.compute(),
            
            # Custom metrics
            'custom_accuracy': self.correct.float() / self.total,
            'correct_predictions': self.correct,
            'total_predictions': self.total,
        }
        
        return results

    def reset(self):
        """Reset all internal states"""
        super().reset()
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()
        self.accuracy_per_class.reset()
        self.f1_per_class.reset()

    def get_class_metrics(self):
        """Get detailed per-class metrics"""
        results = self.compute()
        
        per_class_info = {
            'per_class_accuracy': results['accuracy_per_class'],
            'per_class_f1': results['f1_per_class'],
        }
        
        return per_class_info