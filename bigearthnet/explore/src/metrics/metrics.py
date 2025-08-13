import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics import Metric

class MultiLabelMetrics(Metric):
    """ multi-label classification .
    
     expects BINARY predictions (0s and 1s), not probabilities or logits!
    Apply sigmoid and thresholding in your training loop before calling update().
    """
    
    def __init__(self, num_classes=19, threshold=0.5, average='macro'):
        """
        Initialize the multi-label metrics.
        
        Args:
            num_classes (int): Number of classes
            threshold (float): Not used in this version - handle thresholding before calling update()
            average (str): Averaging method - 'macro', 'micro', 'weighted', or None
        """
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold  # Keep for compatibility but don't use
        self.average = average
        
        # Initialize torchmetrics for multilabel task
        metric_task = "multilabel"
        
        # Core metrics
        self.accuracy = Accuracy(
            task=metric_task, 
            num_labels=num_classes,
            average='micro'
        )
        
        self.f1_score = F1Score(
            task=metric_task,
            num_labels=num_classes,
            average=average
        )
        
        self.precision = Precision(
            task=metric_task,
            num_labels=num_classes,
            average=average
        )
        
        self.recall = Recall(
            task=metric_task,
            num_labels=num_classes,
            average=average
        )
        
        # Per-class metrics
        self.f1_per_class = F1Score(
            task=metric_task,
            num_labels=num_classes,
            average=None
        )
        self.accuracy_per_class = Accuracy(
            task=metric_task,
            num_labels=num_classes,
            average=None
        )
        
        # Custom state tracking
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.
        
        Args:
            preds: BINARY predictions tensor [B, C] (must be 0s and 1s)
            target: Target tensor [B, C] (binary labels, 0s and 1s)
        """
        # Validate inputs
        assert preds.dim() == 2, f"Predictions should be [B, C], got {preds.shape}"
        assert target.dim() == 2, f"Targets should be [B, C], got {target.shape}"
        assert preds.shape == target.shape, f"Shape mismatch: preds {preds.shape} vs target {target.shape}"
        
        # Ensure binary and correct dtype
        preds = preds.int()
        target = target.int()
        
        # Ensure same device
        preds = preds.to(target.device)
        
        # Validate that predictions are binary
        unique_preds = torch.unique(preds)
        if not all(val in [0, 1] for val in unique_preds.cpu().numpy()):
            print(f"WARNING: Non-binary predictions detected: {unique_preds}")
            # Clamp to binary just in case
            preds = torch.clamp(preds, 0, 1)
        
        # Update all metrics
        self.accuracy.update(preds, target)
        self.f1_score.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1_per_class.update(preds, target)
        self.accuracy_per_class.update(preds, target)
        
        # Update custom states
        correct = torch.sum(preds == target)
        total = target.numel()
        
        self.correct += correct
        self.total += total

    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        try:
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
                'element_accuracy': self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0),
                'correct_predictions': self.correct,
                'total_predictions': self.total,
            }
            
            return results
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return default values if computation fails
            return {
                'accuracy': torch.tensor(0.0),
                'f1_score': torch.tensor(0.0),
                'precision': torch.tensor(0.0),
                'recall': torch.tensor(0.0),
                'accuracy_per_class': torch.zeros(self.num_classes),
                'f1_per_class': torch.zeros(self.num_classes),
                'element_accuracy': torch.tensor(0.0),
                'correct_predictions': torch.tensor(0),
                'total_predictions': torch.tensor(0),
            }

    def reset(self):
        """Reset all internal states"""
        super().reset()
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_per_class.reset()
        self.accuracy_per_class.reset()
def avg_metric_bands(val_metrics, metric_name):
    """
    Compute the average of a given metric_name across all bands.

    Parameters:
    -----------
    val_metrics : dict
        Dictionary with metrics as keys.
    metric_name: str
        metric name: e.g: accuracy
    Returns:
    --------
    float
        The average metrics value.
    """
    total_sam = 0.0
    band_count = len(val_metrics.keys())

    for band, metrics in val_metrics.items():
        total_sam += metrics[metric_name]

    return total_sam / band_count