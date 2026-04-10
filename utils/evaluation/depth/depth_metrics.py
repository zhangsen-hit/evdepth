"""
Depth estimation evaluation metrics
Includes standard depth metrics: RMSE, MAE, AbsRel, SqRel, and threshold accuracy
All metrics operate in real depth space (not log)
"""
import numpy as np
import torch
from typing import Dict, Optional


class DepthMetrics:
    """
    Compute standard depth estimation metrics
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values"""
        self.num_samples = 0
        self.sum_rmse = 0.0
        self.sum_mae = 0.0
        self.sum_abs_rel = 0.0
        self.sum_sq_rel = 0.0
        self.sum_delta1 = 0.0
        self.sum_delta2 = 0.0
        self.sum_delta3 = 0.0
    
    @staticmethod
    def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute depth metrics for a single prediction
        
        Args:
            pred: predicted depth (B, 1, H, W) in real space (not log)
            target: ground truth depth (B, 1, H, W) in real space (not log)
            mask: valid depth mask (B, 1, H, W) [optional]
        
        Returns:
            dict of metrics
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        else:
            pred = pred.flatten()
            target = target.flatten()
        
        # Ensure positive depths
        pred = torch.clamp(pred, min=1e-3)
        target = torch.clamp(target, min=1e-3)
        
        # Absolute differences
        abs_diff = torch.abs(pred - target)
        
        # RMSE (Root Mean Square Error)
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        
        # MAE (Mean Absolute Error)
        mae = torch.mean(abs_diff)
        
        # AbsRel (Absolute Relative Error)
        abs_rel = torch.mean(abs_diff / target)
        
        # SqRel (Squared Relative Error)
        sq_rel = torch.mean(((pred - target) ** 2) / target)
        
        # Threshold accuracies (δ < 1.25, 1.25^2, 1.25^3)
        max_ratio = torch.max(pred / target, target / pred)
        delta1 = torch.mean((max_ratio < 1.25).float())
        delta2 = torch.mean((max_ratio < 1.25 ** 2).float())
        delta3 = torch.mean((max_ratio < 1.25 ** 3).float())
        
        metrics = {
            'rmse': rmse.item(),
            'mae': mae.item(),
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item(),
        }
        
        return metrics
    
    def add_batch(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Add a batch of predictions for metric accumulation
        
        Args:
            pred: predicted depth (B, 1, H, W)
            target: ground truth depth (B, 1, H, W)
            mask: valid depth mask (B, 1, H, W)
        """
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_i = pred[i:i+1]
            target_i = target[i:i+1]
            mask_i = mask[i:i+1] if mask is not None else None
            
            metrics = self.compute_metrics(pred_i, target_i, mask_i)
            
            self.sum_rmse += metrics['rmse']
            self.sum_mae += metrics['mae']
            self.sum_abs_rel += metrics['abs_rel']
            self.sum_sq_rel += metrics['sq_rel']
            self.sum_delta1 += metrics['delta1']
            self.sum_delta2 += metrics['delta2']
            self.sum_delta3 += metrics['delta3']
            self.num_samples += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get averaged metrics across all accumulated samples
        
        Returns:
            dict of averaged metrics
        """
        if self.num_samples == 0:
            return {}
        
        metrics = {
            'rmse': self.sum_rmse / self.num_samples,
            'mae': self.sum_mae / self.num_samples,
            'abs_rel': self.sum_abs_rel / self.num_samples,
            'sq_rel': self.sum_sq_rel / self.num_samples,
            'delta1': self.sum_delta1 / self.num_samples,
            'delta2': self.sum_delta2 / self.num_samples,
            'delta3': self.sum_delta3 / self.num_samples,
        }
        
        return metrics
    
    def has_data(self) -> bool:
        """Check if any data has been added"""
        return self.num_samples > 0


class DepthEvaluator:
    """
    Evaluator for depth estimation
    Similar to PropheseeEvaluator but for depth
    """
    def __init__(self, min_depth: float = 0.1, max_depth: float = 100.0):
        """
        Args:
            min_depth: minimum valid depth value
            max_depth: maximum valid depth value
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.metrics_calculator = DepthMetrics()
    
    def reset_buffer(self):
        """Reset accumulated predictions and targets"""
        self.metrics_calculator.reset()
    
    def add_predictions(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Add predictions for evaluation
        
        Args:
            pred: predicted depth in normalized log space (norm_log)
            target: ground truth depth in normalized log space (norm_log)
            mask: valid depth mask
        """
        # Convert from norm_log -> log(depth) -> real depth (meters)
        log_min = torch.log(torch.tensor(self.min_depth, device=pred.device, dtype=pred.dtype))
        log_max = torch.log(torch.tensor(self.max_depth, device=pred.device, dtype=pred.dtype))
        denom = torch.clamp(log_max - log_min, min=1e-6)

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pred_log_depth = pred * denom + log_min
        target_log_depth = target * denom + log_min

        pred_real = torch.exp(pred_log_depth)
        target_real = torch.exp(target_log_depth)
        
        # Create valid mask based on depth range
        if mask is None:
            mask = torch.ones_like(target_real, dtype=torch.bool)
        
        depth_mask = (target_real >= self.min_depth) & (target_real <= self.max_depth)
        valid_mask = mask & depth_mask
        
        self.metrics_calculator.add_batch(pred_real, target_real, valid_mask)
    
    def evaluate_buffer(self) -> Dict[str, float]:
        """
        Evaluate accumulated predictions
        
        Returns:
            dict of metrics
        """
        return self.metrics_calculator.get_metrics()
    
    def has_data(self) -> bool:
        """Check if any data has been accumulated"""
        return self.metrics_calculator.has_data()

