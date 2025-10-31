"""
Metrics Reporter
================
Generates comprehensive evaluation reports with metrics and visualizations.
Saves PR curves, confusion matrices, and detailed classification metrics.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsReporter:
    """
    Handles evaluation metrics reporting and visualization.
    
    Generates:
    - Precision-Recall curves
    - ROC curves
    - Confusion matrices
    - Detailed classification reports
    - JSON metrics files
    """
    
    def __init__(self, reports_dir: str = 'reports'):
        """
        Initialize metrics reporter.
        
        Args:
            reports_dir: Directory to save reports
        """
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
    
    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        split_name: str = 'test'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            split_name: Dataset split name (train/val/test)
            
        Returns:
            Dictionary with all metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_scores)
        
        # Generate visualizations
        pr_path = os.path.join(self.reports_dir, f'{split_name}_pr_curve_{timestamp}.png')
        roc_path = os.path.join(self.reports_dir, f'{split_name}_roc_curve_{timestamp}.png')
        cm_path = os.path.join(self.reports_dir, f'{split_name}_confusion_matrix_{timestamp}.png')
        
        self.plot_pr_curve(y_true, y_scores, save_path=pr_path)
        self.plot_roc_curve(y_true, y_scores, save_path=roc_path)
        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        # Add paths to metrics
        metrics['visualizations'] = {
            'pr_curve': pr_path,
            'roc_curve': roc_path,
            'confusion_matrix': cm_path
        }
        
        # Save metrics JSON
        json_path = os.path.join(self.reports_dir, f'{split_name}_metrics_{timestamp}.json')
        self.save_metrics_json(metrics, json_path)
        
        print(f"\n✓ Full report generated:")
        print(f"  Metrics JSON: {json_path}")
        print(f"  PR Curve:     {pr_path}")
        print(f"  ROC Curve:    {roc_path}")
        print(f"  Confusion Matrix: {cm_path}")
        
        return metrics
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores
            
        Returns:
            Dictionary with metrics
        """
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Best F1 threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'overall': {
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score'],
                'support': int(report['1']['support'])
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'auc': {
                'pr_auc': float(pr_auc),
                'roc_auc': float(roc_auc)
            },
            'optimal_threshold': {
                'value': float(best_threshold),
                'f1_score': float(f1_scores[best_f1_idx])
            },
            'per_class': {
                'class_0': {
                    'precision': report['0']['precision'],
                    'recall': report['0']['recall'],
                    'f1_score': report['0']['f1-score'],
                    'support': int(report['0']['support'])
                },
                'class_1': {
                    'precision': report['1']['precision'],
                    'recall': report['1']['recall'],
                    'f1_score': report['1']['f1-score'],
                    'support': int(report['1']['support'])
                }
            }
        }
        
        return metrics
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Path to save plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Path to save plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['Non-Face', 'Face']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def save_metrics_json(
        self,
        metrics: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            filepath: Output path
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics JSON saved to {filepath}")
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print metrics summary to console.
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy:  {metrics['overall']['accuracy']:.4f}")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall:    {metrics['overall']['recall']:.4f}")
        print(f"F1 Score:  {metrics['overall']['f1_score']:.4f}")
        print(f"\nPR AUC:    {metrics['auc']['pr_auc']:.4f}")
        print(f"ROC AUC:   {metrics['auc']['roc_auc']:.4f}")
        print(f"\nOptimal Threshold: {metrics['optimal_threshold']['value']:.4f}")
        print(f"  (F1 Score: {metrics['optimal_threshold']['f1_score']:.4f})")
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm['true_negatives']}  FP: {cm['false_positives']}")
        print(f"  FN: {cm['false_negatives']}  TP: {cm['true_positives']}")
        print("="*60)
