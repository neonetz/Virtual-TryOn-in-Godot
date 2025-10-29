"""
Metrics and Evaluation
======================
Compute classification metrics, generate plots, and save evaluation reports.
"""

import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (optional)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC metrics if scores provided
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['average_precision'] = average_precision_score(y_true, y_scores)
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True negatives, false positives, false negatives, true positives
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    return metrics


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str
) -> None:
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        save_path: Path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"PR curve saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str
) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    labels: list = ['Non-Body', 'Body']
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save plot
        labels: Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def save_metrics_report(
    metrics: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics report saved to {save_path}")


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
    print("="*50)
    
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    if 'average_precision' in metrics:
        print(f"Avg Precision: {metrics['average_precision']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
        
        if 'true_positives' in metrics:
            print(f"\nTP: {metrics['true_positives']}, "
                  f"TN: {metrics['true_negatives']}")
            print(f"FP: {metrics['false_positives']}, "
                  f"FN: {metrics['false_negatives']}")
    
    print("="*50 + "\n")
