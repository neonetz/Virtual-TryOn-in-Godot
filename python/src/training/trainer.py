"""
Training Pipeline
=================
End-to-end training orchestration for ORB+SVM face detector.

Pipeline:
1. Load and split dataset (train/val/test: 70/15/15)
2. Extract ORB descriptors from all images
3. Build BoVW codebook from training descriptors
4. Encode all descriptors to BoVW histograms
5. Train SVM classifier with hyperparameter tuning
6. Evaluate on validation and test sets
7. Save trained models and reports
"""

import os
import cv2
import numpy as np
import json
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import glob

from ..features import ORBExtractor, BoVWEncoder
from ..detector import SVMClassifier
from ..utils import compute_metrics, plot_pr_curve, plot_roc_curve, plot_confusion_matrix


class Trainer:
    """
    Orchestrates the complete training pipeline.
    
    High Cohesion: Manages all training-related workflows.
    Low Coupling: Coordinates between feature, detector, and utility modules.
    """
    
    def __init__(
        self,
        pos_dir: str,
        neg_dir: str,
        models_dir: str = 'models',
        reports_dir: str = 'reports',
        random_state: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            pos_dir: Directory containing positive (face) images
            neg_dir: Directory containing negative (non-face) images
            models_dir: Directory to save trained models
            reports_dir: Directory to save evaluation reports
            random_state: Random seed for reproducibility
        """
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.random_state = random_state
        
        # Create output directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Components (initialized during training)
        self.orb_extractor = None
        self.bovw_encoder = None
        self.svm_classifier = None
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        print(f"Trainer initialized:")
        print(f"  Positive samples: {pos_dir}")
        print(f"  Negative samples: {neg_dir}")
        print(f"  Models directory: {models_dir}")
        print(f"  Reports directory: {reports_dir}")
    
    def load_dataset(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels from directories.
        
        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Tuple of (image_paths, labels)
        """
        print("\nLoading dataset...")
        
        # Load positive samples
        pos_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        pos_paths = []
        for ext in pos_extensions:
            pos_paths.extend(glob.glob(os.path.join(self.pos_dir, ext)))
            pos_paths.extend(glob.glob(os.path.join(self.pos_dir, ext.upper())))
        
        # Load negative samples
        neg_paths = []
        for ext in pos_extensions:
            neg_paths.extend(glob.glob(os.path.join(self.neg_dir, ext)))
            neg_paths.extend(glob.glob(os.path.join(self.neg_dir, ext.upper())))
        
        print(f"  Positive samples: {len(pos_paths)}")
        print(f"  Negative samples: {len(neg_paths)}")
        
        if len(pos_paths) == 0 or len(neg_paths) == 0:
            raise ValueError("No images found in dataset directories")
        
        # Combine and create labels
        all_paths = pos_paths + neg_paths
        all_labels = [1] * len(pos_paths) + [0] * len(neg_paths)
        
        # Split: train/temp
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels,
            test_size=(test_size + val_size),
            stratify=all_labels,
            random_state=self.random_state
        )
        
        # Split temp: val/test
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_size / (test_size + val_size)),
            stratify=temp_labels,
            random_state=self.random_state
        )
        
        # Store splits
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_paths)} samples")
        print(f"  Val:   {len(val_paths)} samples")
        print(f"  Test:  {len(test_paths)} samples")
        
        return all_paths, all_labels
    
    def extract_features(
        self,
        n_features: int = 500,
        roi_size: Tuple[int, int] = (128, 128)
    ) -> None:
        """
        Extract ORB features from all images.
        
        Args:
            n_features: Max ORB features per image
            roi_size: Resize dimensions
        """
        print("\nExtracting ORB features...")
        
        # Initialize ORB extractor
        self.orb_extractor = ORBExtractor(n_features=n_features)
        
        # Extract from each split
        self.train_descriptors = self._extract_from_paths(
            self.train_paths, roi_size, "train"
        )
        self.val_descriptors = self._extract_from_paths(
            self.val_paths, roi_size, "val"
        )
        self.test_descriptors = self._extract_from_paths(
            self.test_paths, roi_size, "test"
        )
    
    def _extract_from_paths(
        self,
        paths: List[str],
        roi_size: Tuple[int, int],
        split_name: str
    ) -> List[Optional[np.ndarray]]:
        """Extract ORB descriptors from a list of image paths."""
        descriptors = []
        
        for i, path in enumerate(paths):
            if (i + 1) % 50 == 0:
                print(f"  {split_name}: {i+1}/{len(paths)}")
            
            img = cv2.imread(path)
            if img is None:
                print(f"  Warning: Failed to load {path}")
                descriptors.append(None)
                continue
            
            desc = self.orb_extractor.extract(img, roi_size)
            descriptors.append(desc)
        
        print(f"  {split_name}: {len(paths)}/{len(paths)} complete")
        return descriptors
    
    def build_codebook(
        self,
        k: int = 256,
        max_descriptors: int = 200000
    ) -> None:
        """
        Build BoVW codebook from training descriptors.
        
        Args:
            k: Number of visual words (clusters)
            max_descriptors: Max descriptors for k-means
        """
        print("\nBuilding BoVW codebook...")
        
        self.bovw_encoder = BoVWEncoder(n_clusters=k, random_state=self.random_state)
        self.bovw_encoder.build_codebook(
            self.train_descriptors,
            max_descriptors=max_descriptors
        )
        
        # Save codebook
        codebook_path = os.path.join(self.models_dir, 'codebook.pkl')
        self.bovw_encoder.save(codebook_path)
    
    def encode_features(self) -> None:
        """Encode all descriptors to BoVW histograms."""
        print("\nEncoding features with BoVW...")
        
        self.X_train = self.bovw_encoder.encode_batch(self.train_descriptors)
        self.X_val = self.bovw_encoder.encode_batch(self.val_descriptors)
        self.X_test = self.bovw_encoder.encode_batch(self.test_descriptors)
        
        self.y_train = np.array(self.train_labels)
        self.y_val = np.array(self.val_labels)
        self.y_test = np.array(self.test_labels)
        
        print(f"  Train features: {self.X_train.shape}")
        print(f"  Val features:   {self.X_val.shape}")
        print(f"  Test features:  {self.X_test.shape}")
    
    def train_svm(
        self,
        kernel: str = 'linear',
        C: float = 1.0,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Train SVM classifier.
        
        Args:
            kernel: 'linear' or 'rbf'
            C: Regularization parameter
            use_grid_search: Use GridSearchCV for tuning
            
        Returns:
            Training metrics
        """
        print(f"\nTraining SVM classifier (kernel={kernel})...")
        
        self.svm_classifier = SVMClassifier(
            kernel=kernel,
            C=C,
            random_state=self.random_state
        )
        
        if use_grid_search:
            metrics = self.svm_classifier.train_with_grid_search(
                self.X_train,
                self.y_train
            )
        else:
            metrics = self.svm_classifier.train(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val
            )
        
        # Save SVM and scaler
        svm_path = os.path.join(self.models_dir, 'svm.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        self.svm_classifier.save(svm_path, scaler_path)
        
        return metrics
    
    def evaluate(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate on test set and generate reports.
        
        Args:
            save_plots: Save evaluation plots
            
        Returns:
            Test metrics
        """
        print("\nEvaluating on test set...")
        
        # Predictions
        y_pred = self.svm_classifier.predict(self.X_test)
        y_scores = self.svm_classifier.predict_proba(self.X_test)
        
        # Compute metrics
        metrics = compute_metrics(self.y_test, y_pred, y_scores)
        
        # Save metrics
        metrics_path = os.path.join(self.reports_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
        
        # Generate plots
        if save_plots:
            pr_path = os.path.join(self.reports_dir, 'pr_curve.png')
            roc_path = os.path.join(self.reports_dir, 'roc_curve.png')
            cm_path = os.path.join(self.reports_dir, 'confusion_matrix.png')
            
            plot_pr_curve(self.y_test, y_scores, pr_path)
            plot_roc_curve(self.y_test, y_scores, roc_path)
            plot_confusion_matrix(self.y_test, y_pred, cm_path)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics.get('roc_auc', 0):.4f}")
        print("="*60)
        
        return metrics
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save training configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.models_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    
    def run_full_pipeline(
        self,
        k: int = 256,
        max_descriptors: int = 200000,
        n_features: int = 500,
        kernel: str = 'linear',
        C: float = 1.0,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            k: Number of visual words
            max_descriptors: Max descriptors for codebook
            n_features: ORB features per image
            kernel: SVM kernel type
            C: SVM regularization
            use_grid_search: Use hyperparameter tuning
            
        Returns:
            Final test metrics
        """
        # Save configuration
        config = {
            'k': k,
            'max_descriptors': max_descriptors,
            'n_features': n_features,
            'kernel': kernel,
            'C': C,
            'use_grid_search': use_grid_search,
            'random_state': self.random_state,
            'pos_dir': self.pos_dir,
            'neg_dir': self.neg_dir
        }
        self.save_config(config)
        
        # Execute pipeline
        self.load_dataset()
        self.extract_features(n_features=n_features)
        self.build_codebook(k=k, max_descriptors=max_descriptors)
        self.encode_features()
        self.train_svm(kernel=kernel, C=C, use_grid_search=use_grid_search)
        metrics = self.evaluate()
        
        print("\n✓ Training pipeline complete!")
        
        return metrics
