"""
SVM Classifier for Face Detection
===================================
Trains and applies Support Vector Machine classifier on BoVW feature vectors.
Provides both Linear and RBF kernels with hyperparameter tuning via GridSearchCV.
"""

import numpy as np
import joblib
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Tuple, Optional, Dict, Any
import os


class SVMClassifier:
    """
    SVM-based body/non-body classifier operating on BoVW features.
    
    High Cohesion: Handles only SVM training, prediction, and evaluation.
    Low Coupling: Works with any fixed-length feature vectors.
    """
    
    def __init__(
        self, 
        kernel: str = 'linear',
        C: float = 1.0,
        gamma: str = 'scale',
        random_state: int = 42
    ):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: 'linear' or 'rbf'
            C: Regularization parameter
            gamma: Kernel coefficient (for RBF)
            random_state: Random seed
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        # Initialize classifier
        if kernel == 'linear':
            self.clf = LinearSVC(
                C=C,
                random_state=random_state,
                max_iter=5000,
                dual=True
            )
        elif kernel == 'rbf':
            self.clf = SVC(
                kernel='rbf',
                C=C,
                gamma=gamma,
                random_state=random_state,
                probability=True
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel}. Use 'linear' or 'rbf'")
        
        # Scaler for feature standardization
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train SVM classifier with feature standardization.
        
        Pipeline:
        1. Fit StandardScaler on training data
        2. Transform features to zero mean, unit variance
        3. Fit SVM classifier
        4. Optionally evaluate on validation set
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels (0=non-body, 1=body)
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {self.kernel} SVM...")
        print(f"Training samples: {len(X_train)} ({np.sum(y_train)} positive, {len(y_train) - np.sum(y_train)} negative)")
        
        # Step 1 & 2: Fit scaler and transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Step 3: Train classifier
        self.clf.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Training accuracy
        train_acc = self.clf.score(X_train_scaled, y_train)
        print(f"Training accuracy: {train_acc:.4f}")
        
        metrics = {'train_accuracy': train_acc}
        
        # Step 4: Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics.update({'val_' + k: v for k, v in val_metrics.items()})
        
        return metrics
    
    def train_with_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Train with hyperparameter tuning via GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid, e.g., {'C': [0.1, 1, 10]}
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and CV scores
        """
        print(f"\nPerforming {cv}-fold GridSearchCV for {self.kernel} SVM...")
        
        # Default parameter grids
        if param_grid is None:
            if self.kernel == 'linear':
                param_grid = {'C': [0.1, 1.0, 10.0]}
            else:  # rbf
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Grid search
        grid_search = GridSearchCV(
            self.clf,
            param_grid,
            cv=cv,
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Update classifier with best parameters
        self.clf = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature vectors
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for RBF) or decision function (for Linear).
        
        Args:
            X: Feature vectors
            
        Returns:
            Confidence scores for positive class
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained.")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.clf, 'predict_proba'):
            # RBF kernel with probability=True
            proba = self.clf.predict_proba(X_scaled)
            return proba[:, 1]  # Return positive class probability
        else:
            # Linear SVM: use decision function
            scores = self.clf.decision_function(X_scaled)
            return scores
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with accuracy, precision, recall, F1, AUC
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained.")
        
        y_pred = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_test, y_scores)
        except:
            auc = None
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if auc is not None:
            print(f"  ROC AUC:   {auc:.4f}")
        
        return metrics
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained classifier and scaler.
        
        Args:
            model_path: Path for SVM model (e.g., 'models/svm.pkl')
            scaler_path: Path for scaler (e.g., 'models/scaler.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save untrained classifier.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.clf, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'SVMClassifier':
        """
        Load trained classifier and scaler.
        
        Args:
            model_path: Path to SVM model
            scaler_path: Path to scaler
            
        Returns:
            Loaded SVMClassifier instance
        """
        clf_loaded = joblib.load(model_path)
        scaler_loaded = joblib.load(scaler_path)
        
        # Determine kernel type
        if isinstance(clf_loaded, LinearSVC):
            kernel = 'linear'
        else:
            kernel = clf_loaded.kernel
        
        # Create instance and set loaded components
        classifier = cls(kernel=kernel)
        classifier.clf = clf_loaded
        classifier.scaler = scaler_loaded
        classifier.is_fitted = True
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        
        return classifier
