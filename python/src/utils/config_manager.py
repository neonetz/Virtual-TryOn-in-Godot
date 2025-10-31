"""
Configuration Manager
=====================
Handles saving and loading of model configuration and hyperparameters.
Ensures reproducibility and tracks training settings.
"""

import json
import os
from typing import Dict, Any
from datetime import datetime


class ConfigManager:
    """
    Manages configuration persistence for reproducibility.
    
    Saves:
    - Training hyperparameters
    - Model architecture settings
    - Dataset information
    - Timestamp and version info
    """
    
    @staticmethod
    def save_config(
        config: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            filepath: Output path (e.g., 'models/config.json')
        """
        # Add metadata
        config['saved_at'] = datetime.now().isoformat()
        config['version'] = '1.0'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Config file path
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print(f"Configuration loaded from {filepath}")
        return config
    
    @staticmethod
    def create_training_config(
        pos_dir: str,
        neg_dir: str,
        k: int,
        max_desc: int,
        n_features: int,
        svm_kernel: str,
        C: float,
        use_tfidf: bool,
        seed: int
    ) -> Dict[str, Any]:
        """
        Create training configuration dictionary.
        
        Args:
            pos_dir: Positive samples directory
            neg_dir: Negative samples directory
            k: Number of visual words
            max_desc: Max descriptors for codebook
            n_features: ORB features per image
            svm_kernel: SVM kernel type
            C: SVM regularization
            use_tfidf: Use TF-IDF weighting
            seed: Random seed
            
        Returns:
            Config dictionary
        """
        return {
            'dataset': {
                'positive_dir': pos_dir,
                'negative_dir': neg_dir
            },
            'features': {
                'extractor': 'ORB',
                'n_features': n_features
            },
            'bovw': {
                'n_clusters': k,
                'max_descriptors': max_desc,
                'use_tfidf': use_tfidf,
                'normalization': 'l1'
            },
            'classifier': {
                'type': 'SVM',
                'kernel': svm_kernel,
                'C': C
            },
            'training': {
                'random_seed': seed,
                'test_split': 0.15,
                'val_split': 0.15
            }
        }
