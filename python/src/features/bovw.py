"""
Bag of Visual Words (BoVW) Encoder
===================================
Converts variable-length ORB descriptors into fixed-length feature vectors
using a learned visual vocabulary (codebook) via k-means clustering.

Pipeline:
1. Build codebook: k-means clustering on sampled descriptors
2. Encode: Assign each descriptor to nearest centroid, create histogram
3. Normalize: L1 or L2 normalization for scale invariance
"""

import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from typing import List, Optional, Tuple
import os


class BoVWEncoder:
    """
    Bag of Visual Words encoder for converting ORB descriptors to fixed vectors.
    
    High Cohesion: Handles only visual vocabulary creation and encoding.
    Low Coupling: Independent from feature extraction and classification.
    """
    
    def __init__(self, n_clusters: int = 256, random_state: int = 42, use_tfidf: bool = False):
        """
        Initialize BoVW encoder.
        
        Args:
            n_clusters: Number of visual words (codebook size, k)
            random_state: Random seed for reproducibility
            use_tfidf: Apply TF-IDF weighting to histograms
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_tfidf = use_tfidf
        self.kmeans = None
        self.idf_weights = None  # Inverse Document Frequency weights
        self.is_fitted = False
        
    def build_codebook(
        self, 
        descriptors_list: List[np.ndarray],
        max_descriptors: int = 200000,
        batch_size: int = 1000
    ) -> None:
        """
        Build visual vocabulary by clustering sampled descriptors.
        
        Strategy:
        1. Collect all descriptors from training images
        2. Subsample if total exceeds max_descriptors (for speed)
        3. Run MiniBatchKMeans to find k cluster centers (visual words)
        4. Store codebook as self.kmeans
        
        Args:
            descriptors_list: List of descriptor arrays from training images
            max_descriptors: Maximum descriptors to use for clustering
            batch_size: Batch size for MiniBatchKMeans
        """
        # Step 1: Aggregate all descriptors
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
        
        if len(all_descriptors) == 0:
            raise ValueError("No valid descriptors provided for codebook building")
        
        # Concatenate into single array
        all_descriptors = np.vstack(all_descriptors)
        print(f"Collected {len(all_descriptors)} descriptors from {len(descriptors_list)} images")
        
        # Step 2: Subsample if needed
        if len(all_descriptors) > max_descriptors:
            indices = np.random.choice(
                len(all_descriptors), 
                max_descriptors, 
                replace=False
            )
            all_descriptors = all_descriptors[indices]
            print(f"Subsampled to {max_descriptors} descriptors")
        
        # Step 3: Cluster with MiniBatchKMeans (efficient for large datasets)
        # Convert binary descriptors (uint8) to float for clustering
        all_descriptors_float = all_descriptors.astype(np.float32)
        
        print(f"Building codebook with k={self.n_clusters} clusters...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=batch_size,
            random_state=self.random_state,
            verbose=1,
            max_iter=100
        )
        self.kmeans.fit(all_descriptors_float)
        self.is_fitted = True
        print("Codebook built successfully")
        
        # Compute IDF weights if TF-IDF enabled
        if self.use_tfidf:
            self._compute_idf_weights(descriptors_list)
    
    def _compute_idf_weights(self, descriptors_list: List[np.ndarray]) -> None:
        """
        Compute Inverse Document Frequency weights for each visual word.
        
        IDF(w) = log(N / df(w))
        where N = total documents, df(w) = documents containing word w
        
        Args:
            descriptors_list: List of all training descriptors
        """
        print("Computing IDF weights for TF-IDF...")
        n_documents = len([d for d in descriptors_list if d is not None and len(d) > 0])
        word_doc_counts = np.zeros(self.n_clusters)
        
        # Count documents containing each word
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                desc_float = desc.astype(np.float32)
                labels = self.kmeans.predict(desc_float)
                unique_words = np.unique(labels)
                word_doc_counts[unique_words] += 1
        
        # Compute IDF: log(N / df)
        # Add smoothing to avoid division by zero
        self.idf_weights = np.log((n_documents + 1) / (word_doc_counts + 1))
        print(f"IDF weights computed for {self.n_clusters} visual words")
        
    def encode(
        self, 
        descriptors: Optional[np.ndarray],
        normalize: str = 'l1'
    ) -> np.ndarray:
        """
        Encode ORB descriptors into fixed-length BoVW histogram.
        
        Process:
        1. Assign each descriptor to nearest visual word (cluster center)
        2. Count occurrences to create histogram
        3. Normalize histogram (L1 or L2)
        
        Args:
            descriptors: Array of shape (n_keypoints, 32) or None
            normalize: Normalization type ('l1', 'l2', or None)
            
        Returns:
            histogram: Fixed-length vector of size n_clusters
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not built. Call build_codebook() first.")
        
        # Handle empty descriptors (no keypoints detected)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        # Step 1: Predict cluster assignments
        descriptors_float = descriptors.astype(np.float32)
        labels = self.kmeans.predict(descriptors_float)
        
        # Step 2: Build histogram
        histogram = np.bincount(labels, minlength=self.n_clusters).astype(np.float32)
        
        # Step 3: Apply TF-IDF weighting if enabled
        if self.use_tfidf and self.idf_weights is not None:
            histogram = histogram * self.idf_weights
        
        # Step 4: Normalize
        if normalize == 'l1':
            norm = np.sum(histogram)
            if norm > 0:
                histogram = histogram / norm
        elif normalize == 'l2':
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram = histogram / norm
        
        return histogram
    
    def encode_batch(
        self, 
        descriptors_list: List[Optional[np.ndarray]],
        normalize: str = 'l1'
    ) -> np.ndarray:
        """
        Encode multiple descriptor sets into BoVW histograms.
        
        Args:
            descriptors_list: List of descriptor arrays
            normalize: Normalization type
            
        Returns:
            Feature matrix of shape (n_images, n_clusters)
        """
        histograms = [self.encode(desc, normalize) for desc in descriptors_list]
        return np.array(histograms)
    
    def save(self, filepath: str) -> None:
        """
        Save codebook to disk.
        
        Args:
            filepath: Path to save codebook (e.g., 'models/codebook.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted codebook")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'kmeans': self.kmeans,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'use_tfidf': self.use_tfidf,
            'idf_weights': self.idf_weights
        }, filepath)
        print(f"Codebook saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BoVWEncoder':
        """
        Load codebook from disk.
        
        Args:
            filepath: Path to saved codebook
            
        Returns:
            Loaded BoVWEncoder instance
        """
        data = joblib.load(filepath)
        
        # Backward compatibility: check if data is dict or direct KMeans object
        if isinstance(data, dict):
            # New format: dictionary with metadata
            encoder = cls(
                n_clusters=data['n_clusters'],
                random_state=data.get('random_state', 42),
                use_tfidf=data.get('use_tfidf', False)
            )
            encoder.kmeans = data['kmeans']
            encoder.idf_weights = data.get('idf_weights', None)
        else:
            # Old format: direct KMeans object
            encoder = cls(
                n_clusters=data.n_clusters,
                random_state=42,
                use_tfidf=False
            )
            encoder.kmeans = data
            encoder.idf_weights = None
        
        encoder.is_fitted = True
        print(f"Codebook loaded from {filepath}")
        return encoder
    
    def get_codebook_stats(self) -> dict:
        """
        Get statistics about the learned codebook.
        
        Returns:
            Dictionary with cluster stats
        """
        if not self.is_fitted:
            return {"status": "not fitted"}
        
        return {
            "n_clusters": self.n_clusters,
            "cluster_centers_shape": self.kmeans.cluster_centers_.shape,
            "inertia": self.kmeans.inertia_,
            "n_iter": self.kmeans.n_iter_
        }
