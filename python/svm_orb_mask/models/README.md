# Trained Models Directory

This directory should contain the following trained model files:

- **svm_model.pkl** - Trained SVM classifier (scikit-learn SVC)
- **codebook.pkl** - k-means codebook for Bag of Visual Words (MiniBatchKMeans)
- **scaler.pkl** - Feature scaler (StandardScaler)

## Training

These models must be trained separately (e.g., in Google Colab) and placed here before running inference.

Refer to the main README.md for the complete training pipeline.

## Model Specifications

### svm_model.pkl
- Type: `sklearn.svm.SVC`
- Kernel: Linear
- Input: BoVW histogram (256-dimensional vector, normalized)
- Output: Binary classification (1 = face, 0 = non-face)

### codebook.pkl
- Type: `sklearn.cluster.MiniBatchKMeans`
- n_clusters: 256 (codebook size)
- Input: ORB descriptors (32-dimensional binary features)
- Output: Cluster assignments (visual words)

### scaler.pkl
- Type: `sklearn.preprocessing.StandardScaler`
- Fitted on: Training set BoVW histograms
- Purpose: Normalize features before SVM classification
