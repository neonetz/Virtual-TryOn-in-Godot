"""
Dataset Auto-ROI Generator
===========================
Automatically generates positive and negative ROIs from face images.

Rules:
- Positive: IOU >= 0.7 with ground truth
- Negative: IOU < 0.3 with ground truth
- Skip ambiguous (0.3 <= IOU < 0.7)

No pre-trained models used - pure geometric analysis.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import json
from pathlib import Path


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IOU) between two boxes.
    
    Args:
        box1: (x, y, w, h)
        box2: (x, y, w, h)
        
    Returns:
        IOU value [0, 1]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2)
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class DatasetGenerator:
    """
    Generates positive and negative ROIs from face images.
    
    Classical CV approach:
    - No pre-trained models
    - Uses sliding window + IOU rules
    - Generates balanced dataset
    """
    
    def __init__(
        self,
        positive_iou_threshold: float = 0.7,
        negative_iou_threshold: float = 0.3,
        window_sizes: List[int] = None,
        stride: int = 16,
        max_per_image: int = 10
    ):
        """
        Initialize dataset generator.
        
        Args:
            positive_iou_threshold: Min IOU for positive samples (default 0.7)
            negative_iou_threshold: Max IOU for negative samples (default 0.3)
            window_sizes: List of window sizes for sliding window
            stride: Stride for sliding window
            max_per_image: Max samples per image per class
        """
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.window_sizes = window_sizes or [64, 80, 96, 112, 128]
        self.stride = stride
        self.max_per_image = max_per_image
        
        print(f"DatasetGenerator initialized")
        print(f"  Positive IOU threshold: >= {positive_iou_threshold}")
        print(f"  Negative IOU threshold: < {negative_iou_threshold}")
        print(f"  Window sizes: {self.window_sizes}")
        print(f"  Stride: {stride}")
    
    def _propose_windows(
        self,
        image_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding window proposals.
        
        Args:
            image_shape: (height, width)
            
        Returns:
            List of (x, y, w, h) proposals
        """
        height, width = image_shape[:2]
        proposals = []
        
        for size in self.window_sizes:
            # Slide window
            for y in range(0, height - size + 1, self.stride):
                for x in range(0, width - size + 1, self.stride):
                    proposals.append((x, y, size, size))
        
        return proposals
    
    def generate_from_image(
        self,
        image: np.ndarray,
        ground_truth_box: Tuple[int, int, int, int],
        visualize: bool = False
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Generate positive and negative ROIs from a single image.
        
        Args:
            image: Input image
            ground_truth_box: Face bounding box (x, y, w, h)
            visualize: Show visualization (for debugging)
            
        Returns:
            Dict with 'positive' and 'negative' lists of boxes
        """
        # Propose windows
        proposals = self._propose_windows(image.shape)
        
        # Classify proposals by IOU
        positive_boxes = []
        negative_boxes = []
        
        for proposal in proposals:
            iou = compute_iou(proposal, ground_truth_box)
            
            if iou >= self.positive_iou_threshold:
                positive_boxes.append(proposal)
            elif iou < self.negative_iou_threshold:
                negative_boxes.append(proposal)
            # Skip ambiguous (0.3 <= IOU < 0.7)
        
        # Sample max_per_image from each class
        if len(positive_boxes) > self.max_per_image:
            indices = np.random.choice(len(positive_boxes), self.max_per_image, replace=False)
            positive_boxes = [positive_boxes[i] for i in indices]
        
        if len(negative_boxes) > self.max_per_image:
            indices = np.random.choice(len(negative_boxes), self.max_per_image, replace=False)
            negative_boxes = [negative_boxes[i] for i in indices]
        
        # Visualization
        if visualize:
            self._visualize_samples(image, ground_truth_box, positive_boxes, negative_boxes)
        
        return {
            'positive': positive_boxes,
            'negative': negative_boxes
        }
    
    def _visualize_samples(
        self,
        image: np.ndarray,
        ground_truth: Tuple[int, int, int, int],
        positive_boxes: List[Tuple[int, int, int, int]],
        negative_boxes: List[Tuple[int, int, int, int]]
    ) -> None:
        """Visualize generated samples."""
        vis = image.copy()
        
        # Draw ground truth (yellow)
        x, y, w, h = ground_truth
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(vis, "GT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw positive samples (green)
        for box in positive_boxes[:5]:  # Show only first 5
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Draw negative samples (red)
        for box in negative_boxes[:5]:  # Show only first 5
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        # Add legend
        cv2.putText(vis, f"Positive: {len(positive_boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Negative: {len(negative_boxes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Dataset Generator - Samples', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def generate_from_directory(
        self,
        images_dir: str,
        annotations_file: str,
        output_dir: str,
        visualize_first: int = 0
    ) -> None:
        """
        Generate dataset from directory with annotations.
        
        Annotations format (JSON):
        {
            "image1.jpg": {"box": [x, y, w, h]},
            "image2.jpg": {"box": [x, y, w, h]},
            ...
        }
        
        Args:
            images_dir: Directory containing images
            annotations_file: Path to annotations JSON
            output_dir: Output directory for generated dataset
            visualize_first: Visualize first N images (0 = no visualization)
        """
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"\nGenerating dataset from {images_dir}")
        print(f"Total images: {len(annotations)}")
        
        # Create output directories
        pos_dir = os.path.join(output_dir, 'positive')
        neg_dir = os.path.join(output_dir, 'negative')
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
        
        pos_count = 0
        neg_count = 0
        
        for idx, (image_name, annotation) in enumerate(annotations.items()):
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load: {image_path}")
                continue
            
            # Get ground truth box
            gt_box = tuple(annotation['box'])
            
            # Generate samples
            visualize = idx < visualize_first
            samples = self.generate_from_image(image, gt_box, visualize=visualize)
            
            # Save positive samples
            for i, box in enumerate(samples['positive']):
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]
                
                # Resize to standard size (96x96)
                roi_resized = cv2.resize(roi, (96, 96))
                
                # Save
                filename = f"{Path(image_name).stem}_pos_{i}.png"
                cv2.imwrite(os.path.join(pos_dir, filename), roi_resized)
                pos_count += 1
            
            # Save negative samples
            for i, box in enumerate(samples['negative']):
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]
                
                # Resize to standard size (96x96)
                roi_resized = cv2.resize(roi, (96, 96))
                
                # Save
                filename = f"{Path(image_name).stem}_neg_{i}.png"
                cv2.imwrite(os.path.join(neg_dir, filename), roi_resized)
                neg_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(annotations)} images")
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Positive samples: {pos_count}")
        print(f"  Negative samples: {neg_count}")
        print(f"  Output directory: {output_dir}")
    
    def generate_from_simple_directory(
        self,
        images_dir: str,
        output_dir: str,
        center_face_ratio: float = 0.5,
        visualize_first: int = 0
    ) -> None:
        """
        Generate dataset from directory without annotations.
        Assumes face is roughly centered in each image.
        
        Args:
            images_dir: Directory containing face images
            output_dir: Output directory
            center_face_ratio: Ratio of image size for center face (0.5 = 50%)
            visualize_first: Visualize first N images
        """
        print(f"\nGenerating dataset from {images_dir} (center-face assumption)")
        
        # Find images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(images_dir).glob(ext))
        
        print(f"Found {len(image_files)} images")
        
        # Create output directories
        pos_dir = os.path.join(output_dir, 'positive')
        neg_dir = os.path.join(output_dir, 'negative')
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
        
        pos_count = 0
        neg_count = 0
        
        for idx, image_path in enumerate(image_files):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Estimate face box (center, 50% of image size)
            face_size = int(min(w, h) * center_face_ratio)
            face_x = (w - face_size) // 2
            face_y = (h - face_size) // 2
            gt_box = (face_x, face_y, face_size, face_size)
            
            # Generate samples
            visualize = idx < visualize_first
            samples = self.generate_from_image(image, gt_box, visualize=visualize)
            
            # Save samples
            for i, box in enumerate(samples['positive']):
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (96, 96))
                filename = f"{image_path.stem}_pos_{i}.png"
                cv2.imwrite(os.path.join(pos_dir, filename), roi_resized)
                pos_count += 1
            
            for i, box in enumerate(samples['negative']):
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (96, 96))
                filename = f"{image_path.stem}_neg_{i}.png"
                cv2.imwrite(os.path.join(neg_dir, filename), roi_resized)
                neg_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(image_files)} images")
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Positive samples: {pos_count}")
        print(f"  Negative samples: {neg_count}")


if __name__ == '__main__':
    # Example usage
    generator = DatasetGenerator(
        positive_iou_threshold=0.7,
        negative_iou_threshold=0.3,
        window_sizes=[64, 80, 96, 112, 128],
        stride=16,
        max_per_image=10
    )
    
    # Option 1: With annotations file
    # generator.generate_from_directory(
    #     images_dir='data/raw_faces',
    #     annotations_file='data/annotations.json',
    #     output_dir='data/generated',
    #     visualize_first=3
    # )
    
    # Option 2: Simple (center-face assumption)
    generator.generate_from_simple_directory(
        images_dir='data/raw_faces',
        output_dir='data/generated',
        center_face_ratio=0.5,
        visualize_first=3
    )
