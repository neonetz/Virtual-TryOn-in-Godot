"""
Face Detector using HOG + SVM
Implements sliding window detection with multi-scale pyramid
"""

import cv2
import numpy as np
import pickle
import json
from skimage.feature import hog
from typing import List, Dict, Tuple, Optional


class FaceDetector:
    """Face detection using trained HOG+SVM model"""

    def __init__(self, model_path: str, scaler_path: str, config_path: str):
        """
        Initialize face detector

        Args:
            model_path: Path to trained SVM model (.pkl)
            scaler_path: Path to feature scaler (.pkl)
            config_path: Path to configuration file (.json)
        """
        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Detection parameters
        self.window_size = self.config["target_size"]
        self.window_stride = 16  # Pixels to move window
        self.scale_factor = 1.2  # Scale multiplier for pyramid
        self.min_scale = 0.5  # Minimum scale
        self.max_scale = 2.0  # Maximum scale
        self.confidence_threshold = 0.0  # SVM decision threshold
        self.nms_threshold = 0.3  # Non-maximum suppression threshold

        print(f"✓ Face detector loaded")
        print(f"  Window size: {self.window_size}")
        print(
            f"  Feature dimensions: {self.config.get('training_metadata', {}).get('feature_dimensions', 'N/A')}"
        )
        print(
            f"  Model accuracy: {self.config.get('training_metadata', {}).get('test_accuracy', 'N/A')}"
        )

    def extract_hog_features(self, img: np.ndarray) -> np.ndarray:
        """Extract HOG features from image"""
        features = hog(
            img,
            orientations=self.config["hog_orientations"],
            pixels_per_cell=tuple(self.config["hog_pixels_per_cell"]),
            cells_per_block=tuple(self.config["hog_cells_per_block"]),
            block_norm=self.config["hog_block_norm"],
            visualize=False,
            feature_vector=True,
        )
        return features

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Histogram equalization for better contrast
        img = cv2.equalizeHist(img)

        return img

    def sliding_window(
        self, img: np.ndarray, window_size: Tuple[int, int], stride: int
    ) -> List[Tuple[int, int, np.ndarray]]:
        """
        Generate sliding windows over image

        Yields:
            (x, y, window) tuples
        """
        h, w = img.shape[:2]
        win_h, win_w = window_size

        windows = []
        for y in range(0, h - win_h + 1, stride):
            for x in range(0, w - win_w + 1, stride):
                window = img[y : y + win_h, x : x + win_w]
                if window.shape[0] == win_h and window.shape[1] == win_w:
                    windows.append((x, y, window))

        return windows

    def image_pyramid(self, img: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """
        Generate multi-scale image pyramid

        Yields:
            (scale, resized_image) tuples
        """
        pyramid = []
        scale = self.min_scale

        while scale <= self.max_scale:
            # Calculate new dimensions
            h, w = img.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Skip if too small
            if new_h < self.window_size[0] or new_w < self.window_size[1]:
                scale *= self.scale_factor
                continue

            # Resize
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pyramid.append((scale, resized))

            scale *= self.scale_factor

        return pyramid

    def non_max_suppression(
        self, boxes: List[Dict], threshold: float = 0.3
    ) -> List[Dict]:
        """
        Apply non-maximum suppression to remove overlapping detections

        Args:
            boxes: List of detection dictionaries with 'bbox' and 'confidence'
            threshold: IoU threshold for suppression

        Returns:
            Filtered list of detections
        """
        if len(boxes) == 0:
            return []

        # Extract coordinates and scores
        boxes_array = np.array(
            [
                [
                    b["bbox"]["x"],
                    b["bbox"]["y"],
                    b["bbox"]["x"] + b["bbox"]["w"],
                    b["bbox"]["y"] + b["bbox"]["h"],
                    b["confidence"],
                ]
                for b in boxes
            ]
        )

        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        scores = boxes_array[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def detect(self, img: np.ndarray, single_face: bool = True) -> List[Dict]:
        """
        Detect faces in image

        Args:
            img: Input image (BGR or grayscale)
            single_face: If True, return only the best detection

        Returns:
            List of detection dictionaries with keys:
            - bbox: {x, y, w, h}
            - confidence: detection score
        """
        # Preprocess
        processed = self.preprocess_image(img)
        original_h, original_w = processed.shape[:2]

        # Store all detections
        all_detections = []

        # Generate pyramid and detect at each scale
        pyramid = self.image_pyramid(processed)

        for scale, scaled_img in pyramid:
            # Get sliding windows
            windows = self.sliding_window(
                scaled_img, self.window_size, self.window_stride
            )

            # Process each window
            for x, y, window in windows:
                # Extract features
                features = self.extract_hog_features(window)
                features_scaled = self.scaler.transform([features])

                # Predict
                prediction = self.model.predict(features_scaled)[0]

                if prediction == 1:  # Face detected
                    # Get confidence score
                    confidence = self.model.decision_function(features_scaled)[0]

                    if confidence >= self.confidence_threshold:
                        # Convert coordinates back to original image scale
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(self.window_size[1] / scale)
                        orig_h = int(self.window_size[0] / scale)

                        # Ensure within bounds
                        orig_x = max(0, min(orig_x, original_w - orig_w))
                        orig_y = max(0, min(orig_y, original_h - orig_h))

                        all_detections.append(
                            {
                                "bbox": {
                                    "x": orig_x,
                                    "y": orig_y,
                                    "w": orig_w,
                                    "h": orig_h,
                                },
                                "confidence": float(confidence),
                            }
                        )

        # Apply non-maximum suppression
        detections = self.non_max_suppression(all_detections, self.nms_threshold)

        # Sort by confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Return single best detection if requested
        if single_face and len(detections) > 0:
            return [detections[0]]

        return detections

    def detect_with_roi_tracking(
        self,
        img: np.ndarray,
        previous_detection: Optional[Dict] = None,
        expansion_factor: float = 1.5,
    ) -> List[Dict]:
        """
        Detect faces with ROI tracking for better performance

        If previous detection is provided, search only in expanded ROI

        Args:
            img: Input image
            previous_detection: Previous frame's detection
            expansion_factor: How much to expand ROI

        Returns:
            List of detections
        """
        if previous_detection is None:
            # Full detection
            return self.detect(img, single_face=True)

        # Extract ROI from previous detection
        bbox = previous_detection["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Expand ROI
        expand_w = int(w * (expansion_factor - 1) / 2)
        expand_h = int(h * (expansion_factor - 1) / 2)

        roi_x = max(0, x - expand_w)
        roi_y = max(0, y - expand_h)
        roi_w = min(img.shape[1] - roi_x, w + 2 * expand_w)
        roi_h = min(img.shape[0] - roi_y, h + 2 * expand_h)

        # Extract ROI
        roi = img[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        # Detect in ROI
        roi_detections = self.detect(roi, single_face=True)

        # Adjust coordinates back to full image
        for det in roi_detections:
            det["bbox"]["x"] += roi_x
            det["bbox"]["y"] += roi_y

        # Fallback to full detection if nothing found in ROI
        if len(roi_detections) == 0:
            return self.detect(img, single_face=True)

        return roi_detections

    def visualize_detection(
        self, img: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            img: Input image
            detections: List of detection dictionaries

        Returns:
            Image with bounding boxes drawn
        """
        vis_img = img.copy()

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]

            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Draw rectangle
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"Face: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_img,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return vis_img


if __name__ == "__main__":
    # Test the detector
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python face_detector.py <model.pkl> <scaler.pkl> <config.json> <test_image.jpg>"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    scaler_path = sys.argv[2]
    config_path = sys.argv[3]
    image_path = sys.argv[4]

    # Initialize detector
    detector = FaceDetector(model_path, scaler_path, config_path)

    # Load test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)

    print(f"\nDetecting faces in {image_path}...")

    # Detect
    import time

    start = time.time()
    detections = detector.detect(img, single_face=True)
    elapsed = (time.time() - start) * 1000

    print(f"✓ Detection completed in {elapsed:.1f}ms")
    print(f"✓ Found {len(detections)} face(s)")

    for i, det in enumerate(detections):
        print(f"\nFace {i + 1}:")
        print(f"  Position: ({det['bbox']['x']}, {det['bbox']['y']})")
        print(f"  Size: {det['bbox']['w']}x{det['bbox']['h']}")
        print(f"  Confidence: {det['confidence']:.3f}")

    # Visualize
    vis_img = detector.visualize_detection(img, detections)

    # Save result
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"\n✓ Result saved to {output_path}")
