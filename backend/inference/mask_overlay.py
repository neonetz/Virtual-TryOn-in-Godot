"""
Mask Overlay Module
Handles Halloween mask overlay with perspective transformation
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class LandmarkEstimator:
    """Geometric facial landmark estimation from bounding box"""

    @staticmethod
    def estimate_5_points(bbox: Dict) -> Dict:
        """
        Estimate 5 facial landmarks from bounding box
        Uses geometric proportions based on average face structure

        Args:
            bbox: Dictionary with keys {x, y, w, h}

        Returns:
            Dictionary with landmark coordinates:
            - left_eye: [x, y]
            - right_eye: [x, y]
            - nose: [x, y]
            - left_mouth: [x, y]
            - right_mouth: [x, y]
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Geometric proportions (based on facial anatomy)
        # Eyes: ~35% from top, ~25% and 75% horizontally
        left_eye_x = int(x + w * 0.30)
        left_eye_y = int(y + h * 0.35)

        right_eye_x = int(x + w * 0.70)
        right_eye_y = int(y + h * 0.35)

        # Nose: center, ~55% from top
        nose_x = int(x + w * 0.50)
        nose_y = int(y + h * 0.55)

        # Mouth corners: ~75% from top, ~30% and 70% horizontally
        left_mouth_x = int(x + w * 0.35)
        left_mouth_y = int(y + h * 0.75)

        right_mouth_x = int(x + w * 0.65)
        right_mouth_y = int(y + h * 0.75)

        return {
            "left_eye": [left_eye_x, left_eye_y],
            "right_eye": [right_eye_x, right_eye_y],
            "nose": [nose_x, nose_y],
            "left_mouth": [left_mouth_x, left_mouth_y],
            "right_mouth": [right_mouth_x, right_mouth_y],
        }

    @staticmethod
    def estimate_head_pose(landmarks: Dict) -> Dict:
        """
        Estimate head pose (pitch, yaw, roll) from landmarks

        Args:
            landmarks: Dictionary with facial landmarks

        Returns:
            Dictionary with angles in degrees:
            - pitch: up/down rotation
            - yaw: left/right rotation
            - roll: tilt rotation
        """
        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])
        nose = np.array(landmarks["nose"])

        # Calculate eye center
        eye_center = (left_eye + right_eye) / 2

        # Calculate roll (head tilt)
        eye_vector = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

        # Calculate yaw (left/right turn)
        # Based on nose position relative to eye center
        nose_offset_x = nose[0] - eye_center[0]
        inter_eye_distance = np.linalg.norm(right_eye - left_eye)
        yaw = np.degrees(
            np.arcsin(np.clip(nose_offset_x / (inter_eye_distance * 0.5), -1, 1))
        )

        # Calculate pitch (up/down)
        # Based on vertical distance between eyes and nose
        nose_offset_y = nose[1] - eye_center[1]
        pitch = np.degrees(np.arctan2(nose_offset_y, inter_eye_distance))

        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


class MaskOverlay:
    """Apply Halloween mask overlay with perspective transformation"""

    def __init__(self, mask_path: str):
        """
        Initialize mask overlay

        Args:
            mask_path: Path to mask image (PNG with alpha channel)
        """
        # Load mask with alpha channel
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.mask is None:
            raise ValueError(f"Could not load mask from {mask_path}")

        if self.mask.shape[2] != 4:
            raise ValueError(f"Mask must have alpha channel (RGBA)")

        self.mask_h, self.mask_w = self.mask.shape[:2]

        # Define mask anchor points (in mask coordinate space)
        # These correspond to facial landmarks
        self.mask_anchors = {
            "left_eye": [self.mask_w * 0.30, self.mask_h * 0.35],
            "right_eye": [self.mask_w * 0.70, self.mask_h * 0.35],
            "nose": [self.mask_w * 0.50, self.mask_h * 0.55],
            "left_mouth": [self.mask_w * 0.35, self.mask_h * 0.75],
            "right_mouth": [self.mask_w * 0.65, self.mask_h * 0.75],
        }

        print(f"✓ Mask loaded: {mask_path}")
        print(f"  Size: {self.mask_w}x{self.mask_h}")

    def simple_overlay(
        self, img: np.ndarray, bbox: Dict, alpha: float = 0.9
    ) -> np.ndarray:
        """
        Simple mask overlay without perspective correction
        Just resize and blend

        Args:
            img: Input image
            bbox: Face bounding box {x, y, w, h}
            alpha: Mask opacity (0-1)

        Returns:
            Image with mask applied
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Ensure within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        # Resize mask to match face size
        mask_resized = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_AREA)

        # Split into color and alpha channels
        mask_rgb = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3:4] / 255.0 * alpha

        # Extract ROI
        roi = img[y : y + h, x : x + w]

        # Alpha blending
        blended = (mask_rgb * mask_alpha + roi * (1 - mask_alpha)).astype(np.uint8)

        # Place back into image
        result = img.copy()
        result[y : y + h, x : x + w] = blended

        return result

    def perspective_overlay(
        self, img: np.ndarray, landmarks: Dict, alpha: float = 0.9
    ) -> np.ndarray:
        """
        Apply mask with perspective transformation
        Uses facial landmarks for realistic alignment

        Args:
            img: Input image
            landmarks: Facial landmarks dictionary
            alpha: Mask opacity (0-1)

        Returns:
            Image with mask applied
        """
        # Get source points (mask anchor points)
        src_points = np.float32(
            [
                self.mask_anchors["left_eye"],
                self.mask_anchors["right_eye"],
                self.mask_anchors["nose"],
                self.mask_anchors["left_mouth"],
            ]
        )

        # Get destination points (face landmarks)
        dst_points = np.float32(
            [
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["nose"],
                landmarks["left_mouth"],
            ]
        )

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Calculate bounding box for warped mask
        # Transform all four corners of mask
        mask_corners = np.float32(
            [[0, 0], [self.mask_w, 0], [self.mask_w, self.mask_h], [0, self.mask_h]]
        )

        # Transform corners to find bounding box
        mask_corners_transformed = cv2.perspectiveTransform(
            mask_corners.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        # Get bounding box
        min_x = int(np.floor(mask_corners_transformed[:, 0].min()))
        max_x = int(np.ceil(mask_corners_transformed[:, 0].max()))
        min_y = int(np.floor(mask_corners_transformed[:, 1].min()))
        max_y = int(np.ceil(mask_corners_transformed[:, 1].max()))

        # Clip to image bounds
        min_x = max(0, min_x)
        max_x = min(img.shape[1], max_x)
        min_y = max(0, min_y)
        max_y = min(img.shape[0], max_y)

        # Adjust transform matrix for ROI
        M_adjusted = M.copy()
        M_adjusted[0, 2] -= min_x
        M_adjusted[1, 2] -= min_y

        # Warp mask
        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return img  # Invalid ROI, return original

        warped_mask = cv2.warpPerspective(
            self.mask,
            M_adjusted,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Split channels
        mask_rgb = warped_mask[:, :, :3]
        mask_alpha = warped_mask[:, :, 3:4] / 255.0 * alpha

        # Extract ROI from original image
        roi = img[min_y:max_y, min_x:max_x]

        # Handle size mismatches
        h_min = min(roi.shape[0], mask_rgb.shape[0])
        w_min = min(roi.shape[1], mask_rgb.shape[1])

        roi = roi[:h_min, :w_min]
        mask_rgb = mask_rgb[:h_min, :w_min]
        mask_alpha = mask_alpha[:h_min, :w_min]

        # Alpha blending
        blended = (mask_rgb * mask_alpha + roi * (1 - mask_alpha)).astype(np.uint8)

        # Place back into image
        result = img.copy()
        result[min_y : min_y + h_min, min_x : min_x + w_min] = blended

        return result

    def apply(
        self,
        img: np.ndarray,
        detection: Dict,
        use_perspective: bool = True,
        alpha: float = 0.9,
    ) -> np.ndarray:
        """
        Apply mask overlay to detected face

        Args:
            img: Input image
            detection: Detection dictionary with 'bbox' and optional 'landmarks'
            use_perspective: If True, use perspective transform; else simple resize
            alpha: Mask opacity (0-1)

        Returns:
            Image with mask applied
        """
        bbox = detection["bbox"]

        if use_perspective:
            # Estimate landmarks if not provided
            if "landmarks" not in detection:
                landmarks = LandmarkEstimator.estimate_5_points(bbox)
            else:
                landmarks = detection["landmarks"]

            return self.perspective_overlay(img, landmarks, alpha)
        else:
            return self.simple_overlay(img, bbox, alpha)

    def visualize_landmarks(self, img: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw landmarks on image for debugging

        Args:
            img: Input image
            landmarks: Landmarks dictionary

        Returns:
            Image with landmarks drawn
        """
        vis_img = img.copy()

        colors = {
            "left_eye": (0, 255, 0),
            "right_eye": (0, 255, 0),
            "nose": (255, 0, 0),
            "left_mouth": (0, 0, 255),
            "right_mouth": (0, 0, 255),
        }

        for name, point in landmarks.items():
            x, y = int(point[0]), int(point[1])
            color = colors.get(name, (255, 255, 255))
            cv2.circle(vis_img, (x, y), 3, color, -1)
            cv2.putText(
                vis_img,
                name.split("_")[0],
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )

        return vis_img


class MaskManager:
    """Manage multiple Halloween masks"""

    def __init__(self):
        self.masks = {}

    def load_mask(self, mask_id: str, mask_path: str):
        """Load a mask and store with ID"""
        try:
            self.masks[mask_id] = MaskOverlay(mask_path)
            print(f"✓ Loaded mask '{mask_id}' from {mask_path}")
        except Exception as e:
            print(f"✗ Failed to load mask '{mask_id}': {e}")

    def get_mask(self, mask_id: str) -> Optional[MaskOverlay]:
        """Get mask by ID"""
        return self.masks.get(mask_id)

    def list_masks(self):
        """List all loaded masks"""
        return list(self.masks.keys())

    def apply_mask(
        self,
        img: np.ndarray,
        detection: Dict,
        mask_id: str,
        use_perspective: bool = True,
        alpha: float = 0.9,
    ) -> Optional[np.ndarray]:
        """
        Apply specific mask to detected face

        Args:
            img: Input image
            detection: Detection dictionary
            mask_id: ID of mask to apply
            use_perspective: Use perspective transform
            alpha: Mask opacity

        Returns:
            Image with mask applied, or None if mask not found
        """
        mask = self.get_mask(mask_id)
        if mask is None:
            print(f"✗ Mask '{mask_id}' not found")
            return None

        return mask.apply(img, detection, use_perspective, alpha)


if __name__ == "__main__":
    # Test mask overlay
    import sys

    if len(sys.argv) < 3:
        print("Usage: python mask_overlay.py <image.jpg> <mask.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)

    # Create dummy detection (for testing)
    h, w = img.shape[:2]
    dummy_detection = {
        "bbox": {"x": w // 4, "y": h // 4, "w": w // 2, "h": h // 2},
        "confidence": 1.0,
    }

    # Load mask
    mask_overlay = MaskOverlay(mask_path)

    # Apply with perspective
    print("\nApplying mask with perspective transform...")
    result_perspective = mask_overlay.apply(img, dummy_detection, use_perspective=True)

    # Apply simple
    print("Applying mask with simple resize...")
    result_simple = mask_overlay.apply(img, dummy_detection, use_perspective=False)

    # Save results
    cv2.imwrite("result_perspective.jpg", result_perspective)
    cv2.imwrite("result_simple.jpg", result_simple)

    print("\n✓ Results saved:")
    print("  - result_perspective.jpg")
    print("  - result_simple.jpg")

    # Visualize landmarks
    landmarks = LandmarkEstimator.estimate_5_points(dummy_detection["bbox"])
    vis_landmarks = mask_overlay.visualize_landmarks(img, landmarks)
    cv2.imwrite("landmarks_visualization.jpg", vis_landmarks)
    print("  - landmarks_visualization.jpg")
