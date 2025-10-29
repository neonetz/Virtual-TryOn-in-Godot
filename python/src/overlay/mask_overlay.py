"""
Mask Overlay Engine
===================
Overlays PNG mask images with alpha channel onto detected face regions.
Handles scaling, positioning, and alpha blending for realistic placement.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os


class MaskOverlay:
    """
    Handles mask PNG overlay on detected face bounding boxes.
    
    High Cohesion: Focuses only on image overlay operations.
    Low Coupling: Independent from detection and classification.
    """
    
    def __init__(
        self,
        mask_path: str,
        scale_factor: float = 1.0,
        x_offset_ratio: float = 0.0,
        y_offset_ratio: float = 0.0
    ):
        """
        Initialize mask overlay engine.
        
        Args:
            mask_path: Path to mask PNG with alpha channel
            scale_factor: Mask size as ratio of face size (1.0 = same size)
            x_offset_ratio: Horizontal offset (-0.5 to 0.5, 0 = centered)
            y_offset_ratio: Vertical offset (-0.5 to 0.5, 0 = centered)
        """
        self.mask_path = mask_path
        self.scale_factor = scale_factor
        self.x_offset_ratio = x_offset_ratio
        self.y_offset_ratio = y_offset_ratio
        
        # Load mask image with alpha channel
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        
        self.mask_template = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.mask_template is None:
            raise ValueError(f"Failed to load mask image from {mask_path}")
        
        # Ensure 4 channels (BGRA)
        if self.mask_template.shape[2] != 4:
            raise ValueError(f"Mask image must have alpha channel (4 channels)")
        
        print(f"Loaded mask template from: {mask_path}")
        print(f"  Shape: {self.mask_template.shape}")
    
    def overlay_on_face(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        preserve_aspect: bool = True
    ) -> np.ndarray:
        """
        Overlay mask on a single face bounding box.
        
        Process:
        1. Calculate target mask dimensions based on face size
        2. Resize mask maintaining aspect ratio
        3. Compute position (centered on face)
        4. Alpha blend mask onto image
        
        Args:
            image: Background image (BGR)
            face_bbox: Face bounding box (x, y, w, h)
            preserve_aspect: Maintain mask aspect ratio during scaling
            
        Returns:
            Image with mask overlaid
        """
        x, y, w, h = face_bbox
        
        # Step 1: Calculate target mask dimensions
        target_width = int(w * self.scale_factor)
        
        if preserve_aspect:
            # Maintain aspect ratio
            aspect = self.mask_template.shape[1] / self.mask_template.shape[0]
            target_height = int(target_width / aspect)
        else:
            # Stretch to fit face dimensions
            target_height = int(h * self.scale_factor)
        
        # Step 2: Resize mask
        mask_resized = cv2.resize(
            self.mask_template,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Step 3: Compute overlay position
        # Center horizontally with optional offset
        overlay_x = x + (w - target_width) // 2 + int(w * self.x_offset_ratio)
        # Center vertically with optional offset
        overlay_y = y + (h - target_height) // 2 + int(h * self.y_offset_ratio)
        
        # Step 4: Alpha blend
        result = self._alpha_blend(
            image,
            mask_resized,
            (overlay_x, overlay_y)
        )
        
        return result
    
    def overlay_on_faces(
        self,
        image: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Overlay mask on multiple detected faces.
        
        Args:
            image: Background image
            face_bboxes: List of face bounding boxes
            
        Returns:
            Image with masks overlaid on all faces
        """
        result = image.copy()
        
        for bbox in face_bboxes:
            result = self.overlay_on_face(result, bbox)
        
        return result
    
    def _alpha_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        position: Tuple[int, int]
    ) -> np.ndarray:
        """
        Alpha blend foreground onto background at specified position.
        
        Handles boundary clamping to prevent out-of-bounds errors.
        
        Args:
            background: Background image (BGR, 3 channels)
            foreground: Foreground image with alpha (BGRA, 4 channels)
            position: Top-left position (x, y) for overlay
            
        Returns:
            Blended image
        """
        result = background.copy()
        
        overlay_x, overlay_y = position
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate valid regions (clamp to image boundaries)
        # Foreground region
        fg_x1 = max(0, -overlay_x)
        fg_y1 = max(0, -overlay_y)
        fg_x2 = min(fg_w, bg_w - overlay_x)
        fg_y2 = min(fg_h, bg_h - overlay_y)
        
        # Background region
        bg_x1 = max(0, overlay_x)
        bg_y1 = max(0, overlay_y)
        bg_x2 = min(bg_w, overlay_x + fg_w)
        bg_y2 = min(bg_h, overlay_y + fg_h)
        
        # Check if overlay is completely outside image
        if fg_x2 <= fg_x1 or fg_y2 <= fg_y1 or bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
            return result
        
        # Extract regions
        fg_region = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
        bg_region = result[bg_y1:bg_y2, bg_x1:bg_x2]
        
        # Separate alpha channel
        fg_rgb = fg_region[:, :, :3]
        fg_alpha = fg_region[:, :, 3:4] / 255.0  # Normalize to [0, 1]
        
        # Alpha blending formula: output = foreground * alpha + background * (1 - alpha)
        blended = (fg_rgb * fg_alpha + bg_region * (1 - fg_alpha)).astype(np.uint8)
        
        # Update result
        result[bg_y1:bg_y2, bg_x1:bg_x2] = blended
        
        return result
    
    def update_mask(self, new_mask_path: str) -> None:
        """
        Load a new mask template.
        
        Args:
            new_mask_path: Path to new mask PNG
        """
        if not os.path.exists(new_mask_path):
            raise FileNotFoundError(f"Mask image not found: {new_mask_path}")
        
        new_mask = cv2.imread(new_mask_path, cv2.IMREAD_UNCHANGED)
        
        if new_mask is None or new_mask.shape[2] != 4:
            raise ValueError(f"Invalid mask image: {new_mask_path}")
        
        self.mask_template = new_mask
        self.mask_path = new_mask_path
        print(f"Updated mask template: {new_mask_path}")
    
    def preview_mask(self, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """
        Generate a preview of the mask template.
        
        Args:
            size: Preview size (width, height)
            
        Returns:
            Mask preview with checkerboard background
        """
        # Create checkerboard background
        checker = self._create_checkerboard(size)
        
        # Resize mask to fit
        mask_resized = cv2.resize(self.mask_template, size, interpolation=cv2.INTER_AREA)
        
        # Overlay mask on checkerboard
        preview = self._alpha_blend(checker, mask_resized, (0, 0))
        
        return preview
    
    def _create_checkerboard(
        self,
        size: Tuple[int, int],
        square_size: int = 20
    ) -> np.ndarray:
        """
        Create a checkerboard pattern background.
        
        Args:
            size: Image size (width, height)
            square_size: Size of each square
            
        Returns:
            Checkerboard image (BGR)
        """
        width, height = size
        board = np.zeros((height, width, 3), dtype=np.uint8)
        
        color1 = (200, 200, 200)
        color2 = (150, 150, 150)
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    board[i:i+square_size, j:j+square_size] = color1
                else:
                    board[i:i+square_size, j:j+square_size] = color2
        
        return board
