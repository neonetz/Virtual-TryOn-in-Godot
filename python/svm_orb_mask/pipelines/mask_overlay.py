"""
Mask Overlay Module
Handles loading PNG masks with transparency and overlaying them on face ROIs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class MaskOverlay:
    """
    Handles mask image loading, transformation, and overlay on face regions.
    """
    
    def __init__(self, mask_path: str, use_rotation: bool = True):
        """
        Initialize mask overlay system.
        
        Args:
            mask_path: Path to mask PNG file with alpha channel
            use_rotation: Enable eye-based rotation alignment
        """
        self.mask_path = Path(mask_path)
        self.use_rotation = use_rotation
        
        # Load mask image
        self.mask_template = self._load_mask()
        
        print(f"[Mask Overlay] Loaded mask: {mask_path}")
        print(f"[Mask Overlay] Mask size: {self.mask_template.shape[:2]}")
        print(f"[Mask Overlay] Rotation alignment: {'enabled' if use_rotation else 'disabled'}")
    
    def _load_mask(self) -> np.ndarray:
        """
        Load mask image with alpha channel.
        
        Returns:
            Mask image as BGRA array
        """
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {self.mask_path}")
        
        # Load with alpha channel
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            raise ValueError(f"Failed to load mask image: {self.mask_path}")
        
        # Ensure BGRA format
        if mask.shape[2] == 3:
            # Add full opacity alpha channel
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
        
        return mask
    
    def scale_mask(self, 
                   face_box: Tuple[int, int, int, int],
                   scale_factor: float = 1.0) -> np.ndarray:
        """
        Scale mask to match face size.
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            scale_factor: Scale adjustment (1.0 = face width, 1.2 = 120% of face)
            
        Returns:
            Scaled mask image
        """
        _, _, face_w, face_h = face_box
        
        # Target mask width (scaled by factor)
        target_width = int(face_w * scale_factor)
        
        # Maintain aspect ratio
        mask_h, mask_w = self.mask_template.shape[:2]
        aspect_ratio = mask_h / mask_w
        target_height = int(target_width * aspect_ratio)
        
        # Resize mask
        scaled_mask = cv2.resize(
            self.mask_template, 
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        
        return scaled_mask
    
    def rotate_mask(self, mask: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate mask by given angle (for eye alignment).
        
        Args:
            mask: Input mask image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated mask image
        """
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate with transparent background
        rotated = cv2.warpAffine(
            mask, 
            M, 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # Transparent
        )
        
        return rotated
    
    def compute_mask_position(self,
                            face_box: Tuple[int, int, int, int],
                            mask: np.ndarray,
                            offset_y: float = 0.3) -> Tuple[int, int]:
        """
        Compute mask center position relative to face box.
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            mask: Scaled mask image
            offset_y: Vertical offset as fraction of face height (0.3 = 30% down)
            
        Returns:
            Mask top-left position (x, y)
        """
        face_x, face_y, face_w, face_h = face_box
        mask_h, mask_w = mask.shape[:2]
        
        # Center mask horizontally on face
        mask_x = face_x + (face_w - mask_w) // 2
        
        # Position mask with vertical offset
        mask_y = face_y + int(face_h * offset_y)
        
        return mask_x, mask_y
    
    def alpha_blend(self,
                   background: np.ndarray,
                   overlay: np.ndarray,
                   position: Tuple[int, int]) -> np.ndarray:
        """
        Alpha blend overlay onto background at given position.
        
        Args:
            background: Background image (BGR)
            overlay: Overlay image with alpha channel (BGRA)
            position: Overlay top-left position (x, y)
            
        Returns:
            Blended image
        """
        x, y = position
        overlay_h, overlay_w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Clip overlay to fit within background
        if x < 0 or y < 0 or x + overlay_w > bg_w or y + overlay_h > bg_h:
            # Compute valid region
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(bg_w, x + overlay_w)
            y_end = min(bg_h, y + overlay_h)
            
            # Compute corresponding overlay region
            overlay_x_start = max(0, -x)
            overlay_y_start = max(0, -y)
            overlay_x_end = overlay_x_start + (x_end - x_start)
            overlay_y_end = overlay_y_start + (y_end - y_start)
            
            # Extract valid regions
            bg_region = background[y_start:y_end, x_start:x_end]
            overlay_region = overlay[overlay_y_start:overlay_y_end, 
                                   overlay_x_start:overlay_x_end]
        else:
            # Full overlay fits
            bg_region = background[y:y+overlay_h, x:x+overlay_w]
            overlay_region = overlay
        
        # Extract alpha channel
        alpha = overlay_region[:, :, 3] / 255.0
        
        # Perform alpha blending
        for c in range(3):  # BGR channels
            bg_region[:, :, c] = (
                alpha * overlay_region[:, :, c] +
                (1 - alpha) * bg_region[:, :, c]
            )
        
        return background
    
    def apply_mask(self,
                  image: np.ndarray,
                  face_box: Tuple[int, int, int, int],
                  rotation_angle: Optional[float] = None,
                  scale_factor: float = 1.2,
                  offset_y: float = 0.3) -> np.ndarray:
        """
        Apply mask to image at face location.
        
        Args:
            image: Input image (BGR)
            face_box: Face bounding box (x, y, w, h)
            rotation_angle: Optional rotation angle for eye alignment
            scale_factor: Mask scale relative to face width
            offset_y: Vertical offset as fraction of face height
            
        Returns:
            Image with mask overlaid
        """
        # Create copy to avoid modifying original
        result = image.copy()
        
        # Scale mask to face size
        scaled_mask = self.scale_mask(face_box, scale_factor)
        
        # Rotate mask if angle provided and rotation enabled
        if self.use_rotation and rotation_angle is not None:
            scaled_mask = self.rotate_mask(scaled_mask, rotation_angle)
        
        # Compute mask position
        mask_pos = self.compute_mask_position(face_box, scaled_mask, offset_y)
        
        # Alpha blend mask onto image
        result = self.alpha_blend(result, scaled_mask, mask_pos)
        
        return result
    
    def apply_masks_batch(self,
                         image: np.ndarray,
                         face_boxes: list,
                         rotation_angles: Optional[list] = None,
                         scale_factor: float = 1.2,
                         offset_y: float = 0.3) -> np.ndarray:
        """
        Apply mask to multiple faces in image.
        
        Args:
            image: Input image (BGR)
            face_boxes: List of face bounding boxes
            rotation_angles: Optional list of rotation angles (one per face)
            scale_factor: Mask scale relative to face width
            offset_y: Vertical offset as fraction of face height
            
        Returns:
            Image with masks overlaid on all faces
        """
        result = image.copy()
        
        for i, face_box in enumerate(face_boxes):
            # Get rotation angle for this face (if available)
            angle = None
            if rotation_angles is not None and i < len(rotation_angles):
                angle = rotation_angles[i]
            
            # Apply mask
            result = self.apply_mask(
                result, 
                face_box, 
                angle, 
                scale_factor, 
                offset_y
            )
        
        return result


class MaskLibrary:
    """
    Manages multiple mask options for runtime switching.
    """
    
    def __init__(self, masks_dir: str):
        """
        Initialize mask library from directory.
        
        Args:
            masks_dir: Directory containing mask PNG files
        """
        self.masks_dir = Path(masks_dir)
        self.masks = {}
        self.current_mask = None
        
        # Load all masks
        self._load_all_masks()
    
    def _load_all_masks(self):
        """Load all PNG masks from directory."""
        if not self.masks_dir.exists():
            print(f"[Mask Library] Warning: Directory not found: {self.masks_dir}")
            return
        
        mask_files = list(self.masks_dir.glob("*.png"))
        
        for mask_file in mask_files:
            mask_name = mask_file.stem
            try:
                mask_overlay = MaskOverlay(str(mask_file))
                self.masks[mask_name] = mask_overlay
                print(f"[Mask Library] Loaded mask: {mask_name}")
            except Exception as e:
                print(f"[Mask Library] Failed to load {mask_name}: {e}")
        
        # Set first mask as current
        if self.masks:
            self.current_mask = list(self.masks.keys())[0]
            print(f"[Mask Library] Current mask: {self.current_mask}")
    
    def get_mask(self, mask_name: Optional[str] = None) -> Optional[MaskOverlay]:
        """
        Get mask overlay by name.
        
        Args:
            mask_name: Name of mask (None = current mask)
            
        Returns:
            MaskOverlay instance or None if not found
        """
        if mask_name is None:
            mask_name = self.current_mask
        
        return self.masks.get(mask_name)
    
    def set_current_mask(self, mask_name: str) -> bool:
        """
        Set current active mask.
        
        Args:
            mask_name: Name of mask to activate
            
        Returns:
            True if successful, False if mask not found
        """
        if mask_name in self.masks:
            self.current_mask = mask_name
            print(f"[Mask Library] Switched to mask: {mask_name}")
            return True
        else:
            print(f"[Mask Library] Mask not found: {mask_name}")
            return False
    
    def list_masks(self) -> list:
        """Get list of available mask names."""
        return list(self.masks.keys())
