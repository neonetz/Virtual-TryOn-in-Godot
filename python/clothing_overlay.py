"""
Clothing Overlay Module
Handles loading and overlaying clothing images onto video frames.
Positions clothing based on detected body bounding box.
"""

import cv2
import numpy as np
import os


class ClothingOverlay:
    """
    Manages clothing PNG files and overlays them onto video frames.
    Positions clothing based on upper-body bounding box coordinates.
    """
    
    def __init__(self, clothes_dir="clothes"):
        """
        Initializes the clothing overlay system.
        
        Args:
            clothes_dir: Directory containing clothing PNG files
        """
        self.clothes_dir = clothes_dir
        self.clothing_items = []
        self.current_index = 0
        self.current_clothing = None
        
        # Overlay positioning parameters
        self.scale_factor = 1.0
        self.vertical_offset = 0
        self.horizontal_offset = 0
        
        # Load available clothing items
        self.load_clothing_catalog()
        
    def load_clothing_catalog(self):
        """
        Scans the clothes directory and loads all PNG files.
        """
        if not os.path.exists(self.clothes_dir):
            print(f"[WARNING] Clothes directory not found: {self.clothes_dir}")
            print(f"[INFO] Creating directory: {self.clothes_dir}")
            os.makedirs(self.clothes_dir, exist_ok=True)
            return
        
        # Find all PNG files in directory
        for filename in os.listdir(self.clothes_dir):
            if filename.lower().endswith('.png'):
                filepath = os.path.join(self.clothes_dir, filename)
                self.clothing_items.append(filepath)
        
        if len(self.clothing_items) == 0:
            print(f"[WARNING] No PNG files found in {self.clothes_dir}")
        else:
            print(f"[INFO] Loaded {len(self.clothing_items)} clothing items")
            # Load first clothing item
            self.load_clothing(0)
    
    def load_clothing(self, index):
        """
        Loads a specific clothing item by index.
        
        Args:
            index: Index in clothing_items list
        """
        if len(self.clothing_items) == 0:
            return
        
        if index < 0 or index >= len(self.clothing_items):
            return
        
        filepath = self.clothing_items[index]
        
        # Load image with alpha channel (transparency)
        clothing_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if clothing_img is None:
            print(f"[ERROR] Failed to load clothing: {filepath}")
            return
        
        # Ensure image has alpha channel
        if clothing_img.shape[2] == 3:
            # Add alpha channel if not present
            alpha = np.ones((clothing_img.shape[0], clothing_img.shape[1], 1), 
                           dtype=clothing_img.dtype) * 255
            clothing_img = np.concatenate([clothing_img, alpha], axis=2)
        
        self.current_clothing = clothing_img
        self.current_index = index
        print(f"[INFO] Loaded clothing: {os.path.basename(filepath)}")
    
    def next_clothing(self):
        """Switches to next clothing item."""
        if len(self.clothing_items) == 0:
            return
        next_index = (self.current_index + 1) % len(self.clothing_items)
        self.load_clothing(next_index)
    
    def previous_clothing(self):
        """Switches to previous clothing item."""
        if len(self.clothing_items) == 0:
            return
        prev_index = (self.current_index - 1) % len(self.clothing_items)
        self.load_clothing(prev_index)
    
    def overlay_clothing(self, frame, bbox):
        """
        Overlays clothing onto the frame based on bounding box.
        
        Args:
            frame: BGR frame to overlay clothing on
            bbox: Tuple (x, y, w, h) of upper-body bounding box
            
        Returns:
            Frame with clothing overlaid
        """
        if self.current_clothing is None or bbox is None:
            return frame
        
        x, y, w, h = bbox
        
        if w == 0 or h == 0:
            return frame
        
        # Calculate clothing dimensions based on bounding box
        clothing_width = int(w * self.scale_factor)
        
        # Maintain aspect ratio
        aspect_ratio = self.current_clothing.shape[0] / self.current_clothing.shape[1]
        clothing_height = int(clothing_width * aspect_ratio)
        
        # Resize clothing to fit bounding box
        resized_clothing = cv2.resize(
            self.current_clothing, 
            (clothing_width, clothing_height), 
            interpolation=cv2.INTER_AREA
        )
        
        # Calculate center position of bounding box
        center_x = x + w // 2 + self.horizontal_offset
        center_y = y + h // 2 + self.vertical_offset
        
        # Calculate top-left corner for overlay
        overlay_x = center_x - clothing_width // 2
        overlay_y = center_y - clothing_height // 2
        
        # Ensure overlay is within frame boundaries
        overlay_x = max(0, min(overlay_x, frame.shape[1] - clothing_width))
        overlay_y = max(0, min(overlay_y, frame.shape[0] - clothing_height))
        
        # Perform alpha blending
        frame = self.alpha_blend(
            frame, 
            resized_clothing, 
            overlay_x, 
            overlay_y
        )
        
        return frame
    
    def alpha_blend(self, background, foreground, x, y):
        """
        Blends foreground image with transparency onto background.
        
        Args:
            background: Background BGR image
            foreground: Foreground BGRA image (with alpha channel)
            x, y: Top-left position to place foreground
            
        Returns:
            Blended image
        """
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Ensure overlay fits within background
        if x + fg_w > bg_w:
            fg_w = bg_w - x
            foreground = foreground[:, :fg_w]
        
        if y + fg_h > bg_h:
            fg_h = bg_h - y
            foreground = foreground[:fg_h, :]
        
        if fg_w <= 0 or fg_h <= 0:
            return background
        
        # Extract regions of interest
        bg_roi = background[y:y+fg_h, x:x+fg_w]
        
        # Separate color and alpha channels
        fg_bgr = foreground[:, :, :3]
        fg_alpha = foreground[:, :, 3:4] / 255.0
        
        # Alpha blending formula: result = fg * alpha + bg * (1 - alpha)
        blended = (fg_bgr * fg_alpha + bg_roi * (1 - fg_alpha)).astype(np.uint8)
        
        # Place blended region back into background
        background[y:y+fg_h, x:x+fg_w] = blended
        
        return background
    
    def set_scale_factor(self, factor):
        """Sets the scale factor for clothing overlay."""
        self.scale_factor = max(0.5, min(factor, 2.0))
    
    def adjust_scale(self, delta):
        """Adjusts scale factor by delta amount."""
        self.scale_factor = max(0.5, min(self.scale_factor + delta, 2.0))
    
    def set_vertical_offset(self, offset):
        """Sets vertical position offset."""
        self.vertical_offset = offset
    
    def set_horizontal_offset(self, offset):
        """Sets horizontal position offset."""
        self.horizontal_offset = offset
