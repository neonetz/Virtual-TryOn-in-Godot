"""
Tkinter GUI for Virtual Try-On
================================
Simple desktop GUI for real-time webcam mask overlay with mask switching.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
from typing import Optional, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference import Inferencer


class VirtualTryOnGUI:
    """
    Desktop GUI for Virtual Try-On system.
    
    Features:
    - Real-time webcam feed
    - Multiple mask selection buttons
    - FPS counter
    - Face detection visualization
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Virtual Try-On System")
        self.root.geometry("1200x800")
        
        # State variables
        self.inferencer: Optional[Inferencer] = None
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_mask: Optional[str] = None
        self.available_masks: List[str] = []
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create UI
        self._create_ui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _create_ui(self) -> None:
        """Create GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E))
        
        # Title
        title = ttk.Label(control_frame, text="Virtual Try-On", font=('Arial', 18, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Load models button
        self.load_btn = ttk.Button(
            control_frame,
            text="Load Models",
            command=self.load_models
        )
        self.load_btn.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Start/Stop button
        self.start_btn = ttk.Button(
            control_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            state='disabled'
        )
        self.start_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E)
        )
        
        # Mask selection label
        mask_label = ttk.Label(control_frame, text="Select Mask:", font=('Arial', 12, 'bold'))
        mask_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Mask buttons frame
        self.mask_frame = ttk.Frame(control_frame)
        self.mask_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Add Mask button
        add_mask_btn = ttk.Button(
            control_frame,
            text="Add Mask PNG",
            command=self.add_mask
        )
        add_mask_btn.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Remove All Masks button
        remove_all_btn = ttk.Button(
            control_frame,
            text="Remove All Masks",
            command=self.remove_all_masks
        )
        remove_all_btn.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=8, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E)
        )
        
        # FPS label
        self.fps_label = ttk.Label(
            control_frame,
            text="FPS: 0.0",
            font=('Arial', 10)
        )
        self.fps_label.grid(row=9, column=0, columnspan=2, pady=5)
        
        # Status label
        self.status_label = ttk.Label(
            control_frame,
            text="Status: Ready",
            font=('Arial', 10),
            foreground='green'
        )
        self.status_label.grid(row=10, column=0, columnspan=2, pady=5)
        
        # Right panel - Video display
        video_frame = ttk.Frame(main_frame, padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=960, height=720)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
    
    def load_models(self) -> None:
        """Load trained models."""
        # Ask for models directory
        models_dir = filedialog.askdirectory(
            title="Select Models Directory",
            initialdir=os.path.join(os.path.dirname(__file__), 'models')
        )
        
        if not models_dir:
            return
        
        try:
            self.update_status("Loading models...", 'orange')
            self.root.update()
            
            # Create inferencer
            self.inferencer = Inferencer(models_dir=models_dir)
            
            self.update_status("Models loaded successfully!", 'green')
            self.start_btn['state'] = 'normal'
            
            messagebox.showinfo("Success", "Models loaded successfully!")
        
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 'red')
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def add_mask(self) -> None:
        """Add a mask PNG to the selection."""
        # Ask for mask file
        mask_path = filedialog.askopenfilename(
            title="Select Mask PNG",
            initialdir=os.path.join(os.path.dirname(__file__), '..', 'assets', 'masks'),
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not mask_path:
            return
        
        # Verify it's a valid image
        try:
            img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Invalid image file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask:\n{str(e)}")
            return
        
        # Add to list
        self.available_masks.append(mask_path)
        
        # Update UI
        self._update_mask_buttons()
        
        messagebox.showinfo("Success", f"Mask added: {os.path.basename(mask_path)}")
    
    def remove_all_masks(self) -> None:
        """Remove all masks from selection."""
        self.available_masks.clear()
        self.current_mask = None
        self._update_mask_buttons()
        messagebox.showinfo("Success", "All masks removed")
    
    def _update_mask_buttons(self) -> None:
        """Update mask selection buttons."""
        # Clear existing buttons
        for widget in self.mask_frame.winfo_children():
            widget.destroy()
        
        # Create button for each mask
        for i, mask_path in enumerate(self.available_masks):
            mask_name = os.path.basename(mask_path)
            
            btn = ttk.Button(
                self.mask_frame,
                text=mask_name,
                command=lambda p=mask_path: self.select_mask(p)
            )
            btn.grid(row=i, column=0, pady=2, sticky=(tk.W, tk.E))
    
    def select_mask(self, mask_path: str) -> None:
        """
        Select a mask for overlay.
        
        Args:
            mask_path: Path to mask PNG
        """
        self.current_mask = mask_path
        self.update_status(f"Mask: {os.path.basename(mask_path)}", 'green')
    
    def toggle_webcam(self) -> None:
        """Start or stop webcam."""
        if not self.running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self) -> None:
        """Start webcam capture."""
        if self.inferencer is None:
            messagebox.showerror("Error", "Please load models first!")
            return
        
        try:
            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open webcam")
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.running = True
            self.start_btn['text'] = "Stop Webcam"
            self.update_status("Webcam running...", 'green')
            
            # Reset FPS tracking
            self.frame_count = 0
            self.start_time = time.time()
            
            # Start video thread
            thread = threading.Thread(target=self._video_loop, daemon=True)
            thread.start()
        
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 'red')
            messagebox.showerror("Error", f"Failed to start webcam:\n{str(e)}")
    
    def stop_webcam(self) -> None:
        """Stop webcam capture."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.start_btn['text'] = "Start Webcam"
        self.update_status("Webcam stopped", 'orange')
    
    def _video_loop(self) -> None:
        """Main video processing loop (runs in thread)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            try:
                # Detect faces
                faces, scores = self.inferencer.detect_faces(frame)
                
                # Overlay mask if selected
                if self.current_mask and len(faces) > 0:
                    # Set mask if not already set
                    if self.inferencer.mask_overlay is None or \
                       self.inferencer.mask_overlay.mask_path != self.current_mask:
                        self.inferencer.set_mask(self.current_mask)
                    
                    # Apply mask overlay
                    frame = self.inferencer.mask_overlay.overlay_on_faces(frame, faces)
                
                # Draw bounding boxes
                for ((x, y, w, h), conf) in zip(faces, scores):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"Face: {conf:.2f}"
                    cv2.putText(
                        frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
            
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Display frame
            self._display_frame(frame)
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _display_frame(self, frame: np.ndarray) -> None:
        """
        Display frame on canvas.
        
        Args:
            frame: OpenCV frame (BGR)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))
        else:
            frame_resized = frame_rgb
        
        # Convert to PIL Image
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.video_canvas.image = img_tk  # Keep reference
        
        # Update FPS label
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
    
    def update_status(self, message: str, color: str = 'black') -> None:
        """
        Update status label.
        
        Args:
            message: Status message
            color: Text color
        """
        self.status_label.config(text=f"Status: {message}", foreground=color)
    
    def on_close(self) -> None:
        """Handle window close event."""
        self.stop_webcam()
        self.root.destroy()


def main():
    """Main GUI application."""
    root = tk.Tk()
    app = VirtualTryOnGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
