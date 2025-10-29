"""
Godot Integration Bridge
=========================
Provides a simple HTTP server that Godot can call to perform face detection
and mask overlay. This enables the Python CV pipeline to work with Godot.

Usage:
    python godot_bridge.py --port 5000 --models_dir models --mask assets/mask.png

Godot can then make HTTP requests to:
    POST http://localhost:5000/detect
    Body: { "image_base64": "..." }
    Response: { "faces": [...], "image_base64": "..." }
"""

import sys
import os
import base64
import io
from flask import Flask, request, jsonify
import cv2
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import Inferencer


app = Flask(__name__)
inferencer = None


def decode_base64_image(base64_string):
    """Decode base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_image_base64(img):
    """Encode OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'Face detector is running'
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect faces and overlay mask.
    
    Request JSON:
        {
            "image_base64": "base64 encoded image",
            "overlay_mask": true/false (optional, default true),
            "draw_boxes": true/false (optional, default false)
        }
    
    Response JSON:
        {
            "faces": [
                {"x": int, "y": int, "w": int, "h": int, "score": float},
                ...
            ],
            "image_base64": "base64 encoded result image",
            "num_faces": int
        }
    """
    try:
        data = request.get_json()
        
        if 'image_base64' not in data:
            return jsonify({'error': 'Missing image_base64'}), 400
        
        # Decode image
        img = decode_base64_image(data['image_base64'])
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Options
        overlay_mask = data.get('overlay_mask', True)
        draw_boxes = data.get('draw_boxes', False)
        
        # Detect faces
        boxes, scores = inferencer.detect_faces(img)
        
        # Process output
        output = img.copy()
        
        if overlay_mask and inferencer.mask_overlay is not None and len(boxes) > 0:
            output = inferencer.mask_overlay.overlay_on_faces(output, boxes)
        
        if draw_boxes and len(boxes) > 0:
            from src.utils import draw_face_detections
            output = draw_face_detections(output, boxes, scores)
        
        # Encode result
        result_base64 = encode_image_base64(output)
        
        # Format faces
        faces = [
            {
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'score': float(score)
            }
            for (x, y, w, h), score in zip(boxes, scores)
        ]
        
        return jsonify({
            'faces': faces,
            'image_base64': result_base64,
            'num_faces': len(faces)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_mask', methods=['POST'])
def update_mask():
    """
    Update the mask template.
    
    Request JSON:
        {
            "mask_path": "path/to/new/mask.png"
        }
    """
    try:
        data = request.get_json()
        
        if 'mask_path' not in data:
            return jsonify({'error': 'Missing mask_path'}), 400
        
        mask_path = data['mask_path']
        
        if not os.path.exists(mask_path):
            return jsonify({'error': f'Mask not found: {mask_path}'}), 404
        
        inferencer.set_mask(mask_path)
        
        return jsonify({
            'status': 'ok',
            'message': f'Mask updated to {mask_path}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    global inferencer
    
    parser = argparse.ArgumentParser(description='Godot-Python CV Bridge')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory with trained models')
    parser.add_argument('--mask', type=str, default=None,
                       help='Initial mask PNG path')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GODOT-PYTHON CV BRIDGE")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Models: {args.models_dir}")
    if args.mask:
        print(f"Mask: {args.mask}")
    print("="*60)
    
    # Initialize inferencer
    print("\nInitializing face detector...")
    try:
        inferencer = Inferencer(
            models_dir=args.models_dir,
            mask_path=args.mask
        )
        print("✓ Face detector ready")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nMake sure you've trained models first:")
        print("  python app.py train --pos_dir data/face --neg_dir data/non_face")
        sys.exit(1)
    
    # Start server
    print(f"\n✓ Server starting on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  POST http://{args.host}:{args.port}/detect")
    print(f"  POST http://{args.host}:{args.port}/update_mask")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
