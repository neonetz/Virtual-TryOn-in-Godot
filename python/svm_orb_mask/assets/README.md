# Assets Directory

This directory contains resources used by the SVM+ORB face detector.

## Required Assets

### mask.png
- **Purpose**: Default mask overlay image
- **Format**: PNG with alpha channel (transparency)
- **Resolution**: High-resolution recommended (will be auto-scaled)
- **Content**: Face mask, sunglasses, hat, or any overlay graphic

**Example**: Place your `mask.png` file here before running the application.

### cascades/ (Optional)
Contains custom Haar Cascade XML files for face/eye detection.

If not provided, the system will use OpenCV's built-in cascades:
- `haarcascade_frontalface_default.xml` (face detection)
- `haarcascade_eye.xml` (eye detection for rotation alignment)

## Creating Custom Masks

1. Use any image editor (Photoshop, GIMP, etc.)
2. Create transparent background (alpha channel)
3. Draw your mask design
4. Export as PNG with transparency
5. Place in `assets/mask.png`

## Mask Design Tips

- **Resolution**: Use high resolution (1000x1000+) for quality
- **Transparency**: Use gradual alpha for smooth edges
- **Aspect Ratio**: Match typical face proportions (4:5 ratio)
- **Content**: Design should align with face features (eyes, nose, mouth)
