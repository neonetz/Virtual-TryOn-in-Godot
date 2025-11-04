# Assets Directory

## Mask Image

Place your face mask image here as `mask.png`.

### Requirements

- **Format**: PNG with alpha channel (RGBA)
- **Size**: Recommended 512×512 pixels or larger
- **Content**: Face mask covering nose, mouth, chin area
- **Background**: Must be transparent (alpha = 0)
- **Orientation**: Mask should be facing forward (frontal view)

### Mask Positioning

The system will automatically position the mask on the lower part of detected faces:

```
mask_width = 90% of face width
mask_height = 55% of face height
mask_x = face_x + 5% of face width
mask_y = face_y + 40% of face height
```

This places the mask over the nose-mouth-chin area.

### Example Masks

You can use masks like:
- Medical/surgical masks
- Halloween masks (vampire, zombie, etc.)
- Gas masks
- Bandanas
- Face paint patterns
- Custom artistic designs

### Creating Your Own Mask

Use any image editor (Photoshop, GIMP, Paint.NET):

1. Create new image with transparent background
2. Draw or import your mask artwork
3. Ensure mask is centered and facing forward
4. Export as PNG with alpha channel
5. Save as `mask.png` in this directory

### Testing Your Mask

```bash
cd backend
python app.py infer --image test.jpg --out result.jpg --mask assets/mask.png --show
```

## Cascades Directory

The `cascades/` subfolder is for Haar cascade XML files (optional).

The system will auto-detect cascades from OpenCV installation, but you can place custom cascades here:

- `haarcascade_frontalface_default.xml` - Face detection
- `haarcascade_eye.xml` - Eye detection (for rotation)

Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades
