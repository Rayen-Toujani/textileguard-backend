from PIL import Image, ImageOps
import numpy as np
from typing import List

def extract_patches(image: Image.Image, patch_size: int = 64) -> List[Image.Image]:
    """
    Extract 64x64 patches from a larger image (TILDA-style)
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to square (preserve aspect ratio)
    width, height = image.size
    size = min(width, height)
    image = image.crop(((width - size) // 2, 
                        (height - size) // 2,
                        (width + size) // 2,
                        (height + size) // 2))
    
    # Resize to multiple of patch_size
    target_size = (size // patch_size) * patch_size
    if target_size < patch_size:
        target_size = patch_size
    
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Extract patches
    patches = []
    for y in range(0, target_size, patch_size):
        for x in range(0, target_size, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            # Auto-contrast normalization
            patch = ImageOps.autocontrast(patch)
            # Convert to RGB for YOLO
            patch = patch.convert('RGB')
            patches.append(patch)
    
    return patches