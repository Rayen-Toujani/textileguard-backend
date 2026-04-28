from PIL import Image, ImageOps
from typing import List

def extract_patches(image: Image.Image, patch_size: int = 64) -> List[Image.Image]:
    """
    Extract 64x64 patches from a larger image (TILDA-style preprocessing)
    
    Process:
    1. Convert to grayscale
    2. Resize to square
    3. Extract non-overlapping 64x64 patches
    4. Apply auto-contrast normalization
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Make image square (crop to center)
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    
    # Resize to multiple of patch_size
    num_patches = max(1, size // patch_size)
    target_size = num_patches * patch_size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Extract patches
    patches = []
    for y in range(0, target_size, patch_size):
        for x in range(0, target_size, patch_size):
            # Extract patch
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            
            # Normalize contrast (like TILDA dataset)
            patch = ImageOps.autocontrast(patch)
            
            # Convert back to RGB for YOLO (needs 3 channels)
            patch_rgb = patch.convert('RGB')
            
            patches.append(patch_rgb)
    
    return patches


def preprocess_single_patch(image: Image.Image) -> Image.Image:
    """
    Preprocess a single image to TILDA format (64x64 grayscale with auto-contrast)
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 64x64
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    
    # Auto-contrast normalization
    image = ImageOps.autocontrast(image)
    
    # Convert to RGB for YOLO
    image = image.convert('RGB')
    
    return image