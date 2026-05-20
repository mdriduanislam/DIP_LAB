"""
Test script for DIP Lab Q1 Assignment
Creates two sample grayscale images for testing
"""

import numpy as np
from PIL import Image

def create_sample_images():
    """Create two sample grayscale images for testing"""
    
    # Create sample Image A: Horizontal gradient
    width, height = 256, 256
    img_A = np.linspace(0, 255, width * height).reshape(height, width).astype(np.uint8)
    img_A_pil = Image.fromarray(img_A)
    img_A_pil.save('imageA.jpg')
    print("✓ Created imageA.jpg (horizontal gradient)")
    
    # Create sample Image B: Vertical gradient
    img_B = np.linspace(255, 0, width * height).reshape(height, width).astype(np.uint8)
    img_B_pil = Image.fromarray(img_B)
    img_B_pil.save('imageB.jpg')
    print("✓ Created imageB.jpg (vertical gradient)")
    
    print("\nTest images created. Now run: python 20701058_Q1.py")

if __name__ == "__main__":
    create_sample_images()
