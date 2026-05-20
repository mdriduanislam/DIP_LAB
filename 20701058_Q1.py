import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_images(path_A, path_B):
    """
    DIP Lab Q1 Assignment: Process two grayscale images
    1. Reads two grayscale images A and B
    2. Applies linear operation: C_linear = 0.5*A + 0.5*B (weighted average)
    3. Applies nonlinear operation: C_nonlinear = abs(A-B) (absolute difference)
    4. Displays all four results visually
    """
    
    try:
        # Step 1: Read two grayscale images A and B using PIL
        img_A_pil = Image.open(path_A).convert('L')  # Convert to grayscale
        img_B_pil = Image.open(path_B).convert('L')  # Convert to grayscale
        
        print(f"✓ Image A loaded: {img_A_pil.size}")
        print(f"✓ Image B loaded: {img_B_pil.size}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not read one or both images. {e}")
        return
    
    # Convert PIL images to NumPy arrays
    img_A = np.array(img_A_pil)
    img_B = np.array(img_B_pil)
    
    # Ensure images are the same size (resize B to match A if necessary)
    if img_A.shape != img_B.shape:
        print(f"⚠ Images have different sizes: A={img_A.shape}, B={img_B.shape}")
        print(f"  Resizing Image B to match Image A...")
        # Simple nearest-neighbor resize using NumPy
        img_B = np.array(Image.fromarray(img_B).resize((img_A.shape[1], img_A.shape[0]), Image.NEAREST))
    
    # Convert to float32 for precise mathematical operations
    A = img_A.astype(np.float32)
    B = img_B.astype(np.float32)
    
    # Step 2: Apply LINEAR operation: C_linear = 0.5*A + 0.5*B
    C_linear = 0.5 * A + 0.5 * B
    C_linear = np.clip(C_linear, 0, 255).astype(np.uint8)
    print("✓ Linear operation applied: C_linear = 0.5*A + 0.5*B")
    
    # Step 3: Apply NONLINEAR operation: C_nonlinear = abs(A-B)
    C_nonlinear = np.abs(A - B)
    C_nonlinear = np.clip(C_nonlinear, 0, 255).astype(np.uint8)
    print("✓ Nonlinear operation applied: C_nonlinear = |A - B|")
    
    # Step 4: Display all four images visually
    plt.figure(figsize=(14, 10))
    
    # Original Image A
    plt.subplot(2, 2, 1)
    plt.imshow(img_A, cmap='gray')
    plt.title('Original Image A', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Original Image B
    plt.subplot(2, 2, 2)
    plt.imshow(img_B, cmap='gray')
    plt.title('Original Image B', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Linear Combination Result
    plt.subplot(2, 2, 3)
    plt.imshow(C_linear, cmap='gray')
    plt.title('Linear: C = 0.5*A + 0.5*B', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Absolute Difference Result
    plt.subplot(2, 2, 4)
    plt.imshow(C_nonlinear, cmap='gray')
    plt.title('Nonlinear: C = |A - B|', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    print("\n✓ Displaying results...")
    plt.show()

if __name__ == "__main__":
    # Specify the image paths here (ensure these files exist in your workspace)
    image_file_A = 'imageA.jpg' 
    image_file_B = 'imageB.jpg'
    
    print("=" * 60)
    print("DIP Lab Q1 Assignment: Image Processing Operations")
    print("=" * 60)
    process_images(image_file_A, image_file_B)
    print("=" * 60)