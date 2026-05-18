import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse --no-display early so plotting can be gated
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--no-display', action='store_true')
args, _ = parser.parse_known_args()
DISPLAY = not getattr(args, 'no_display', False)

img = cv2.imread('20701058_Q3_Input.jpg')

if img is None:
    raise FileNotFoundError("Input image '20701058_Q3_Input.jpg' was not found.")

# Check original pixel range
print(f"Original Min pixel value: {np.min(img)}")
print(f"Original Max pixel value: {np.max(img)}")

# Simulate a low contrast image (clip to 80-180 range)
img_low_contrast = np.clip(img, 80, 180).astype(np.uint8)

print(f"Low Contrast Min pixel value: {np.min(img_low_contrast)}")
print(f"Low Contrast Max pixel value: {np.max(img_low_contrast)}")

# Apply contrast stretching
min_val = np.min(img_low_contrast)
max_val = np.max(img_low_contrast)

output = ((img_low_contrast - min_val) / (max_val - min_val) * 255).astype(np.uint8)

output_path = "20701058_Q3_Output.jpg"
cv2.imwrite(output_path, output)

print(f"Saved output image: {output_path}")

if DISPLAY:
    # Display images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Low Contrast Image (Clipped 80-180)")
    plt.imshow(cv2.cvtColor(img_low_contrast, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Output Image (Contrast Stretched)")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.suptitle("Min-Max Contrast Stretching", fontsize=16)
    plt.tight_layout()
    plt.show()