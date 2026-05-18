
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse --no-display early so plotting can be gated
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--no-display', action='store_true')
args, _ = parser.parse_known_args()
DISPLAY = not getattr(args, 'no_display', False)

# Read grayscale image
img = cv2.imread('histogram.jpg', 0)

# Check image
if img is None:
    raise FileNotFoundError("Input image 'histogram.jpg' was not found.")

# Get image dimensions
h, w = img.shape

# Use OpenCV's histogram equalization (vectorized, optimized C implementation)
# This replaces the manual histogram/CDF loops for much better performance.
output = cv2.equalizeHist(img)

# Save output image
output_path = "histogram_equalized_output.jpg"
cv2.imwrite(output_path, output)

print(f"Saved output image: {output_path}")

if DISPLAY:
    # Display images
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Histogram Equalized")
    plt.imshow(output, cmap='gray')
    plt.axis('off')

    plt.show()