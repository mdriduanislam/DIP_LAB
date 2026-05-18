import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse --no-display early so plotting can be gated
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--no-display', action='store_true')
args, _ = parser.parse_known_args()
DISPLAY = not getattr(args, 'no_display', False)

# Read image
img = cv2.imread('median.png')

# Check image
if img is None:
    raise FileNotFoundError("Input image 'median.png' was not found.")

# Apply Median Filter
# (5x5 kernel size - must be odd number)
output = cv2.medianBlur(img, 5)

# Save output image
output_path = "median_filtered_output.png"
cv2.imwrite(output_path, output)
print(f"Saved output image: {output_path}")

if DISPLAY:
    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Median Filter Output")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.suptitle("Median Filter", fontsize=16)
    plt.tight_layout()
    plt.show()