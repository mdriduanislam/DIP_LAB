import cv2
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import argparse

# Parse --no-display early so plotting can be gated
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--no-display', action='store_true')
args, _ = parser.parse_known_args()
DISPLAY = not getattr(args, 'no_display', False)

# Read image
img = cv2.imread('average.jpg')

# Check image
if img is None:
    raise FileNotFoundError("Input image 'average.jpg' was not found.")

# Apply Averaging Filter
# (5x5 kernel)
output = cv2.blur(img, (5, 5))

# Save output image
output_path = "average_filtered_output.jpg"
cv2.imwrite(output_path, output)

print(f"Saved output image: {output_path}")

if DISPLAY:
    # Display images
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Averaging Filter Output")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()