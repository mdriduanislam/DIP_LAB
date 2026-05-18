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
img = cv2.imread('sobel.jpg')

# Check image
if img is None:
    raise FileNotFoundError("Input image 'sobel.jpg' was not found.")

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- Apply Sobel Filter ----

# Sobel in X direction (detects vertical edges)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel in Y direction (detects horizontal edges)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute values (handle negative values)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Combined Sobel (magnitude of gradient)
# Formula: sqrt(Sobel_X^2 + Sobel_Y^2)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# Save output images
cv2.imwrite("sobel_x_output.jpg", sobel_x_abs)
cv2.imwrite("sobel_y_output.jpg", sobel_y_abs)
cv2.imwrite("sobel_combined_output.jpg", sobel_combined)

print("Saved: sobel_x_output.jpg")
print("Saved: sobel_y_output.jpg")
print("Saved: sobel_combined_output.jpg")

if DISPLAY:
    # ---- Display All Results ----
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original (Grayscale)")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Sobel - X Direction (Vertical Edges)")
    plt.imshow(sobel_x_abs, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Sobel - Y Direction (Horizontal Edges)")
    plt.imshow(sobel_y_abs, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Sobel - Combined (All Edges)")
    plt.imshow(sobel_combined, cmap='gray')
    plt.axis('off')

    plt.suptitle("Sobel Edge Detection", fontsize=16)
    plt.tight_layout()
    plt.show()