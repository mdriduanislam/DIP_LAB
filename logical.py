
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
img = cv2.imread('logical.jpg')

# Check imageS
if img is None:
    raise FileNotFoundError("Input image 'logical.jpg' was not found.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a simple mask (circle in the center)
mask = np.zeros_like(gray)
h, w = gray.shape
cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)

# ---- Logical Operations ----

# 1. NOT (Bitwise NOT)
img_not = cv2.bitwise_not(gray)

# 2. AND (Bitwise AND with mask)
img_and = cv2.bitwise_and(gray, mask)

# 3. OR (Bitwise OR with mask)
img_or = cv2.bitwise_or(gray, mask)

# 4. XOR (Bitwise XOR with mask)
img_xor = cv2.bitwise_xor(gray, mask)

# ---- Save Output Images ----
cv2.imwrite("logical_NOT.jpg", img_not)
cv2.imwrite("logical_AND.jpg", img_and)
cv2.imwrite("logical_OR.jpg", img_or)
cv2.imwrite("logical_XOR.jpg", img_xor)

print("Saved: logical_NOT.jpg")
print("Saved: logical_AND.jpg")
print("Saved: logical_OR.jpg")
print("Saved: logical_XOR.jpg")

if DISPLAY:
    # ---- Display All Results ----
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original (Grayscale)")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Mask (Circle)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("NOT Operation")
    plt.imshow(img_not, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("AND Operation")
    plt.imshow(img_and, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("OR Operation")
    plt.imshow(img_or, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("XOR Operation")
    plt.imshow(img_xor, cmap='gray')
    plt.axis('off')

    plt.suptitle("Logical Operations on Image", fontsize=16)
    plt.tight_layout()
    plt.show()