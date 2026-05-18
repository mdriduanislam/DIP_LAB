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
img = cv2.imread('arithmatic.jpg')

# Check image
if img is None:
    raise FileNotFoundError("Input image 'arithmatic.jpg' was not found.")

# Arithmetic Operations

# Addition (Brighten image)
add_img = cv2.add(img, 50)

# Subtraction (Darken image)
sub_img = cv2.subtract(img, 50)

# Multiplication (Increase intensity)
mul_img = cv2.multiply(img, 1.5)

# Division (Reduce intensity)
div_img = cv2.divide(img, 2)

if DISPLAY:
    # Display images
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title("Addition")
    plt.imshow(cv2.cvtColor(add_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.title("Subtraction")
    plt.imshow(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.title("Multiplication")
    plt.imshow(cv2.cvtColor(mul_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.title("Division")
    plt.imshow(cv2.cvtColor(div_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Save outputs
cv2.imwrite("addition_output.jpg", add_img)
cv2.imwrite("subtraction_output.jpg", sub_img)
cv2.imwrite("multiplication_output.jpg", mul_img)
cv2.imwrite("division_output.jpg", div_img)

print("All output images saved successfully.")