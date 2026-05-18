import numpy as np
import cv2

img = cv2.imread('20701058_Q1_Input.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Input image '20701058_Q1_Input.jpg' was not found.")

scale = 0.5

h, w = img.shape
new_h = int(h * scale)
new_w = int(w * scale)

# Use OpenCV's built-in bilinear interpolation for efficiency
output = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

output_ext = '.png'
output_path = f'20701058_Q1_Output{output_ext}'
cv2.imwrite(output_path, output)
print(f"Saved output image: {output_path}")