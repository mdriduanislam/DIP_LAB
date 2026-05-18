import cv2
import numpy as np

img = cv2.imread('20701058_Q3_Input.jpg')

if img is None:
    raise FileNotFoundError("Input image '20701058_Q3_Input.jpg' was not found.")

min_val = np.min(img)
max_val = np.max(img)

output = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

output_ext = ".jpg"
output_path = f"20701058_Q3_Output{output_ext}"
cv2.imwrite(output_path, output)

print(f"Saved output image: {output_path}")