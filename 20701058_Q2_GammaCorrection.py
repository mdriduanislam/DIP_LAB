import cv2
import numpy as np

img = cv2.imread('20701058_Q2_Input.jpg')

if img is None:
    raise FileNotFoundError("Input image '20701058_Q2_Input.jpg' was not found.")

gamma = 0.5

gamma_table = np.array([((i / 255.0) ** gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

output = cv2.LUT(img, gamma_table)

output_ext = ".jpg"
output_path = f"20701058_Q2_Output{output_ext}"
cv2.imwrite(output_path, output)
print(f"Saved output image: {output_path}")