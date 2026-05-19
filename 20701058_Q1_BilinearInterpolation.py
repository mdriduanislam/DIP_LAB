import numpy as np
import cv2

def bilinear_interpolation(img, new_h, new_w):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        resized = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    else:
        resized = np.zeros((new_h, new_w), dtype=np.uint8)

    r_scale = (h - 1) / (new_h - 1)
    c_scale = (w - 1) / (new_w - 1)

    for i in range(new_h):
        for j in range(new_w):
            y = i * r_scale
            x = j * c_scale
            y0, x0 = int(np.floor(y)), int(np.floor(x))
            y1, x1 = min(y0+1, h-1), min(x0+1, w-1)
            dy, dx = y - y0, x - x0

            if len(img.shape) == 2:
                val = (1-dy)*(1-dx)*img[y0,x0] + (1-dy)*dx*img[y0,x1] + \
                      dy*(1-dx)*img[y1,x0] + dy*dx*img[y1,x1]
                resized[i,j] = val
            else:
                for c in range(img.shape[2]):
                    val = (1-dy)*(1-dx)*img[y0,x0,c] + (1-dy)*dx*img[y0,x1,c] + \
                          dy*(1-dx)*img[y1,x0,c] + dy*dx*img[y1,x1,c]
                    resized[i,j,c] = val
    return resized

if __name__ == "__main__":
    img = cv2.imread('20701058_Q1_Input.jpg')
    if img is None:
        raise FileNotFoundError("Input image '20701058_Q1_Input.jpg' was not found.")
    out = bilinear_interpolation(img, 100, 100)
    output_path = '20701058_Q1_Output.png'
    cv2.imwrite(output_path, out)
    print(f"Done: {output_path}")