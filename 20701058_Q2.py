import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys


def ideal_lowpass_filter(image_path, D0, show=True):
    """Apply an Ideal Lowpass Filter (ILPF) in the frequency domain.

    Args:
        image_path (str): Path to the input grayscale image.
        D0 (float): Cutoff frequency (radius) in pixels.
        show (bool): If True, display intermediate and final results.

    Returns:
        np.ndarray: Filtered image as uint8 array.
    """

    # 1) Read grayscale image using PIL and convert to NumPy array
    img_pil = Image.open(image_path).convert('L')
    img = np.array(img_pil).astype(np.float32)

    # 2) Compute 2D DFT and shift zero-frequency to center
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    # Magnitude spectrum for visualization (log scale)
    magnitude_spectrum = np.log(1 + np.abs(F_shift))

    # 3) Create ideal lowpass filter mask H(u,v)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Create distance grid
    u = np.arange(rows)[:, None]
    v = np.arange(cols)[None, :]
    D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)

    H = np.zeros_like(D, dtype=np.float32)
    H[D <= D0] = 1.0

    # 4) Apply the filter in frequency domain: G(u,v) = H(u,v) * F(u,v)
    G_shift = H * F_shift

    # Magnitude spectrum after filtering (for visualization)
    magnitude_spectrum_filtered = np.log(1 + np.abs(G_shift))

    # 5) Compute inverse DFT to obtain the filtered image
    G = np.fft.ifftshift(G_shift)
    g = np.fft.ifft2(G)
    g_real = np.real(g)

    # Normalize output to 0-255 and convert to uint8
    g_min, g_max = g_real.min(), g_real.max()
    if g_max > g_min:
        g_norm = (g_real - g_min) / (g_max - g_min) * 255.0
    else:
        g_norm = np.clip(g_real, 0, 255)

    g_uint8 = np.clip(g_norm, 0, 255).astype(np.uint8)

    if show:
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(img.astype(np.uint8), cmap='gray')
        plt.title('Original (grayscale)')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='inferno')
        plt.title('Magnitude Spectrum (log)')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(magnitude_spectrum_filtered, cmap='inferno')
        plt.title(f'Filtered Spectrum (D0={D0})')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(g_uint8, cmap='gray')
        plt.title('Filtered Image (ILPF)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return g_uint8


def parse_args_and_run():
    if len(sys.argv) < 2:
        print('Usage: python 20701058_Q2.py <image_path> [D0]')
        print('Example: python 20701058_Q2.py imageA.jpg 50')
        return

    image_path = sys.argv[1]
    D0 = float(sys.argv[2]) if len(sys.argv) >= 3 else 50.0

    print('=' * 60)
    print('Ideal Lowpass Filter (ILPF) - Frequency Domain')
    print(f'Image: {image_path}    Cutoff D0: {D0}')
    print('=' * 60)

    ideal_lowpass_filter(image_path, D0, show=True)


if __name__ == '__main__':
    parse_args_and_run()
