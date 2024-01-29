import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def perform_fft_image_compression(file_path, compression_lvl):
    compression_lvl = np.clip(compression_lvl, 0, 100)

    img = Image.open(file_path).convert('L')

    fft_img = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_img)

    magnitude = 20 * np.log(np.abs(fft_shifted) + 1)

    threshold = np.percentile(magnitude, compression_lvl)
    fft_shifted[magnitude < threshold] = 0

    inverse_fft_shifted = np.fft.ifftshift(fft_shifted)
    reconstructed_img = np.fft.ifft2(inverse_fft_shifted)
    reconstructed_img = np.abs(reconstructed_img)

    final_img = Image.fromarray(reconstructed_img.astype('uint8'))

    plot_images(img, final_img)

    return final_img, magnitude

def plot_images(original, compressed):
    plt.figure(figsize=(12, 6))

    plt.subplot(121), plt.imshow(original, cmap='gray')
    plt.title('Initial Image'), plt.axis('off')

    plt.subplot(122), plt.imshow(compressed, cmap='gray')
    plt.title('Image after FFT Compression'), plt.axis('off')

    plt.show()

compressed_img, freq_spectrum = perform_fft_image_compression('obraz.jpg', 90)
