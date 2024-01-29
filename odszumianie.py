from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

def denoise_image(image_fp):
    img = Image.open(image_fp).convert('L')

    img_array = np.array(img)

    noise = np.random.normal(0, 15, img_array.shape)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype('uint8')
    noisy_img = Image.fromarray(noisy_img_array)

    fft_img = fftn(noisy_img_array)
    fft_img_shifted = fftshift(fft_img)

    rows, cols = fft_img_shifted.shape
    center_row, center_col = rows // 2, cols // 2
    fraction = 0.1
    mask = np.zeros((rows, cols), np.uint8)
    mask[center_row-int(rows*fraction):center_row+int(rows*fraction),
         center_col-int(cols*fraction):center_col+int(cols*fraction)] = 1
    
    fft_img_shifted *= mask
    ifft_img_shifted = fftshift(fft_img_shifted)
    restored_img_array = ifftn(ifft_img_shifted)
    restored_img_array = np.abs(restored_img_array)
    restored_img_array = (restored_img_array / np.max(restored_img_array) * 255).astype('uint8')
    restored_img = Image.fromarray(restored_img_array)

    plot_images(img, noisy_img, restored_img)

def plot_images(original, noisy, denoised):
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Grayscale'), plt.axis('off')

    plt.subplot(132)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Grayscale'), plt.axis('off')

    plt.subplot(133)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Grayscale'), plt.axis('of')

    plt.show()

denoise_image('obraz.jpg')
