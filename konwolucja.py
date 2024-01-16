import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import as_strided

def apply_convolution(image, kernel):
    if len(image.shape) == 2:
        return convolve2d(image, kernel, boundary='symm', mode='same')
    elif len(image.shape) == 3:
        return np.stack([convolve2d(image[:, :, channel], kernel, boundary='symm', mode='same') for channel in
                         range(image.shape[2])], axis=-1)
    else:
        raise ValueError("Invalid image shape: expected 2D or 3D array")

image_path = 'mond.npy'
image = np.load(image_path)
if image.shape[-1] == 4:
    image = image[..., :3]

def convert_to_grayscale(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

gray_image = convert_to_grayscale(image)

def max_pool_2d(image, pool_size, stride):
    output_shape = ((image.shape[0] - pool_size[0]) // stride + 1, (image.shape[1] - pool_size[1]) // stride + 1)
    return np.max(as_strided(image, shape=output_shape + pool_size, strides=image.strides + image.strides), axis=(2, 3))

laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
blur_kernel = np.array([[1, 1, 1], [2, 4, 2], [1, 1, 1]]) / 16
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

laplace_image = apply_convolution(image, laplace_kernel)
blur_image = apply_convolution(image, blur_kernel)
sharpen_image = apply_convolution(max_pool_2d(gray_image, (2, 2), 1), sharpen_kernel)
sharpen_image = np.clip(sharpen_image, 0, 1)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(laplace_image)
axs[1].set_title('Laplace Filter Applied')
axs[1].axis('off')

axs[2].imshow(blur_image)
axs[2].set_title('Blur Filter Applied')
axs[2].axis('off')

axs[3].imshow(sharpen_image, cmap='gray')
axs[3].set_title('Sharpened Grayscale Image')
axs[3].axis('off')

plt.show()