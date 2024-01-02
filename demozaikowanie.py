import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def apply_convolution(channel, kernel):
    return convolve2d(channel, kernel, boundary='symm', mode='same')

image = np.load('pandas.npy')

idx_x, idx_y = np.indices(image.shape[:2])
mask_R = (idx_x % 2 == 0) & (idx_y % 2 != 0)
mask_G1 = (idx_x % 2 == 0) & (idx_y % 2 == 0)
mask_G2 = (idx_x % 2 != 0) & (idx_y % 2 != 0)
mask_B = (idx_x % 2 != 0) & (idx_y % 2 == 0)

masked_image = np.zeros_like(image)
masked_image[:, :, 0] = image[:, :, 0] * mask_R
masked_image[:, :, 1] = image[:, :, 1] * (mask_G1 + mask_G2)
masked_image[:, :, 2] = image[:, :, 2] * mask_B

kernel_R = np.array([[0, 0.4, 0], [0.4, 1, 0.4], [0, 0.4, 0]])
kernel_G = np.array([[0, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 0]])
kernel_B = np.array([[0, 0.4, 0], [0.4, 1, 0.4], [0, 0.4, 0]])

R_interp = apply_convolution(masked_image[:, :, 0], kernel_R)
G_interp = apply_convolution(masked_image[:, :, 1], kernel_G)
B_interp = apply_convolution(masked_image[:, :, 2], kernel_B)

interpolated_image = np.clip(np.dstack((R_interp, G_interp, B_interp)), 0, 1)

kernel = np.array([[1/5, 1/5, 1/5], [1/5, 1.0, 1/5], [1/5, 1/5, 1/5]])

R_conv = apply_convolution(masked_image[:, :, 0], kernel)
G1_conv = apply_convolution(masked_image[:, :, 1] * mask_G1, kernel)
G2_conv = apply_convolution(masked_image[:, :, 1] * mask_G2, kernel)
G_conv = (G1_conv + G2_conv) / 2
B_conv = apply_convolution(masked_image[:, :, 2], kernel)

convoluted_image = np.clip(np.dstack((R_conv, G_conv, B_conv)), 0, 1)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(masked_image)
axs[0].set_title('Image with Bayer GRBG Mask')
axs[0].axis('off')

axs[1].imshow(interpolated_image)
axs[1].set_title('Demosaicked Image (Interpolation)')
axs[1].axis('off')

axs[2].imshow(convoluted_image)
axs[2].set_title('Demosaicked Image (2D Convolution)')
axs[2].axis('off')

plt.show()
