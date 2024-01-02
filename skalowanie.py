import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def downscale_image(image, scale_factor):
    kernel_size = int(1 / scale_factor)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    downscaled_image = cv2.filter2D(image, -1, kernel)

    downscaled_image = downscaled_image[::kernel_size, ::kernel_size]
    return downscaled_image

def upscale_image(image, scale_factor):
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    
    upscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    return upscaled_image

image_path = 'kwiatek.webp'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

scale_factor = 0.5
downscaled_image = downscale_image(original_image, scale_factor)

upscaled_image = upscale_image(downscaled_image, 1/scale_factor)

original_image_resized = cv2.resize(original_image, upscaled_image.shape[::-1])
mse_upscale = mean_squared_error(original_image_resized.flatten(), upscaled_image.flatten())

original_image_resized = cv2.resize(original_image, downscaled_image.shape[::-1])
mse_downscale = mean_squared_error(original_image_resized.flatten(), downscaled_image.flatten())

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(downscaled_image, cmap='gray')
plt.title('Downscaled Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(upscaled_image, cmap='gray')
plt.title('Upscaled Image (Nearest Neighbor)')
plt.axis('off')

plt.show()

print(f"The MSE between the original and upscaled images is: {mse_upscale}")
print(f"The MSE between the original and downscaled images is: {mse_downscale}")
