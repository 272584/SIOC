import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def sin_function(x):
    return np.sin(x)

sampling_frequency = 100

x = np.linspace(-2 * np.pi, 2 * np.pi, 10 * sampling_frequency)
y = sin_function(x)

kernel_size = int(1 / ((2*np.pi)/100))
kernel = np.ones(kernel_size) / kernel_size
h1_kernel = np.ones(kernel_size)

y_convolved = convolve(y, kernel, mode='same')

y_convolved = convolve(y, h1_kernel, mode='same') / kernel_size

mse = mean_squared_error(y, y_convolved)

print(mse)

plt.figure(figsize=(14, 7))
plt.plot(x, y, label='Original sin(x)')
plt.plot(x, y_convolved, label='Convolved sin(x) with h1(x)', linestyle='dashed')
plt.legend()
plt.title('Convolution of sin(x) with kernel h1(x)')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()