import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
     
def f1(x):
    return np.sin(x)
N = 100
x = np.linspace(-np.pi, np.pi, N)
y = f1(x)

N_interp = 4 * N
x_new = np.linspace(-np.pi, np.pi, N_interp)

linear_interpolator = interp1d(x, y, kind='linear')
y_interpolated_linear = linear_interpolator(x_new)

cubic_interpolator = interp1d(x, y, kind='cubic')
y_interpolated_cubic = cubic_interpolator(x_new)

mse_linear = np.mean((f1(x_new) - y_interpolated_linear) ** 2)
mse_cubic = np.mean((f1(x_new) - y_interpolated_cubic) ** 2)

plt.figure(figsize=(14, 7))
plt.grid(True)
plt.plot(x, y, 'o', label='Oryginalne punkty')
plt.plot(x_new, y_interpolated_linear, '-', label='Interpolacja liniowa')
plt.legend()

plt.show()

print(mse_linear,mse_cubic)
