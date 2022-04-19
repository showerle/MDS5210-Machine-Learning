import numpy as np
import matplotlib.pyplot as plt

p = 20
r = 3

def func(X, y, theta):
    diff = np.matmul(X, theta) - y
    return np.linalg.norm(diff, axis=0)**2

theta1 = np.linspace(-2, 4, p)
theta2 = np.linspace(-2, 4, p)

X = np.array([[1, 0]])
y = np.array([1])

Theta1, Theta2 = np.meshgrid(theta1, theta2)
Thetas = np.vstack([np.reshape(Theta1, -1), np.reshape(Theta2, -1)])
f = func(X, y, Thetas).reshape(p, p)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Theta1, Theta2, f, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
# ax.contour3D(Theta1, Theta2, f, 50, cmap='binary')
plt.show()