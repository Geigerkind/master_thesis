import matplotlib.pyplot as plt
import numpy as np


def rosenbrock(x, y):
    return x ** 2 + 100 * (y - x ** 2) ** 2


px = np.arange(-2, 2, 0.01).tolist()
py = np.arange(-2, 2, 0.01).tolist()

points_x = []
points_y = []
points_z = []
together = []
for i in range(len(px)):
    for j in range(len(py)):
        val = rosenbrock(px[i], py[j])
        points_z.append(val)
        points_x.append(px[i])
        points_y.append(py[j])
        together.append([px[i], py[j], val])

fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(points_x, points_y, points_z)
plt.imshow(together, origin='lower', cmap='terrain')

#ax.set_xlim([-2, 2])
#ax.set_ylim([-2, 2])

plt.tight_layout()
plt.savefig("/home/shino/Uni/master_thesis/bin/learn_algorithms.png")
plt.clf()
