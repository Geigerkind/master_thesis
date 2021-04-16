import math

import matplotlib.pyplot as plt
import numpy as np

"""
I plotted the learn algorithms with this file.
You can find the plot in the theses at the KNN part of the learn algorithms.
This is quite interesting to learn of optimization is actually done.
"""


def gradient_descent(max_iterations, w_init,
                     obj_func, grad_func, extra_param=[],
                     learning_rate=0.05, momentum=0.8):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    delta_w = np.zeros(w.shape)
    i = 0

    w_history = np.vstack((w_history, w))
    f_history = np.vstack((f_history, obj_func(w, extra_param)))
    while i < max_iterations:
        delta_w = -learning_rate * grad_func(w, extra_param) + momentum * delta_w
        w = w + delta_w

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # update iteration number and diff between successive values
        # of objective function
        i += 1

    return w_history, f_history


def sgd(max_iterations, w_init,
        obj_func, grad_func, extra_param=[],
        learning_rate=0.05, momentum=0.8):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    delta_w = np.zeros(w.shape)
    i = 0

    w_history = np.vstack((w_history, w))
    f_history = np.vstack((f_history, obj_func(w, extra_param)))
    while i < max_iterations:
        for j in range(len(w)):
            grad = grad_func(w, extra_param)
            delta_w[j] = -learning_rate * grad[j] + momentum * delta_w[j]
            w[j] = w[j] + delta_w[j]

            # store the history of w and f
            w_history = np.vstack((w_history, w))
            f_history = np.vstack((f_history, obj_func(w, extra_param)))

            # update iteration number and diff between successive values
            # of objective function
            i += 1

    return w_history, f_history


def adagrad(max_iterations, w_init,
            obj_func, grad_func, extra_param=[],
            learning_rate=0.05):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    w_saved = np.zeros(w.shape)
    i = 0

    w_history = np.vstack((w_history, w))
    f_history = np.vstack((f_history, obj_func(w, extra_param)))
    while i < max_iterations:
        grad = grad_func(w, extra_param)
        grad_quad = np.asarray([
            grad[0] * grad[0],
            grad[1] * grad[1],
        ])
        w_saved = w_saved + grad_quad
        w = w - grad * (learning_rate / math.sqrt(w_saved[0] ** 2 + w_saved[1] ** 2 + 0.000000001))

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # update iteration number and diff between successive values
        # of objective function
        i += 1

    return w_history, f_history


def rmsprop(max_iterations, w_init,
            obj_func, grad_func, extra_param=[],
            learning_rate=0.05, gamma=0.6):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    w_saved = np.zeros(w.shape)
    i = 0

    w_history = np.vstack((w_history, w))
    f_history = np.vstack((f_history, obj_func(w, extra_param)))
    while i < max_iterations:
        grad = grad_func(w, extra_param)
        grad_quad = np.asarray([
            grad[0] * grad[0],
            grad[1] * grad[1],
        ])
        w_saved = w_saved * gamma + grad_quad * (1 - gamma)
        w = w - grad * (learning_rate / math.sqrt(w_saved[0] ** 2 + w_saved[1] ** 2 + 0.000000001))

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # update iteration number and diff between successive values
        # of objective function
        i += 1

    return w_history, f_history


def adam(max_iterations, w_init,
         obj_func, grad_func, extra_param=[],
         learning_rate=0.05, gamma=0.6, gamma2=0.7):
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    w_saved = np.zeros(w.shape)
    v_saved = np.zeros(w.shape)
    i = 0

    w_history = np.vstack((w_history, w))
    f_history = np.vstack((f_history, obj_func(w, extra_param)))
    while i < max_iterations:
        grad = grad_func(w, extra_param)
        grad_quad = np.asarray([
            grad[0] * grad[0],
            grad[1] * grad[1],
        ])
        v_saved = (v_saved * gamma + grad * (1 - gamma)) / (1 - (gamma ** (i + 1)))
        w_saved = (w_saved * gamma2 + grad_quad * (1 - gamma2)) / (1 - (gamma2 ** (i + 1)))
        w = w - v_saved * (learning_rate / math.sqrt(w_saved[0] ** 2 + w_saved[1] ** 2 + 0.000000001))

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        # update iteration number and diff between successive values
        # of objective function
        i += 1

    return w_history, f_history


# Objective function
def f(w, extra=[]):
    return (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2


# Function to compute the gradient
def grad(w, extra=[]):
    return np.asarray([
        -400 * w[0] * (w[1] - (w[0] ** 2)) - 2 * (1 - w[0]),
        200 * (w[1] - (w[0] ** 2))
    ])


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


px = np.arange(-2, 2, 0.01).tolist()
py = np.arange(-2, 2, 0.01).tolist()

points_x = []
points_y = []
points_z = []
for i in range(len(px)):
    for j in range(len(py)):
        val = rosenbrock(px[i], py[j])
        points_z.append(val)
        points_x.append(px[i])
        points_y.append(py[j])

# Apply Learning algorithms
max_iterations = 20
learning_rate = 0.00005
gd_w, _ = gradient_descent(max_iterations, np.asarray([-2, -2]), f, grad, [], learning_rate, 0)
gd_m_w, _ = gradient_descent(max_iterations, np.asarray([-2, -2]), f, grad, [], learning_rate, 0.8)
sgd_w, _ = sgd(max_iterations, np.asarray([-2, -2]), f, grad, [], learning_rate, 0)
sgd_m_w, _ = sgd(max_iterations, np.asarray([-2, -2]), f, grad, [], learning_rate, 0.8)
ada_w, _ = adagrad(max_iterations, np.asarray([-2, -2]), f, grad, [], 5000)
rmp_w, _ = rmsprop(max_iterations, np.asarray([-2, -2]), f, grad, [], 300)
adam_w, _ = adam(max_iterations, np.asarray([-2, -2]), f, grad, [], 1000)

# Plot everything
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D([x[0] for x in gd_w], [x[1] for x in gd_w], [f(x) for x in gd_w], c="red", zorder=3, label="GD")
ax.plot3D([x[0] for x in sgd_w], [x[1] for x in sgd_w], [f(x) for x in sgd_w], c="yellow", zorder=3.1, label="SGD")
ax.plot3D([x[0] for x in ada_w], [x[1] for x in ada_w], [f(x) for x in ada_w], c="cyan", zorder=3.4, label="Adagrad")
ax.plot3D([x[0] for x in rmp_w], [x[1] for x in rmp_w], [f(x) for x in rmp_w], c="lightcoral", zorder=3.5,
          label="RMSprop")
ax.plot3D([x[0] for x in adam_w], [x[1] for x in adam_w], [f(x) for x in adam_w], "--", c="black", zorder=3.6,
          label="Adam")
ax.plot3D([x[0] for x in gd_m_w], [x[1] for x in gd_m_w], [f(x) for x in gd_m_w], "--", c="orange", zorder=3.2,
          label="GD + Momentum")
ax.plot3D([x[0] for x in sgd_m_w], [x[1] for x in sgd_m_w], [f(x) for x in sgd_m_w], "--", c="green", zorder=3.3,
          label="SGD + Momentum")
ax.scatter3D(points_x, points_y, points_z, c=points_z, cmap="terrain", s=0.5, zorder=2.6)

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

ax.legend(loc=(0.88, 0))

plt.tight_layout()
plt.savefig("/home/shino/Uni/master_thesis/bin/learn_algorithms.png")
plt.clf()
