import math

import matplotlib.pyplot as plt
import numpy as np

"""
I have plotted some activation functions for the KNN part of the thesis.
Nothing related to training.
"""


def relu(x):
    return max(x, 0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def leaky_relu(x):
    if x < 0:
        return 0.05 * x
    return x


def elu(x):
    if x < 0:
        return 1 * (math.exp(x) - 1)
    return x


def soft_plus(x):
    return np.log(1 + math.exp(x))


def swish(x):
    return x * (1 / (1 + math.exp(-x)))


# Relu Varianten
x = np.arange(-10, 10, 0.01)
plt.plot(x, list(map(lambda a: relu(a), x)), c='b', label="ReLU")
plt.plot(x, list(map(lambda a: leaky_relu(a), x)), '--', c='orange', label="leaky ReLU")
plt.plot(x, list(map(lambda a: elu(a), x)), '--', c='g', label="ELU")
plt.plot(x, list(map(lambda a: soft_plus(a), x)), c='r', label="SoftPlus")
plt.plot(x, list(map(lambda a: swish(a), x)), c='purple', label="Swish")

plt.xlim([-5, 5])
plt.ylim([-1, 5])

plt.legend(loc="lower right", fontsize=14)

plt.tight_layout()
plt.savefig("/home/shino/Uni/master_thesis/bin/activation_function_{0}.png".format("relu_varianten"))
plt.clf()

# Heaviside Varianten
x = np.arange(-10, 10, 0.01)
plt.plot(x, list(map(lambda a: sigmoid(a), x)), c='b', label="Sigmoid/SoftMax")

plt.xlim([-5, 5])
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig("/home/shino/Uni/master_thesis/bin/activation_function_{0}.png".format("heaviside"))
plt.clf()
