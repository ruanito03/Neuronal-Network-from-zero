import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Para estabilidad numérica
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def inv_sigmoid(a):
    return np.log(a / (1 - a))


def sigmoid_p(z):
    s = sigmoid(z)
    return s * (1 - s)
