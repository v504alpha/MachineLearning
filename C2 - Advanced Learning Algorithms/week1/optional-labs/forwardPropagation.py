import numpy as np


def Dense(a_in, w, b):
    return np.dot(a_in, w) + b


a = np.array([1, 2, 3])

w = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])

b = np.array([13, 14, 15])

k = Dense(a, w, b)

print(k)
