from keras.optimizers.optimizer_experimental.adam import optimizer
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def load_coffee_data():
    """Creates a coffee roasting data set.
    roasting duration: 12-15 minutes is best
    temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if t > 175 and t < 260 and d > 12 and d < 15 and d <= y:
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


x, y = load_coffee_data()

tf.random.set_seed(1234)
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(x)
x = norm_l(x)
print(x.shape, y.shape)


x = np.tile(x, (1000, 1))
y = np.tile(y, (1000, 1))
print(x.shape, y.shape)


model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(units=3, activation="sigmoid", name="layer1"),
        Dense(units=1, activation="sigmoid", name="layer2"),
    ]
)

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    x,
    y,
    epochs=10,
)

w, b = model.get_layer("layer2").get_weights()
print(w, b)

predictions = model.predict(
    norm_l(np.array([[200, 13.9], [200, 17]]))  # postive example
)
print(predictions)
pre = np.zeros_like(predictions)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for i in range(len(predictions)):
    if sigmoid(predictions[i]) >= 0.5:
        pre[i] = 1
    else:
        pre[i] = 0
print(pre)
