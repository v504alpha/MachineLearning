import numpy as np
from public_tests import *
from utils import *


class LogisticRegressor:
    cost_ = []
    w, b = 0, 0
    nIter, alpha = 0, 0

    def __init__(self, path, w, b, nIter=1000, alpha=1e-3, lambda_=1.00):
        self.x, self.y = load_data(path)
        self.m, self.n = self.x.shape
        self.w, self.b = w, b
        self.nIter, self.alpha = nIter, alpha
        self.lambda_ = lambda_

    def map_features(self, degree):
        """Using Feature engineering to convert input data to useful features"""
        np.random.seed(1)
        x1 = np.atleast_1d(self.x[:, 0])
        x2 = np.atleast_1d(self.x[:, 1])
        out = []
        for i in range(1, degree + 1):
            for j in range(i + 1):
                out.append(x1 ** (i - j) * x2**j)
        self.x = np.stack(out, axis=1)
        self.w = np.random.rand(self.x.shape[1]) - 0.5
        self.m, self.n = self.x.shape

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def f(self):
        return self.sigmoid(np.dot(self.x, self.w) + self.b)

    def calcCost(self):
        return np.sum(
            (-self.lambda_ / self.m)
            * (
                np.dot(self.y, np.log(self.f()))
                + np.dot((1 - self.y), (np.log(1 - self.f())))
            )
            + (self.lambda_ / (2 * self.m)) * np.sum(np.square(self.w))
        )

    def calcGradient(self):
        dj_dw = (1 / self.m) * np.dot(self.f() - self.y, self.x)
        dj_db = (1 / self.m) * np.sum(self.f() - self.y)
        return dj_db, dj_dw + (self.lambda_ / self.m) * self.w

    def gradientDescent(self):
        cost_ = []
        for i in range(self.nIter):
            dj_db, dj_dw = self.calcGradient()
            self.w, self.b = (
                self.w - self.alpha * dj_dw,
                self.b - self.alpha * dj_db,
            )
            cost = self.calcCost()
            if (i % 100 == 0) or (i == self.nIter - 1):
                cost_.append([i, self.w, self.b, cost])

        self.cost_ = cost_

        return cost_

    def predict_(self):
        return np.where(self.f() > 0.5, 1, 0)

    def accuracy_(self):
        return f"Accuracy: {np.mean(self.predict_() == self.y) * 100}"

    def __str__(self):
        res = ""
        for i in self.cost_:
            res += f"Iteration: {i[0]} w: {i[1]} b: {i[2]} cost: {i[3]} \n"
        return res


# Logical Regression objects

# Example 1 Logistic Regression

example_obj = LogisticRegressor(
    "data/ex2data1.txt", w=np.array([-0.00082978, 0.00220324]), b=-8, nIter=10001
)
example_obj.gradientDescent()
print(example_obj)
print(example_obj.accuracy_())

# Example 2 Logical Regressor ( Feature Engineering )

example1_obj = LogisticRegressor(
    "data/ex2data2.txt", w=np.array([]), b=1, nIter=10001, lambda_=0.01, alpha=0.01
)
example1_obj.map_features(6)
example1_obj.gradientDescent()
print(example1_obj)
print(example1_obj.accuracy_())
