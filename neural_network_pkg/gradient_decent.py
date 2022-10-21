import numpy as np
from neural_network_pkg.neuron import neural_network

class GradientDecent:

    def __init__(self, alpha, x, goal, w):
        self.alpha = alpha
        self.x = x
        self.goal = goal
        self.w = w
        self.prediction = None
        self.n = self.w.shape[0]

    def train(self, time):
        for _ in range(time):
            for i in range(self.goal.shape[1]):
                y_i = np.expand_dims(self.goal[:, i], axis=1)
                self.w -= self.alpha * self.delta(self.x[:, i], y_i)

    def delta(self, x_i, y_i, bias=0):
        self.prediction = neural_network(x_i, self.w, bias)
        return 2/self.n * np.outer(self.prediction - y_i, x_i)

    def error(self):
        return np.sum(np.square(self.prediction - self.goal)) / self.n
