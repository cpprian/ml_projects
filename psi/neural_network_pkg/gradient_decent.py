import numpy as np
from neural_network_pkg.neuron import neural_network

class GradientDecent:

    def __init__(self, alpha, x, goal, w):
        self.alpha = alpha
        self.x = x
        self.goal = goal
        self.w = w
        self.prediction = None
        self.n = self.goal.shape[0]
        self.err = 0

        self.all_prediction = np.zeros(self.goal.shape)

    def train(self, time):
        for j in range(time):
            err_era = 0
            for i in range(self.goal.shape[1]):
                y_i = np.expand_dims(self.goal[:, i], axis=1)
                
                d = self.delta(self.x[:, i], y_i)
                err_era += self.error(y_i)
                self.w -= self.alpha * d
                self.find_max(i)

            self.err = err_era
            print("numer: ", j)

    def predict(self, x_i, bias=0):
        self.prediction = neural_network(x_i, self.w, bias)  

    def find_max(self, col):
        pred = np.argmax(self.prediction)

        # clean up the last prediction max
        self.all_prediction[:, col] = 0
        self.all_prediction[pred, col] = 1

    def delta(self, x_i, y_i, bias=0):
        self.predict(x_i, bias)
        return 2/self.n * np.outer(self.prediction - y_i, x_i)

    def error(self, y_i):
        sum_prediction_goal = 0
        for i in range(y_i.shape[0]):
            sum_prediction_goal += np.square(self.prediction[i, 0] - y_i[i, 0])

        return sum_prediction_goal / self.n


    def test_prediction(self):
        for i in range(self.goal.shape[1]):
            self.predict(self.x[:, i])
            self.find_max(i)

    def accuracy(self):
        acc = 0

        self.test_prediction()
        for i in range(self.goal.shape[1]):
            if np.array_equal(self.all_prediction[:, i], self.goal[:, i]):
                acc += 1

        print(acc)
        print(self.goal.shape[1])

        return acc / self.goal.shape[1]
