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

    def train(self, time):
        for _ in range(time):
            err_era = 0
            for i in range(self.goal.shape[1]):
                y_i = np.expand_dims(self.goal[:, i], axis=1)
                
                d = self.delta(self.x[:, i], y_i)
                err_era += self.error(y_i)
                self.w -= self.alpha * d
            self.err = err_era

    def train_until(self, per):
        while True:
            self.train(1)

            # if accuracy is not good enough, train again
            if round(self.err, 5) < per:
                break

    def predict(self, x_i, bias=0):
        self.prediction = neural_network(x_i, self.w, bias)

        # find largest element from prediction and set to 1 and others to 0
        pred = np.argmax(self.prediction)
        self.prediction = np.zeros(self.prediction.shape)
        self.prediction[pred, 0] = 1    

        # TODO: store result to all_preiction to compare with goal later

    def delta(self, x_i, y_i, bias=0):
        self.predict(x_i, bias)
        return 2/self.n * np.outer(self.prediction - y_i, x_i)

    def error(self, y_i):
        sum_prediction_goal = 0
        for i in range(y_i.shape[0]):
            sum_prediction_goal += np.square(self.prediction[i, 0] - y_i[i, 0])

        return sum_prediction_goal / self.n
