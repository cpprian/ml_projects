import numpy as np
from datetime import datetime
from neural_network_pkg.neuron import neural_network

class GradientDecent:

    def __init__(self, alpha, x, goal, wh=None, wy=None, batch_size=1):
        self.alpha = alpha
        self.x = x
        self.goal = goal
        self.Wh = wh
        self.Wy = wy
        self.err = 0
        self.n = self.goal.shape[0]
        self.batch_size = batch_size

        self.prediction_hidden = None
        self.prediction_output = None

        self.delta_hidden = None
        self.delta_output = None

        # for test accuracy
        self.all_prediction = np.zeros(self.goal.shape)

        self.activation_function = []

    def fit(self, time):
        for j in range(time):
            for i in range(int(self.x.shape[1] / self.batch_size)):
                batch_start, batch_end = i * self.batch_size, (i + 1) * self.batch_size
                x_i = self.x[:, batch_start:batch_end]
                y_i = self.goal[:, batch_start:batch_end]
                
                self.prediction_hidden = self.predict(self.Wh, x_i, func=self.activation_function[0])
                dropout = np.random.randint(2, size=self.prediction_hidden.shape)
                self.prediction_hidden *= dropout * 2

                self.prediction_output = self.predict(self.Wy, self.prediction_hidden, func=self.activation_function[1])

                self.delta_output = self.delta(self.prediction_output, y_i) / self.batch_size
                self.delta_hidden = self.Wy.T.dot(self.delta_output) 
                self.delta_hidden *= dropout

                self.Wh -= self.alpha * (self.delta_hidden @ x_i.T)
                self.Wy -= self.alpha * (self.delta_output @ self.prediction_hidden.T)

    def predict(self, w_i, x_i, bias=0, func=None):
        pred = neural_network(x_i, w_i, bias)  

        if func is not None:
            pred = pred * func(pred)
        return pred

    def input_prediction(self, x):
        self.prediction_hidden = self.predict(self.Wh, x, func=self.activation_function[0])
        self.prediction_output = self.predict(self.Wy, self.prediction_hidden, func=self.activation_function[1])
        return self.prediction_output

    def delta(self, x_i, y_i):
        return 2/self.n * (x_i - y_i) 

    def error(self, y_i):
        sum_prediction_goal = 0
        for i in range(y_i.shape[0]):
            sum_prediction_goal += np.square(self.prediction_output[i, 0] - y_i[i, 0])

        return sum_prediction_goal / self.n

    def find_max(self, col):
        pred = np.argmax(self.prediction_output[:, col])

        # clean up the last prediction max
        self.all_prediction[:, col] = 0
        self.all_prediction[pred, col] = 1

    def accuracy(self, f_log):
        acc = 0
        self.prediction_hidden = self.predict(self.Wh, self.x, func=self.tanh)
        self.prediction_output = self.predict(self.Wy, self.prediction_hidden, func=self.softmax)
        
        for i in range(self.goal.shape[1]):
            self.find_max(i)
            if np.array_equal(self.all_prediction[:, i], self.goal[:, i]):
                acc += 1

        with open(f_log, 'a') as f:
            f.write(f"Accuracy: {acc/self.goal.shape[1]}  {datetime.today().strftime('%D-%M-%Y %H:%M:%S')}     Acc: {acc}  Goal: {self.goal.shape[1]}\n")

        return acc / self.goal.shape[1]

    def insert_activation_function(self, func, idx):
        self.activation_function.insert(idx, func)

    def relu_deriv(self, layer):
        return np.greater(layer, 0).astype(int)

    def sigmoid(self, layer):
        return 1 / (1 + np.exp(-layer))

    def sigmoid_deriv(self, layer):
        return layer * (1 - layer)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def tanh_deriv(self, layer):
        return 1 - np.square(layer)

    def softmax(self, x):
        exp = np.exp(x)
        return exp / np.sum(exp)
