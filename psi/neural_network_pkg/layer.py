import numpy as np
import random
import copy
from neural_network_pkg.neuron import deep_neural_network

class Layer:
    wages = None
    X = None
    Y = None

    def __init__(self, x=None, y=None, w=None, file_name="") -> None:
        if file_name != "":
            self.load_weights(file_name)

        X = x
        Y = y
        wages = w

    def create_weights(self, row, col):
        self.wages = np.random.uniform(-1, 1, size=(row, col))

    def add_layer(self, n, weight_min_value=0, weight_max_value=1):
        if weight_max_value < weight_min_value:
            temp = weight_max_value
            weight_max_value = weight_min_value
            weight_min_value = temp

        for _ in range(n):
            neurons = random.randint(2, 7) # random rows for the next hidden layer or output
            w = self.wages[-1].shape[0] # last layer neurons

            new_wage = np.random.uniform(weight_min_value, 
                            weight_max_value, 
                            size=(neurons, w))
            self.add_new_wage(new_wage)


    def load_weights(self, file_name):
        wage = None
        with open(file_name, "r") as f:
            lines = f.readlines()

            for line in lines:
                row = np.fromstring(line, dtype=float, sep=",\n")

                if wage is None:
                    wage = row
                    continue

                wage = np.vstack((wage, row)) 

        self.add_new_wage(wage)

    
    def add_new_wage(self, new_wage):
        if self.wages is None:
            self.wages = [new_wage]
            return
        else:   
            self.wages.append(new_wage)


    def load_input_and_goal(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

            for line in lines:
                row = np.fromstring(line, dtype=float, sep=" \t\n")

                x = row[:3]
                y = row[3:]

                if self.X is None:
                    self.X = x
                    continue

                if self.Y is None:
                    self.Y = y
                    continue

                self.X = np.vstack((self.X, x))
                self.Y = np.vstack((self.Y, y))

    def transpose_input(self):
        self.X = self.X.T

    def transpose_goal(self):
        self.Y = self.Y.T