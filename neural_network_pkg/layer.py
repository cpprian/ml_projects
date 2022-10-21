import numpy as np
import random
import copy
from neuron import deep_neural_network

class Layer:
    wages = None

    def __init__(self, file_name) -> None:
        self.load_weights(file_name)


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


    def predict(self, input_n):
        output_n = copy.deepcopy(input_n)

        return deep_neural_network(output_n, self.wages, 0)


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
