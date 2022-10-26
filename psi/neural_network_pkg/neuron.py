import numpy as np


def neuron(input_n, wage, bias=0):
    return wage.dot(input_n) + bias


def neural_network(input_n, weights, bias=0):
    if len(input_n.shape) == 1:
        col_n = 1
    else:
        col_n = input_n.shape[1]
    output_n = np.ndarray(shape=(0, col_n), dtype=float)

    for i in range(weights.shape[0]):
        output_n = np.insert(output_n, i, neuron(input_n, weights[i], bias), axis=0)

    return output_n


def deep_neural_network(input_n, layers, bias):
    for layer in layers:
        input_n = neural_network(input_n, layer, bias)

    return input_n
