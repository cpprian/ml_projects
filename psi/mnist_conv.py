from neural_network_pkg.convolutional_neural_network import ConvolutionalNeuralNetwork
from neural_network_pkg.layer import Layer
from neural_network_pkg.util import gen_goal
import numpy as np

if __name__ == '__main__':
    n_layer_hidden = 16
    n_layer_output = 10
    alpha = 0.01

    conv = ConvolutionalNeuralNetwork()

    # load mnist data
    x = conv.load_input('psi/static/MNIST_ORG/t10k-images.idx3-ubyte')
    y = conv.load_label('psi/static/MNIST_ORG/t10k-labels.idx1-ubyte')

    wh = conv.make_filter(n_layer_hidden, weight_min_value=-0.01, weight_max_value=0.01)




