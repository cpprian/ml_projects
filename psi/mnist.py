from neural_network_pkg.gradient_decent import GradientDecent
from neural_network_pkg.layer import Layer
from neural_network_pkg.util import gen_goal
import numpy as np

if __name__ == '__main__':
    # n_layer_input = 784
    n_layer_hidden = 40
    n_layer_output = 10
    alpha = 0.1
    batch_size_train = 100
    batch_size_test = 100

    train_layer = Layer()
    train_layer.load_input('psi/static/MNIST_ORG/train-images.idx3-ubyte')
    train_layer.X = train_layer.X / 255

    train_layer.load_label('psi/static/MNIST_ORG/train-labels.idx1-ubyte')
    train_layer.transpose_goal()

    train_layer.add_layer(n_layer_hidden, activation_function="relu")
    train_layer.add_layer(n_layer_output, activation_function="relu")
    train_layer.Y = gen_goal(train_layer.W[1], train_layer.Y, train_layer.amount_input)
    train_layer.Y = train_layer.Y.astype(int)

    test_layer = Layer()
    test_layer.load_input('psi/static/MNIST_ORG/t10k-images.idx3-ubyte')
    test_layer.X = test_layer.X / 255

    test_layer.load_label('psi/static/MNIST_ORG/t10k-labels.idx1-ubyte')
    test_layer.transpose_goal()

    test_layer.add_layer(n_layer_hidden, activation_function="relu")
    test_layer.add_layer(n_layer_output, activation_function="relu")

    test_layer.Y = gen_goal(test_layer.W[1], test_layer.Y, test_layer.amount_input)

    gd = GradientDecent(alpha, train_layer.X, train_layer.Y, train_layer.W[0], train_layer.W[1], batch_size_train)
    gd.insert_activation_function(gd.relu_deriv, 0)
    gd.insert_activation_function(gd.relu_deriv, 1)

    print("Start training")   