from neural_network_pkg.gradient_decent import GradientDecent
from neural_network_pkg.layer import Layer
from neural_network_pkg.util import gen_goal
import numpy as np

if __name__ == '__main__':
    # n_layer_input = 784
    n_layer_hidden = 40
    n_layer_output = 10
    alpha = 0.01
    batch_size_train = 20
    batch_size_test = 20

    train_layer = Layer()
    train_layer.load_input('psi/static/MNIST_ORG/train-images.idx3-ubyte')
    train_layer.X = train_layer.X / 255

    train_layer.load_label('psi/static/MNIST_ORG/train-labels.idx1-ubyte')
    train_layer.transpose_goal()

    train_layer.add_layer(n_layer_hidden, weight_min_value=-0.01, weight_max_value=0.01)
    train_layer.add_layer(n_layer_output, weight_min_value=-0.1, weight_max_value=0.1)
    train_layer.Y = gen_goal(n_layer_output, train_layer.Y, train_layer.X.shape[1])
    train_layer.Y = train_layer.Y.astype(int)

    test_layer = Layer()
    test_layer.load_input('psi/static/MNIST_ORG/t10k-images.idx3-ubyte')
    test_layer.X = test_layer.X / 255

    test_layer.load_label('psi/static/MNIST_ORG/t10k-labels.idx1-ubyte')
    test_layer.transpose_goal()

    test_layer.Y = gen_goal(n_layer_output, test_layer.Y, test_layer.X.shape[1])

    gd = GradientDecent(alpha, train_layer.X, train_layer.Y, train_layer.W[0], train_layer.W[1], batch_size_train)

    print("Start testing")
    result = 0
    while True:  
        gd.fit(100)

        gd2 = GradientDecent(alpha, test_layer.X, test_layer.Y, train_layer.W[0], train_layer.W[1], batch_size_test)
        temp = gd2.accuracy("psi/static/MNIST_ORG/result.txt")
        print(f"Accuracy: {temp}")
        if temp > result:
            result = temp
        else:
            if temp > 0.9:
                break
            else:
                train_layer.W = []
                train_layer.add_layer(n_layer_hidden, weight_min_value=-0.01, weight_max_value=0.01)
                train_layer.add_layer(n_layer_output, weight_min_value=-0.1, weight_max_value=0.1)

                gd = GradientDecent(alpha, train_layer.X, train_layer.Y, train_layer.W[0], train_layer.W[1], batch_size_train)


