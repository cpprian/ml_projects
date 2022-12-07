import numpy as np
from neural_network_pkg.convolutional_neural_network import ConvolutionalNeuralNetwork
from neural_network_pkg.util import gen_goal

if __name__ == '__main__':
    iteration = 50
    alpha = 0.01
    batch_size = 100
    n_layer_hidden = 16
    n_layer_output = 10
    cnn = ConvolutionalNeuralNetwork()
    cnn2 = ConvolutionalNeuralNetwork()

    # ================== Load data ==================
    print("Loading data...")

    input_images_train = cnn.load_input('psi/static/MNIST_ORG/train-images.idx3-ubyte', 1000)
    input_labels_train = cnn.load_label('psi/static/MNIST_ORG/train-labels.idx1-ubyte', 1000)

    kernel = cnn.make_filter(9, n_layer_hidden, weight_min_value=-0.01, weight_max_value=0.01)
    wy = cnn.make_filter(26 * 26 * n_layer_hidden, n_layer_output, weight_min_value=-0.1, weight_max_value=0.1)

    input_images_test = cnn.load_input('psi/static/MNIST_ORG/t10k-images.idx3-ubyte', 500)
    input_labels_test = cnn.load_label('psi/static/MNIST_ORG/t10k-labels.idx1-ubyte', 500)

    # ================== Train ==================
    print("Start training")
    res = 0
    for i in range(iteration):
        print(f"Training {i+1}/{iteration}")
        cnn.train(alpha, iteration, batch_size, input_images_train, input_labels_train, kernel, wy)

        temp = cnn2.accuracy('psi/static/MNIST_ORG/result2.txt', input_labels_test, input_images_test, kernel, wy)

        if temp > res:
            res = temp
        else:
            if temp > 0.9:
                print("Training completed")
                break
            else:
                kernel = cnn.make_filter(9, n_layer_hidden, weight_min_value=-0.01, weight_max_value=0.01)
                wy = cnn.make_filter(26 * 26 * n_layer_hidden, n_layer_output, weight_min_value=-0.1, weight_max_value=0.1)
    