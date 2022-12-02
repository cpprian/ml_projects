import numpy as np
from neural_network_pkg.convolutional_neural_network import ConvolutionalNeuralNetwork
from neural_network_pkg.layer import Layer
from neural_network_pkg.util import gen_goal

if __name__ == '__main__':
    iteration = 50
    alpha = 0.01
    n_layer_hidden = 16
    n_layer_output = 10
    cnn = ConvolutionalNeuralNetwork()

    # ================== Load data ==================
    print("Loading data...")
    train_layer = Layer()
    train_layer.X = cnn.load_input('psi/static/MNIST_ORG/train-images.idx3-ubyte', 1000)
    train_layer.X = train_layer.X / 255

    train_layer.load_label('psi/static/MNIST_ORG/train-labels.idx1-ubyte', 1000)
    train_layer.transpose_goal()

    train_layer.add_new_wage(cnn.make_filter(n_layer_hidden, -0.01, 0.01))
    train_layer.add_new_wage(cnn.make_filter(n_layer_output, -0.1, 0.1))

    train_layer.Y = gen_goal(n_layer_output, train_layer.Y, train_layer.X.shape[1])

    image_sections = cnn.convolve(train_layer.X, train_layer.W[0]) 
    print(f"\n\n image sections shape: {image_sections.shape} train_layer.W[0].T shape: {train_layer.W[0].T.shape}")
    kernel_layer = image_sections @ train_layer.W[0].T

    print("50%\ done...")
    # test_layer = Layer()
    # test_layer.load_input('psi/static/MNIST_ORG/t10k-images.idx3-ubyte', 10000)
    # test_layer.X = test_layer.X / 255

    # test_layer.load_label('psi/static/MNIST_ORG/t10k-labels.idx1-ubyte', 10000)
    # test_layer.transpose_goal()

    # test_layer.Y = gen_goal(n_layer_output, test_layer.Y, test_layer.X.shape[1])

    # # ================== Train ==================
    # print("Training...")
    # for i in range(iteration):
    #     print(f"Start training {i+1}th")
    #     cnn.train(alpha, )
    #     cnn.accuracy(test_layer, "psi/static/MNIST_ORG/result2.txt")