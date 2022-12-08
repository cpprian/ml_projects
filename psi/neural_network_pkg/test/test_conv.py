import numpy as np
from neural_network_pkg.convolutional_neural_network import ConvolutionalNeuralNetwork

def test_example() -> None:
    alpha = 0.01
    iteration = 1
    cnn = ConvolutionalNeuralNetwork()

    input_image = np.array([
        [8.5, 0.65, 1.2],
        [9.5, 0.8, 1.3],
        [9.9, 0.8, 0.5],
        [9.0, 0.9, 1.0]
    ])

    expected_output = np.array([[0], [1]])

    kernel_1_weights = np.array([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1])
    kernel_2_weigths = np.array([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1])
    filter_conv = np.array([kernel_1_weights, kernel_2_weigths])

    wy = np.array([
        [0.1, -0.2, 0.1, 0.3],
        [0.2, 0.1, 0.5, -0.3]
    ])

    expected_kernel_layer = np.array([
        [3.185, 11.995],
        [3.27, 12.03]
    ])
    kernel_layer = cnn.convolve(input_image, filter_conv, filter_conv.shape[0], filter_conv.shape[1])
    assert np.allclose(kernel_layer, expected_kernel_layer)

    wy, kernel_layer = cnn.train(alpha, iteration, input_image, expected_output, filter_conv, wy)
    assert np.allclose(wy, np.array([
        [0.0409023, -0.422567, 0.0393252, 0.0767834],
        [0.236229, 0.236443, 0.537196, -0.163159]
    ]))

    print(kernel_layer)
    assert np.allclose(kernel_layer, np.array([
        [0.13997, 0.2419, -0.0614, -0.0967, 0.1034, 0.9038, 0.1055, 0.4025, 0.104],
        [0.2559, 1.0571, -0.3328, 0.096, 0.1967, -0.0042, -0.0059, 1.3018, 0.0934]
    ]), atol=0.0001)