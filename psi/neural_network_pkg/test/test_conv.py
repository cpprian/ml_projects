import numpy as np
from neural_network_pkg.convolutional_neural_network import ConvolutionalNeuralNetwork

def test_example_1() -> None:
    input_n = np.array([
        [8.5, 0.65, 1.2],
        [9.5, 0.8, 1.3],
        [9.9, 0.8, 0.5],
        [9.0, 0.9, 1.0]], dtype=np.float32)

    goal = np.array([[0], [1]], dtype=np.float32)

    alpha = 0.01

    w_conv_1 = np.array([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1], dtype=np.float32)
    w_conv_2 = np.array([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1], dtype=np.float32)
    w_conv = np.array([w_conv_1, w_conv_2])

    wy = np.array([
        [0.1, -0.2, 0.1, 0.3],
        [0.2, 0.1, 0.5, -0.3]], dtype=np.float32)

    conv = ConvolutionalNeuralNetwork()

    expected_image_sections = np.array([
        [8.5, 9.5, 9.9, 0.65, 0.8, 0.8, 1.2, 1.3, 0.5],
        [9.5, 9.9, 9.0, 0.8, 0.8, 0.9, 1.3, 0.5, 1.0]], dtype=np.float32)
    image_sections = conv.make_image_sections(input_n, w_conv.shape)
    assert np.array_equal(image_sections, expected_image_sections)

    expected_kernel_layer = np.array([
        [3.185, 11.995],
        [3.27, 12.03]], dtype=np.float32)
    kernel_layer = image_sections @ w_conv.T
    kernel_layer = np.round(kernel_layer, 3)
    assert np.array_equal(kernel_layer, expected_kernel_layer)

    expected_wy = np.array([
        [0.0409023, -0.422567, 0.0393252, 0.0767834],
        [0.236229, 0.236443, 0.537196, -0.163159]], dtype=np.float32)

    expected_kernel_layer = np.array([
        [0.13997, 0.2419, -0.0614, -0.0967, 0.1034, 0.9038, 0.1055, 0.4025, 0.104],
        [0.2559, 1.0571, -0.3328, 0.096, 0.1967, -0.0042, -0.0059, 1.3018, 0.0934]], dtype=np.float32)

    wy, kernel_layer = conv.train(alpha, image_sections, wy, kernel_layer, goal, w_conv)

    assert np.allclose(wy, expected_wy)
    assert np.allclose(kernel_layer, expected_kernel_layer, atol=0.0001)

def test_zad01() -> None:
    input_image = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])

    filter_conv = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    output_image = np.array([
        [4, 3, 4],
        [2, 4, 3],
        [2, 3, 4]
    ])

    conv = ConvolutionalNeuralNetwork()
    assert np.array_equal(conv.convolve(input_image, filter_conv), output_image)