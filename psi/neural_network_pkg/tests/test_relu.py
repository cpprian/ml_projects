from neural_network_pkg.gradient_decent import GradientDecent
from neural_network_pkg.layer import Layer
import numpy as np

def test_zad_1() -> None:
    alpha = 0.01
    X = np.array([
        [0.5, 0.1, 0.2, 0.8],
        [0.75, 0.3, 0.1, 0.9],
        [0.1, 0.7, 0.6, 0.2]
    ], dtype=float, ndmin=2)

    Y = np.array([
        [0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2]
    ])

    Wh = np.array([
        [0.1, 0.1, -0.3], 
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ])

    Wy = np.array([
        [0.7, 0.9, -0.4, 0.8, 0.1],
        [0.8, 0.5, 0.3, 0.1, 0.0],
        [-0.3, 0.9, 0.3, 0.1, -0.2]
    ])

    gd = GradientDecent(alpha, X, Y, Wh, Wy)
    gd.insert_activation_function(gd.relu_deriv, 0)
    gd.insert_activation_function(gd.relu_deriv, 1)
    gd.fit(1)

    tested_matrix = np.array([
        [0.376, 0.082, 0.053, 0.49],
        [0.3765, 0.133, 0.067, 0.465],
        [0.305, 0.123, 0.073, 0.402]
    ])
    # TODO: BEZ AKTUALIZACJI WAG
    print(gd.all_prediction_output)
    # assert np.allclose(gd.all_prediction_output, tested_matrix, atol=0.001)


def test_zad_2() -> None:
    alpha = 0.01
    X = np.array([
        [0.5, 0.1, 0.2, 0.8],
        [0.75, 0.3, 0.1, 0.9],
        [0.1, 0.7, 0.6, 0.2]
    ], dtype=float, ndmin=2)

    Y = np.array([
        [0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2]
    ])

    Wh = np.array([
        [0.1, 0.1, -0.3], 
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ])

    Wy = np.array([
        [0.7, 0.9, -0.4, 0.8, 0.1],
        [0.8, 0.5, 0.3, 0.1, 0.0],
        [-0.3, 0.9, 0.3, 0.1, -0.2]
    ])

    gd = GradientDecent(alpha, X, Y, Wh, Wy)
    gd.insert_activation_function(gd.relu_deriv, 0)
    gd.insert_activation_function(gd.relu_deriv, 1)
    gd.fit(1)

    tested_matrix = np.array([
        [0.376, 0.0807193, 0.053154, 0.490493],
        [0.3765, 0.134082, 0.0670564, 0.470195],
        [0.305, 0.122503, 0.07174, 0.398576]
    ])
    print(gd.all_prediction_output)
    assert np.allclose(gd.all_prediction_output, tested_matrix, atol=0.00001)

    gd.fit(49)
    tested_matrix = np.array([
        [0.442756, 0.140418, 0.105594, 0.584537],
        [0.565494, 0.186105, 0.0936588, 0.711427],
        [0.15731, 0.0638947, 0.0450485, 0.216373]
    ])
    print(gd.all_prediction_output)
    assert np.allclose(gd.all_prediction_output, tested_matrix, atol=0.0000001)

def test_zad_3() -> None:
    n_layer_input = 784
    n_layer_hidden = 40
    n_layer_output = 10


    # read bytes from file
    with open('psi/static/MNIST_ORG/train-images.idx3-ubyte', 'rb') as f:
        image = f.read()

        # convert bytes to a numpy array
        image = np.frombuffer(image, dtype=np.uint8, offset=16) 
        image = image.reshape(60000, 784)

    with open('psi/static/MNIST_ORG/train-labels.idx1-ubyte', 'rb') as f:
        label = f.read()

        # convert bytes to a numpy array
        label = np.frombuffer(label, dtype=np.uint8, offset=8) 
        label = label.reshape(60000, 1)

    layer_label = Layer()
    layer_label.set_W([label])
    layer_label.save_weights('psi/static/MNIST_ORG/label_weights.txt')

    layer_input = Layer()
    layer_input.set_W([image])
    layer_input.save_weights('psi/static/MNIST_ORG/input_weights.txt')
