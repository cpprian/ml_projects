from neural_network_pkg.gradient_decent import GradientDecent
from neural_network_pkg.layer import Layer
import numpy as np

def test_relu() -> None:
    gd = GradientDecent()

    layer_1 = np.array([
        [0.05],
        [0],
        [0.15]
    ])
    expected_layer_1 = np.array([
        [1],
        [0],
        [1]
    ])

    layer_2 = np.array([
        [-0.9, 0.8],
        [0.15, 0],
        [-0.15, -1]
    ])
    expected_layer_2 = np.array([
        [0, 1],
        [1, 0],
        [0, 0]
    ])

    print(gd.relu_deriv(layer_1))

    assert (gd.relu_deriv(layer_1) == expected_layer_1).all()
    assert (gd.relu_deriv(layer_2) == expected_layer_2).all()

def zad_1() -> None:
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

    W = [Wh, Wy]
    alpha = 0.01

    gd = GradientDecent(alpha, X, Y, W)
    gd.train(1)

    print(gd.prediction)