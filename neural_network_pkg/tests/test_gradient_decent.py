import numpy as np
from neural_network_pkg.gradient_decent import GradientDecent

def test_one_neuron_prediction() -> None:
    alpha = 0.1
    times = np.array([5, 15])
    x = np.array([2], dtype=float, ndmin=2)
    goal = np.array([0.8], dtype=float, ndmin=2)
    w = np.array([0.5], dtype=float, ndmin=2)

    gd = GradientDecent(alpha, x, goal, w)
    gd.train(times[0])
    assert gd.prediction == 0.80032
    assert np.round(gd.error(), 10) == 0.0000001024


def test_matrix_neuron_prediction() -> None:
    alpha = 0.01
    times = 1000
    x = np.array([
        [0.5, 0.1, 0.2, 0.8], 
        [0.75, 0.3, 0.1, 0.9],
        [0.1, 0.7, 0.6, 0.2]], dtype=float, ndmin=2)
        
    y = np.array([
        [0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2],
        [0.0, 0.3, 0.9, -0.1],
        [-0.1, 0.7, 0.1, 0.8]
    ], dtype=float, ndmin=2)

    w = np.array([
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]], dtype=float, ndmin=2)

    gd = GradientDecent(alpha, x, y, w)
    gd.train(times)
    assert gd.error() == 0.258218

