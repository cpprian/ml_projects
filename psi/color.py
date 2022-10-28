from neural_network_pkg.gradient_decent import GradientDecent
from neural_network_pkg.layer import Layer
import numpy as np

# RED       1 0 0 0
# GREEN     0 1 0 0
# BLUE      0 0 1 0
# YELLOW    0 0 0 1

def make_goal_to_0_1(neurons, goal):
    new_goal = np.zeros((neurons, goal.shape[1]))
    for i in range(goal.shape[1]):
        idx = return_color_index(goal[0, i])
        new_goal[idx, i] = 1

    return new_goal

def return_color_index(goal):
    if goal == 1:
        return 0
    elif goal == 2:
        return 1
    elif goal == 3:
        return 2
    elif goal == 4:
        return 3
    return -1

def return_color(goal):
    if goal[0] == 1:
        return 1
    elif goal[1] == 1:
        return 2
    elif goal[2] == 1:
        return 3
    elif goal[3] == 1:
        return 4
    return -1


if __name__ == '__main__':
    alpha = 0.002
    neurons = 4

    layer = Layer()
    layer.load_input_and_goal("psi/static/color_training.txt")
    layer.transpose_input()
    layer.transpose_goal()
    layer.Y = make_goal_to_0_1(neurons, layer.Y)

    # generate weights (neurons x X.shape[1]) 
    layer.create_weights(neurons, layer.X.shape[0])

    # learn neural network
    gd = GradientDecent(alpha, layer.X, layer.Y, layer.wages)
    gd.train(1000)

    # load color_test.txt and predict the color
    test_layer = Layer()
    test_layer.load_input_and_goal("psi/static/color_test.txt")
    test_layer.transpose_input()
    test_layer.transpose_goal()
    test_layer.Y = make_goal_to_0_1(neurons, test_layer.Y)

    gd2 = GradientDecent(alpha, test_layer.X, test_layer.Y, gd.w)
    print(gd2.accuracy())
