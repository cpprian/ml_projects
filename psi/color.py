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
        print(goal[0, i])
    return new_goal


if __name__ == '__main__':
    alpha = 0.01
    neurons = 4

    layer = Layer()
    layer.load_input_and_goal("psi/static/color_training.txt")
    layer.transpose_input()

    # TODO: gdzieś ucina jedno
    layer.transpose_goal()

    print(layer.Y)
    layer.Y = make_goal_to_0_1(neurons, layer.Y)

    # generate weights (neurons x X.shape[1]) 
    # layer.create_weights(neurons, layer.X.shape[0])
    # print("layer.wages:\n", layer.wages)

    # learn neural network
    # gd = GradientDecent(alpha, layer.X, layer.Y, layer.wages)
    # gd.train_until(0.0001)

    # load color_test.txt and predict the color
    # test_layer = Layer()
    # test_layer.load_input_and_goal("psi/static/color_test.txt")
    # test_layer.transpose_input()

    # TODO: check the accuracy (prediction vs. goal)
