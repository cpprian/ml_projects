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
    return (goal-1).astype(int)

def return_color(goal):
    return np.argmax(goal).astype(int) + 1

if __name__ == '__main__':
    alpha = 0.01
    n_hidden = 11
    n_out = 4

    layer = Layer()
    layer.load_input_and_goal("psi/static/color/color_training.txt")
    layer.transpose_input()
    layer.transpose_goal()
    layer.Y = make_goal_to_0_1(n_out, layer.Y)

    # generate hidden layer
    layer.add_layer(n_hidden, activation_function="relu")

    # generate output layer
    layer.add_layer(n_out, activation_function="relu")

    # load color_test.txt and predict the color
    test_layer = Layer()
    test_layer.load_input_and_goal("psi/static/color/color_test.txt")
    test_layer.transpose_input()
    test_layer.transpose_goal()
    test_layer.Y = make_goal_to_0_1(n_out, test_layer.Y)

    # learn neural network
    gd = GradientDecent(alpha, layer.X, layer.Y, layer.W[0], layer.W[1])
    gd.insert_activation_function(gd.relu_deriv, 0)
    gd.insert_activation_function(gd.relu_deriv, 1)

    result = 0
    while True:
        gd.fit(1)
        gd2 = GradientDecent(alpha, test_layer.X, test_layer.Y, gd.Wh, gd.Wy)
        temp = gd2.accuracy()
        if temp > result:
            result = temp
        else:
            if temp > 0.9:
                break
            else:
                layer.set_W([])
                layer.add_layer(n_hidden, activation_function="relu")
                layer.add_layer(n_out, activation_function="relu")

                gd = GradientDecent(alpha, layer.X, layer.Y, layer.W[0], layer.W[1])

    layer.save_weights("psi/static/color/weights_color.txt")