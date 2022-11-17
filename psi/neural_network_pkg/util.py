import numpy as np

def gen_goal(W, Y, amount_input):
    updated_goal = np.zeros((W.shape[0], amount_input))

    for i in range(amount_input):
        updated_goal[int(Y[0, i]), i] = 1

    return updated_goal