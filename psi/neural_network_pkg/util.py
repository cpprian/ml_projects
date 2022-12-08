import numpy as np

def gen_goal(_n, Y, amount_input):
    updated_goal = np.zeros((_n, amount_input))

    for i in range(amount_input):
        updated_goal[int(Y[0, i]), i] = 1

    return updated_goal