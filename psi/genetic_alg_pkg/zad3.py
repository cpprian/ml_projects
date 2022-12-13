import numpy as np
import genetic_alg as ga

wage = np.array([3, 13, 10, 9, 7, 1, 8, 8, 2, 9])
value = np.array([266, 442, 671, 526, 388, 245, 210, 145, 126, 322])

def solve_knapsack_problem(x: np.ndarray) -> int:
    curr_wage = 0
    curr_value = 0

    for (i, y) in enumerate(x):
        if y: 
            curr_wage += wage[i]
            curr_value += value[i]

    if curr_wage > 35:
        return 0
    return curr_value


if __name__ == "__main__":
    g = ga.GeneticAlgorithm(8, 10, 0.05, solve_knapsack_problem)

    for i in range(10000):
        g.evaluate()
        g.selection_3()
        g.mutation_inversion()

    print("Best gene: ", g.best_gene)
    print(f"Value: {g.best_fitness}")
    print("Population:")
    print(g.population)