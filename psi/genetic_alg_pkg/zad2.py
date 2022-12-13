import numpy as np
import genetic_alg as ga

def convert_binary_to_decimal(x: np.ndarray) -> int:
    return int("".join(map(str, x)), 2)

def solve_equation(x: np.ndarray) -> int:
    a = convert_binary_to_decimal(x[0:4])
    b = convert_binary_to_decimal(x[4:8])
    return 2 * a**2 + b

if __name__ == "__main__":
    g = ga.GeneticAlgorithm(10, 8, 0.1, solve_equation)

    while True:
        g.evaluate_2()
        if g.best_fitness == 33:
            print(f"Find best gene: {g.best_gene}")
            print(f"a={convert_binary_to_decimal(g.best_gene[0:4])} b={convert_binary_to_decimal(g.best_gene[4:8])}")
            break
        g.selection_2()
        g.mutation_single_inversion()