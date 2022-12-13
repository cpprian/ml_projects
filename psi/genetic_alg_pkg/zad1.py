import numpy as np
import genetic_alg as ga

if __name__ == "__main__":
    g = ga.GeneticAlgorithm(10, 10, 0.6, 
                lambda x: np.sum(x), 
                ga.GeneticAlgorithm.rank_selection, 
                ga.GeneticAlgorithm.single_point_crossover, 
                ga.GeneticAlgorithm.mutation_swap)

    for i in range(100000):
        g.evaluate()
        if np.sum(g.best_fitness) == 10:
            print("Find best gene")
            print(g.population)
            break
        g.selection_1()
        g.mutation_single_inversion()