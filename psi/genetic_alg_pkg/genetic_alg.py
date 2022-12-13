import copy
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_size, mutation_rate, fitness_function):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function

        self.population = np.random.randint(2, size=(population_size, chromosome_size), dtype=np.int8)
        self.fitness = np.zeros(population_size)
        self.best_fitness = 0
        self.best_gene = None

        # for rank selection
        self.sorted_population = None

    def evaluate(self):
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i])
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_gene = self.population[i]

    def evaluate_2(self):
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i])
            if self.fitness[i] > self.best_fitness and self.fitness[i] <= 33:
                self.best_fitness = self.fitness[i]
                self.best_gene = self.population[i]

    def selection_1(self):
        self.rank_selection()
        parent1 = self.sorted_population[0]
        parent2 = self.sorted_population[1]
        child1, child2 = self.single_point_crossover(parent1, parent2)

        self.sorted_population[self.population_size-1] = child1
        self.sorted_population[self.population_size-2] = child2
        self.population = self.sorted_population

    def selection_2(self):
        new_population = np.zeros((self.population_size, self.chromosome_size), dtype=np.int8)

        for i in range(self.population_size-1):
            idx_parent1 = self.roulette_wheel_selection()
            idx_parent2 = self.roulette_wheel_selection()
            parent1 = self.population[idx_parent1]
            parent2 = self.population[idx_parent2]
    
            new_population[i] = parent1
            new_population[i+1] = parent2

            i += 2

        idx = np.arange(self.population_size)
        np.random.shuffle(idx)

        for i in range(len(idx)//2-1):
            child1, child2 = self.single_point_crossover(new_population[idx[i]], new_population[idx[i+1]])
            new_population[idx[i]] = child1
            new_population[idx[i+1]] = child2
            i += 2

        self.population = new_population

    def selection_3(self):
        new_population = np.zeros((self.population_size, self.chromosome_size))
        eli_population = self.elitism(0.2)
        new_population[:int(self.population_size*0.2)+1] = eli_population

        for i in range(2, int(self.population_size * 0.8)+1, 2):
            idx_parent1 = self.roulette_wheel_selection()
            idx_parent2 = self.roulette_wheel_selection()
            parent1 = self.population[idx_parent1]
            parent2 = self.population[idx_parent2]

            child1, child2 = self.double_point_crossover(parent1, parent2)
            new_population[i] = child1
            new_population[i+1] = child2

        self.population = new_population

    def roulette_wheel_selection(self):
        fitness_sum = np.sum(self.fitness)
        pick = np.random.uniform(0, fitness_sum)
        pick = pick/fitness_sum

        sorted_fitness = np.sort(self.fitness)/fitness_sum
        for i in range(self.population_size):
            if pick >= sorted_fitness[i]:
                return i

        # FIXME: this should not happen
        return self.population_size-1


    def make_sorted_population(self):
        self.sorted_population = copy.deepcopy(self.population)
        self.sorted_population = self.sorted_population[self.fitness.argsort()[::-1]]

    def rank_selection(self):
        sorted_fitness = np.sort(self.fitness)[::-1]
        self.sorted_population = copy.deepcopy(self.population)

        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.fitness[j] == sorted_fitness[i]:
                    self.sorted_population[i] = self.population[j]
                    break

    def single_point_crossover(self, parent1, parent2):
        point = np.random.randint(1, self.chromosome_size - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def double_point_crossover(self, parent1, parent2):
        point1 = np.random.randint(1, self.chromosome_size - 2)
        point2 = np.random.randint(point1 + 1, self.chromosome_size - 1)
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2

    def mutation_single_inversion(self):
        for i in range(2):
            if np.random.uniform(0, 1) < self.mutation_rate:
                point = np.random.randint(low=0, high=self.chromosome_size)
                self.population[i][point] = 1 if self.population[i][point] == 0 else 0

    def mutation_swap(self):
        for i in range(self.population_size):
            if np.random.uniform(0, 1) < self.mutation_rate:
                point1 = np.random.randint(low=0, high=self.chromosome_size)
                point2 = np.random.randint(low=0, high=self.chromosome_size)
                self.population[i][point1], self.population[i][point2] = self.population[i][point2], self.population[i][point1]

    def mutation_adjacent_swap(self):
        for i in range(self.population_size):
            if np.random.uniform(0, 1) < self.mutation_rate:
                point = np.random.randint(low=0, high=self.chromosome_size - 1)
                self.population[i][point], self.population[i][point + 1] = self.population[i][point + 1], self.population[i][point]

    def mutation_inversion(self):
        for i in range(self.population_size):
            if np.random.uniform(0, 1) < self.mutation_rate:
                point1 = np.random.randint(low=0, high=self.chromosome_size)
                point2 = np.random.randint(low=0, high=self.chromosome_size)
                if point1 > point2:
                    point1, point2 = point2, point1
                self.population[i][point1:point2] = np.flip(self.population[i][point1:point2])

    def elitism(self, rate):
        eli_population = np.zeros((int(self.population_size * rate)+1, self.chromosome_size), dtype=np.int8)
        self.make_sorted_population()

        for i in range(int(self.population_size * rate)):
            eli_population[i] = self.sorted_population[i]

        return eli_population