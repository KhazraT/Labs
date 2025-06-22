import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class GeneticKnapsackGA:
    def __init__(self, weights: List[float], values: List[float], capacity: float,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Генетический алгоритм для задачи о рюкзаке
        Args:
            weights: веса предметов
            values: ценности предметов
            capacity: вместимость рюкзака
            population_size: размер популяции
            generations: количество поколений
            mutation_rate: вероятность мутации
            crossover_rate: вероятность скрещивания
        """
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_items = len(weights)
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []

    def create_individual(self) -> np.ndarray:
        """Создает случайную особь (решение)"""
        return np.random.randint(0, 2, self.num_items)

    def fitness_function(self, individual: np.ndarray) -> float:
        """Функция приспособленности: суммарная ценность, если не превышен вес, иначе 0"""
        total_weight = np.sum(individual * self.weights)
        total_value = np.sum(individual * self.values)
        if total_weight > self.capacity:
            return 0
        return total_value

    def tournament_selection(self, population: List[np.ndarray], k: int = 3) -> np.ndarray:
        """Турнирный отбор"""
        tournament = random.sample(population, k)
        return max(tournament, key=self.fitness_function)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Одноточечное скрещивание"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, self.num_items - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Мутация: инверсия случайного бита"""
        mutant = individual.copy()
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                mutant[i] = 1 - mutant[i]
        return mutant

    def evolve(self) -> Tuple[np.ndarray, float]:
        """Основной цикл эволюции"""
        population = [self.create_individual() for _ in range(self.population_size)]
        for generation in range(self.generations):
            fitness_scores = [(ind, self.fitness_function(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_solution = fitness_scores[0][0].copy()
            self.fitness_history.append(self.best_fitness)
            new_population = []
            elite_size = max(1, self.population_size // 10)
            new_population.extend([ind for ind, _ in fitness_scores[:elite_size]])
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
            if generation % 10 == 0:
                print(f"Поколение {generation}: Лучшая ценность = {self.best_fitness:.2f}")
        return self.best_solution, self.best_fitness

    def plot_results(self):
        """Визуализация результатов"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_history)
        plt.title('Сходимость алгоритма')
        plt.xlabel('Поколение')
        plt.ylabel('Лучшая ценность')
        plt.grid(True)
        plt.show()
