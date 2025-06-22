from main import GeneticKnapsackGA
import numpy as np
import matplotlib.pyplot as plt

def create_knapsack_example():
    # Пример данных задачи о рюкзаке
    weights = [6, 2, 9, 1, 10, 10, 6, 5, 1, 10, 7, 6, 7, 7, 10, 6, 6, 3, 8, 2]
    values = [8, 6, 4, 8, 10, 4, 8, 10, 5, 2, 5, 3, 3, 3, 10, 8, 8, 3, 2, 5]
    capacity = 40
    return weights, values, capacity

def test_knapsack():
    print("Тест: задача о рюкзаке")
    weights, values, capacity = create_knapsack_example()
    ga = GeneticKnapsackGA(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    best_solution, best_fitness = ga.evolve()
    print(f"\nЛучшее решение: {best_solution}")
    print(f"Суммарная ценность: {best_fitness}")
    print(f"Суммарный вес: {np.sum(np.array(weights) * best_solution)}")
    ga.plot_results()

def test_different_parameters():
    print("\nТест с разными размерами популяции")
    weights, values, capacity = create_knapsack_example()
    population_sizes = [20, 50, 100]
    results = []
    for pop_size in population_sizes:
        print(f"\nПопуляция: {pop_size}")
        ga = GeneticKnapsackGA(
            weights=weights,
            values=values,
            capacity=capacity,
            population_size=pop_size,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        _, best_fitness = ga.evolve()
        results.append((pop_size, best_fitness))
        print(f"Результат: {best_fitness}")
    plt.figure(figsize=(8, 5))
    pop_sizes, fitnesses = zip(*results)
    plt.bar(pop_sizes, fitnesses)
    plt.xlabel('Размер популяции')
    plt.ylabel('Лучшая ценность')
    plt.title('Влияние размера популяции на результат')
    plt.grid(True, alpha=0.3)
    plt.show()

def run_all_examples():
    test_knapsack()
    test_different_parameters()

if __name__ == "__main__":
    run_all_examples()
