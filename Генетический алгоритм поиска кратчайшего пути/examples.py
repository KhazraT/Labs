from main import GeneticShortestPath
import random
import matplotlib.pyplot as plt

def create_large_graph(nodes: int = 20, density: float = 0.3) -> dict:
    """Создает большой случайный граф"""
    graph = {}
    
    for i in range(nodes):
        graph[i] = []
        for j in range(nodes):
            if i != j and random.random() < density:
                weight = random.uniform(1, 10)
                graph[i].append((j, weight))
    
    return graph

def create_grid_graph(rows: int, cols: int) -> dict:
    """Создает граф в виде сетки"""
    graph = {}
    
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            graph[node] = []
            
            # Соседи по горизонтали
            if j > 0:
                graph[node].append((node - 1, random.uniform(1, 5)))
            if j < cols - 1:
                graph[node].append((node + 1, random.uniform(1, 5)))
            
            # Соседи по вертикали
            if i > 0:
                graph[node].append((node - cols, random.uniform(1, 5)))
            if i < rows - 1:
                graph[node].append((node + cols, random.uniform(1, 5)))
    
    return graph

def compare_with_dijkstra(graph: dict, start: int, end: int) -> tuple:
    """Сравнивает результат генетического алгоритма с алгоритмом Дейкстры"""
    import heapq
    
    # Алгоритм Дейкстры
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
            
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Восстанавливаем путь
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, distances[end]

def test_different_parameters():
    """Тестирует алгоритм с разными параметрами"""
    print("Тестирование с разными параметрами")
    print("=" * 40)
    
    # Создаем тестовый граф
    graph = create_large_graph(15, 0.4)
    start, end = 0, 14
    
    # Разные размеры популяции
    population_sizes = [20, 50, 100]
    results = []
    
    for pop_size in population_sizes:
        print(f"\nТестирование с размером популяции: {pop_size}")
        
        ga = GeneticShortestPath(
            graph=graph,
            start_node=start,
            end_node=end,
            population_size=pop_size,
            generations=30,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        best_path, best_fitness = ga.evolve()
        results.append((pop_size, best_fitness))
        
        print(f"Результат: {best_fitness:.2f}")
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    pop_sizes, fitnesses = zip(*results)
    plt.bar(pop_sizes, fitnesses)
    plt.xlabel('Размер популяции')
    plt.ylabel('Лучшая длина пути')
    plt.title('Влияние размера популяции на результат')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_grid_problem():
    """Тестирует алгоритм на задаче с сеткой"""
    print("\nТестирование на сетке 5x5")
    print("=" * 30)
    
    # Создаем сетку 5x5
    graph = create_grid_graph(5, 5)
    start, end = 0, 24  # От левого верхнего угла к правому нижнему
    
    print("Граф сетки:")
    for node, edges in graph.items():
        print(f"  {node}: {edges}")
    
    ga = GeneticShortestPath(
        graph=graph,
        start_node=start,
        end_node=end,
        population_size=50,
        generations=100,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    best_path, best_fitness = ga.evolve()
    
    print(f"\nРезультат:")
    print(f"Лучший путь: {best_path}")
    print(f"Длина пути: {best_fitness:.2f}")
    
    # Визуализируем сетку
    ga.plot_results()

def test_comparison_with_dijkstra():
    """Сравнивает генетический алгоритм с алгоритмом Дейкстры"""
    print("\nСравнение с алгоритмом Дейкстры")
    print("=" * 35)
    
    # Создаем небольшой граф для сравнения
    graph = {
        0: [(1, 4), (2, 2)],
        1: [(2, 1), (3, 5)],
        2: [(1, 1), (3, 8), (4, 10)],
        3: [(4, 2)],
        4: [(0, 3)]
    }
    
    start, end = 0, 4
    
    # Генетический алгоритм
    ga = GeneticShortestPath(
        graph=graph,
        start_node=start,
        end_node=end,
        population_size=30,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    ga_path, ga_fitness = ga.evolve()
    
    # Алгоритм Дейкстры
    dijkstra_path, dijkstra_fitness = compare_with_dijkstra(graph, start, end)
    
    print(f"Генетический алгоритм:")
    print(f"  Путь: {ga_path}")
    print(f"  Длина: {ga_fitness:.2f}")
    
    print(f"\nАлгоритм Дейкстры:")
    print(f"  Путь: {dijkstra_path}")
    print(f"  Длина: {dijkstra_fitness:.2f}")
    
    print(f"\nРазница: {abs(ga_fitness - dijkstra_fitness):.2f}")
    
    if abs(ga_fitness - dijkstra_fitness) < 0.01:
        print("✅ Генетический алгоритм нашел оптимальное решение!")
    else:
        print("⚠️ Генетический алгоритм не нашел оптимальное решение")

def run_all_examples():
    """Запускает все примеры"""
    print("Примеры использования генетического алгоритма")
    print("=" * 50)
    
    # Тест с разными параметрами
    test_different_parameters()
    
    # Тест на сетке
    test_grid_problem()
    
    # Сравнение с Дейкстрой
    test_comparison_with_dijkstra()

if __name__ == "__main__":
    run_all_examples() 