import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, Set

class GeneticShortestPath:
    def __init__(self, graph: Dict[int, List[Tuple[int, float]]], 
                 start_node: int, end_node: int,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Генетический алгоритм для нахождения кратчайшего пути
        
        Args:
            graph: Граф в виде словаря {узел: [(сосед, вес), ...]}
            start_node: Начальный узел
            end_node: Конечный узел
            population_size: Размер популяции
            generations: Количество поколений
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность скрещивания
        """
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.nodes = list(graph.keys())
        self.best_path = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def create_individual(self) -> List[int]:
        """Создает случайную особь (путь)"""
        # Начинаем с начального узла
        path = [self.start_node]
        current = self.start_node
        
        # Строим путь случайным образом
        max_steps = len(self.nodes) * 2  # Ограничиваем длину пути
        steps = 0
        
        while current != self.end_node and steps < max_steps:
            if current in self.graph:
                neighbors = [n for n, _ in self.graph[current]]
                if neighbors:
                    # Выбираем случайного соседа
                    next_node = random.choice(neighbors)
                    if next_node not in path:  # Избегаем циклов
                        path.append(next_node)
                        current = next_node
                    else:
                        # Если попали в цикл, выбираем случайный узел
                        available = [n for n in self.nodes if n not in path]
                        if available:
                            current = random.choice(available)
                            path.append(current)
                        else:
                            break
                else:
                    break
            else:
                break
            steps += 1
        
        # Если не дошли до конечного узла, добавляем его
        if path[-1] != self.end_node:
            path.append(self.end_node)
            
        return path
    
    def calculate_path_length(self, path: List[int]) -> float:
        """Вычисляет длину пути"""
        if len(path) < 2:
            return float('inf')
        
        total_length = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if current in self.graph:
                # Ищем вес ребра
                edge_weight = None
                for neighbor, weight in self.graph[current]:
                    if neighbor == next_node:
                        edge_weight = weight
                        break
                
                if edge_weight is not None:
                    total_length += edge_weight
                else:
                    return float('inf')  # Ребро не существует
            else:
                return float('inf')  # Узел не существует в графе
        
        return total_length
    
    def fitness_function(self, path: List[int]) -> float:
        """Функция приспособленности (меньше = лучше)"""
        if len(path) < 2 or path[0] != self.start_node or path[-1] != self.end_node:
            return float('inf')
        
        # Проверяем, что путь корректен
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            if current in self.graph:
                neighbors = [n for n, _ in self.graph[current]]
                if next_node not in neighbors:
                    return float('inf')
            else:
                return float('inf')
        
        return self.calculate_path_length(path)
    
    def tournament_selection(self, population: List[List[int]], k: int = 3) -> List[int]:
        """Турнирный отбор"""
        tournament = random.sample(population, k)
        return min(tournament, key=self.fitness_function)
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Скрещивание двух родителей"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2
        
        # Находим общие узлы
        common_nodes = set(parent1) & set(parent2)
        if len(common_nodes) < 2:
            return parent1, parent2
        
        # Выбираем случайный общий узел для точки скрещивания
        crossover_point = random.choice(list(common_nodes))
        
        # Находим позиции точки скрещивания в обоих родителях
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        
        # Создаем потомков
        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]
        
        # Убираем дубликаты, сохраняя порядок
        child1 = self.remove_duplicates(child1)
        child2 = self.remove_duplicates(child2)
        
        return child1, child2
    
    def remove_duplicates(self, path: List[int]) -> List[int]:
        """Удаляет дубликаты из пути, сохраняя порядок"""
        seen = set()
        result = []
        for node in path:
            if node not in seen:
                result.append(node)
                seen.add(node)
        return result
    
    def mutate(self, path: List[int]) -> List[int]:
        """Мутация особи"""
        if len(path) < 3:
            return path
        
        mutated_path = path.copy()
        
        # Случайная мутация: замена случайного узла
        if random.random() < self.mutation_rate:
            # Выбираем случайную позицию (кроме начала и конца)
            pos = random.randint(1, len(mutated_path) - 2)
            
            # Выбираем случайный узел из графа
            available_nodes = [n for n in self.nodes if n not in mutated_path]
            if available_nodes:
                mutated_path[pos] = random.choice(available_nodes)
        
        # Мутация: добавление случайного узла
        if random.random() < self.mutation_rate and len(mutated_path) < len(self.nodes):
            pos = random.randint(1, len(mutated_path))
            available_nodes = [n for n in self.nodes if n not in mutated_path]
            if available_nodes:
                mutated_path.insert(pos, random.choice(available_nodes))
        
        # Мутация: удаление случайного узла
        if random.random() < self.mutation_rate and len(mutated_path) > 3:
            pos = random.randint(1, len(mutated_path) - 2)
            mutated_path.pop(pos)
        
        return mutated_path
    
    def repair_path(self, path: List[int]) -> List[int]:
        """Исправляет путь, если он некорректен"""
        if len(path) < 2:
            return self.create_individual()
        
        # Убеждаемся, что путь начинается и заканчивается правильно
        if path[0] != self.start_node:
            path[0] = self.start_node
        
        if path[-1] != self.end_node:
            path[-1] = self.end_node
        
        # Убираем дубликаты
        path = self.remove_duplicates(path)
        
        # Проверяем корректность пути
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if current in self.graph:
                neighbors = [n for n, _ in self.graph[current]]
                if next_node not in neighbors:
                    # Если ребро не существует, пытаемся найти путь через другие узлы
                    return self.create_individual()
            else:
                return self.create_individual()
        
        return path
    
    def evolve(self) -> Tuple[List[int], float]:
        """Основной цикл эволюции"""
        # Создаем начальную популяцию
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Вычисляем приспособленность
            fitness_scores = [(individual, self.fitness_function(individual)) 
                            for individual in population]
            
            # Сортируем по приспособленности
            fitness_scores.sort(key=lambda x: x[1])
            
            # Обновляем лучший результат
            if fitness_scores[0][1] < self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_path = fitness_scores[0][0].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Создаем новую популяцию
            new_population = []
            
            # Элитизм: сохраняем лучших особей
            elite_size = max(1, self.population_size // 10)
            new_population.extend([individual for individual, _ in fitness_scores[:elite_size]])
            
            # Генерируем остальных особей
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Скрещивание
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # Мутация
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    # Исправление путей
                    child1 = self.repair_path(child1)
                    child2 = self.repair_path(child2)
                    
                    new_population.extend([child1, child2])
                else:
                    # Клонирование с мутацией
                    parent = self.tournament_selection(population)
                    child = self.mutate(parent.copy())
                    child = self.repair_path(child)
                    new_population.append(child)
            
            # Обновляем популяцию
            population = new_population[:self.population_size]
            
            # Выводим прогресс
            if generation % 10 == 0:
                print(f"Поколение {generation}: Лучшая длина пути = {self.best_fitness:.2f}")
        
        return self.best_path, self.best_fitness
    
    def plot_results(self):
        """Визуализирует результаты"""
        plt.figure(figsize=(15, 5))
        
        # График сходимости
        plt.subplot(1, 3, 1)
        plt.plot(self.fitness_history)
        plt.title('Сходимость алгоритма')
        plt.xlabel('Поколение')
        plt.ylabel('Лучшая длина пути')
        plt.grid(True)
        
        # Визуализация графа
        plt.subplot(1, 3, 2)
        G = nx.DiGraph()
        
        # Добавляем узлы и ребра
        for node, edges in self.graph.items():
            for neighbor, weight in edges:
                G.add_edge(node, neighbor, weight=weight)
        
        pos = nx.spring_layout(G)
        
        # Рисуем граф
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        
        # Рисуем лучший путь
        if self.best_path:
            path_edges = list(zip(self.best_path[:-1], self.best_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                  edge_color='red', width=3, arrows=True)
        
        plt.title('Граф с найденным путем')
        
        # Статистика
        plt.subplot(1, 3, 3)
        plt.axis('off')
        stats_text = f"""
        Результаты:
        
        Начальный узел: {self.start_node}
        Конечный узел: {self.end_node}
        Лучший путь: {self.best_path}
        Длина пути: {self.best_fitness:.2f}
        
        Параметры:
        Размер популяции: {self.population_size}
        Поколений: {self.generations}
        Мутация: {self.mutation_rate}
        Скрещивание: {self.crossover_rate}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        plt.title('Статистика')
        
        plt.tight_layout()
        plt.show()
