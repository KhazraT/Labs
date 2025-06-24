import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict
from GeneticShortestPath import GeneticShortestPath

class SimulatedAnnealingShortestPath:
    def __init__(self, graph: Dict[int, List[Tuple[int, float]]], start_node: int, end_node: int,
                 initial_temp: float = 1000, final_temp: float = 1, alpha: float = 0.95, max_steps: int = 1000):
        """
        Метод отжига для нахождения кратчайшего пути
        Args:
            graph: Граф в виде словаря {узел: [(сосед, вес), ...]}
            start_node: Начальный узел
            end_node: Конечный узел
            initial_temp: Начальная температура
            final_temp: Конечная температура
            alpha: Коэффициент охлаждения
            max_steps: Количество шагов на каждой температуре
        """
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_steps = max_steps
        self.nodes = list(graph.keys())
        self.best_path = None
        self.best_length = float('inf')
        self.history = []

    def is_reachable(self) -> bool:
        # Проверка достижимости end_node из start_node (BFS)
        from collections import deque
        visited = set()
        queue = deque([self.start_node])
        while queue:
            node = queue.popleft()
            if node == self.end_node:
                return True
            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def random_path(self, max_attempts: int = 100) -> List[int]:
        # Генерируем случайный допустимый путь от start_node до end_node, не более max_attempts попыток
        for _ in range(max_attempts):
            path = [self.start_node]
            current = self.start_node
            visited = set(path)
            while current != self.end_node:
                neighbors = [n for n, _ in self.graph.get(current, []) if n not in visited or n == self.end_node]
                if not neighbors:
                    break  # не удалось построить путь
                next_node = random.choice(neighbors)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            if current == self.end_node:
                return path
        raise RuntimeError(f"Не удалось построить допустимый путь из {self.start_node} в {self.end_node} за {max_attempts} попыток. Возможно, путь не существует.")

    def path_length(self, path: List[int]) -> float:
        if len(path) < 2:
            return float('inf')
        total = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            for neighbor, weight in self.graph.get(current, []):
                if neighbor == next_node:
                    total += weight
                    break
            else:
                return float('inf')
        return total

    def neighbor(self, path: List[int]) -> List[int]:
        # Создаем соседа, изменяя один шаг пути, но оставляя путь допустимым
        if len(path) <= 2:
            return self.random_path()
        # Выбираем случайную позицию для изменения (не start и не end)
        idx = random.randint(0, len(path) - 2 - 1) + 1  # от 1 до len(path)-2
        # Строим новый путь до idx, затем случайно продолжаем до конца
        new_path = path[:idx]
        current = new_path[-1]
        visited = set(new_path)
        while current != self.end_node:
            neighbors = [n for n, _ in self.graph.get(current, []) if n not in visited or n == self.end_node]
            if not neighbors:
                # Если некуда идти, возвращаем случайный путь
                return self.random_path()
            next_node = random.choice(neighbors)
            new_path.append(next_node)
            visited.add(next_node)
            current = next_node
        return new_path

    def anneal(self):
        if not self.is_reachable():
            raise RuntimeError(f"Нет пути из {self.start_node} в {self.end_node} — алгоритм отжига невозможен.")
        current_path = self.random_path()
        current_length = self.path_length(current_path)
        best_path = current_path.copy()
        best_length = current_length
        temp = self.initial_temp
        self.history = [current_length]
        while temp > self.final_temp:
            for _ in range(self.max_steps):
                candidate = self.neighbor(current_path)
                candidate_length = self.path_length(candidate)
                delta = candidate_length - current_length
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_path = candidate
                    current_length = candidate_length
                    if current_length < best_length:
                        best_path = current_path.copy()
                        best_length = current_length
            self.history.append(best_length)
            temp *= self.alpha
        self.best_path = best_path
        self.best_length = best_length
        return best_path, best_length

    def plot_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.history)
        plt.title('Сходимость (отжиг)')
        plt.xlabel('Итерация')
        plt.ylabel('Лучшая длина пути')
        plt.grid(True)
        plt.subplot(1, 3, 2)
        G = nx.DiGraph()
        for node, edges in self.graph.items():
            for neighbor, weight in edges:
                G.add_edge(node, neighbor, weight=weight)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold')
        if self.best_path:
            path_edges = list(zip(self.best_path[:-1], self.best_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3, arrows=True)
        plt.title('Граф с найденным путем (отжиг)')
        plt.subplot(1, 3, 3)
        plt.axis('off')
        stats_text = f"""
        Результаты (отжиг):\n\nНачальный узел: {self.start_node}\nКонечный узел: {self.end_node}\nЛучший путь: {self.best_path}\nДлина пути: {self.best_length:.2f}\n\nПараметры:\nT0: {self.initial_temp}\nTmin: {self.final_temp}\nAlpha: {self.alpha}\nШагов: {self.max_steps}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        plt.title('Статистика')
        plt.tight_layout()
        plt.show()

def example_graph():
    return {
        0: [(1, 2), (2, 4)],
        1: [(2, 1), (3, 7)],
        2: [(3, 3)],
        3: [(4, 1)],
        4: []
    }

def random_graph(num_nodes: int = 8, edge_prob: float = 0.4, min_weight: int = 1, max_weight: int = 10) -> Dict[int, List[Tuple[int, float]]]:
    """
    Генерирует случайный ориентированный граф с весами.
    """
    graph = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_prob:
                weight = random.randint(min_weight, max_weight)
                graph[i].append((j, weight))
    return graph

def compare_algorithms(graph=None, start=None, end=None):
    if graph is None:
        graph = example_graph()
        start, end = 0, 4
    if start is None or end is None:
        nodes = list(graph.keys())
        start, end = nodes[0], nodes[-1]
    # Генетический алгоритм
    ga = GeneticShortestPath(graph, start, end, population_size=50, generations=100)
    ga_path, ga_len = ga.evolve()
    # Метод отжига
    sa = SimulatedAnnealingShortestPath(graph, start, end, initial_temp=1000, final_temp=1, alpha=0.95, max_steps=100)
    sa_path, sa_len = sa.anneal()
    print('Генетический алгоритм: путь', ga_path, 'длина', ga_len)
    print('Метод отжига: путь', sa_path, 'длина', sa_len)
    # Визуализация
    ga.plot_results()
    sa.plot_results()

if __name__ == "__main__":
    seed = input("Введите seed: ")
    if seed:
        try:
            seed = int(seed)
        except:
            seed = 42
    else:
        seed = 42
    random.seed(seed)
    print("Выберите режим:")
    print("1 - Заранее определенный маленький граф")
    print("2 - Случайный граф")
    mode = input("Введите 1 или 2: ").strip()
    if mode == '2':
        n = input("Число вершин (по умолчанию 8): ").strip()
        n = int(n) if n.isdigit() and int(n) > 2 else 8
        graph = random_graph(num_nodes=n)
        print(f"Случайный граф с {n} вершинами сгенерирован.")
        for node, edges in graph.items():
            print(node, edges)
        compare_algorithms(graph, 0, n-1)
    else:
        compare_algorithms()
