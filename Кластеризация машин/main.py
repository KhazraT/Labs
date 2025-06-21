import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'cars.csv'
data = pd.read_csv(file_path)

# Выбираем числовые признаки для кластеризации
features = [
    'Horsepower', 'MPG_Highway', 'MPG_City', 'Weight', 'EngineSize', 'Cylinders'
]
X = data[features].dropna()

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального числа кластеров (метод локтя)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Число кластеров')
plt.ylabel('Inertia')
plt.title('Метод локтя для выбора числа кластеров')
plt.grid(True)
plt.show()

# Кластеризация (например, на 16 кластеров)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Добавляем метки кластеров к исходным данным
X['Cluster'] = clusters

# Анализ кластеров
print('Размеры кластеров:')
print(X['Cluster'].value_counts())
print('\nСредние значения признаков по кластерам:')
print(X.groupby('Cluster').mean())

# Визуализация (по первым двум компонентам PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in range(n_clusters):
    plt.scatter(
        X_pca[clusters == cluster, 0],
        X_pca[clusters == cluster, 1],
        label=f'Кластер {cluster}'
    )
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Кластеры автомобилей (PCA)')
plt.legend()
plt.grid(True)
plt.show()
