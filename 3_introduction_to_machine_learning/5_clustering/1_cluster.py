# -*- coding: utf-8 -*-

"""
Вам доступна таблица некоторых синтетических данных, на основании которых необходимо выполнить
кластеризацию на K=3 кластера методом К-средних. (kmeans.csv)
При выполнении задания с помощью библиотеки sklearn используйте начальную инициализацию со
следующими координатами центроидов и параметрами:
KMeans(n_clusters=3, init=np.array([[8.0, 12.0], [10.57, 7.43], [9.5, 9.5]]), max_iter=100, n_init=1)
1) Укажите, к какому кластеру будет отнесен тот или иной объект в результате кластеризации.
2) По результатам выполнения кластеризации определить среднее расстояний между объектами и центроидом,
отнесенных к кластеру 0.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans

# чтение данных
data = pd.read_csv("kmeans.csv", delimiter=',', index_col='Object')

# удаление столбца Cluster из набора данных
coords = data.drop('Cluster', axis=1)

# координаты центроидов
centroid = np.array([[8.0, 12.0], [10.57, 7.43], [9.5, 9.5]])

# инициализация модели
k_means = KMeans(n_clusters=3, init=centroid, max_iter=100, n_init=1)

# обучение модели на данных из coords
model = k_means.fit(coords)

# вывод назначенных кластеров
clusters = model.labels_.tolist()
print(clusters)

# поиск объектов с кластером 0
cluster = 0
indexes = list(ind for ind, item in enumerate(clusters) if item == cluster)
coords_x = list(x_item for x_ind, x_item in enumerate(list(coords['X'])) if x_ind in indexes)
coords_y = list(y_item for y_ind, y_item in enumerate(list(coords['Y'])) if y_ind in indexes)
cluster_coords = list(zip(coords_x, coords_y))

# вычисление среднего расстояний между объектами и центроидом, отнесенных к кластеру 0
x_cent = np.mean(coords_x)
y_cent = np.mean(coords_y)
result_mean = np.mean(list(dist.euclidean(cord, (x_cent, y_cent)) for cord in cluster_coords))
print(round(float(result_mean), 3))
