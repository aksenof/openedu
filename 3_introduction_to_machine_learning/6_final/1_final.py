# -*- coding: utf-8 -*-

"""
В Базе Данных есть таблица pulsar_stars, в которой содержатся сведения о
звездах, полученные в ходе исследовании вселенной (High Time Resolution Universe Survey) с целью
определения одного из типа нейтронных звезд — пульсаров. Поле TARGET таблицы pulsar_stars является
откликом, все остальные поля — предикторы.
Вам необходимо получить выборку из таблицы  с помощью запросов на основании следующих критериев:
Все строки таблицы, где TARGET = 0 и MIP ∈ [104.1953125, 104.5859375];
Все строки таблицы, где TARGET = 1 и MIP ∈ [101.9609375, 115.3515625].
1) Укажите число строк в полученной выборке.
2) Определите выборочное среднее для столбца MIP.
Выполните линейную нормировку всех значений предикторов полученной выборки
3) Определите выборочное среднее для столбца MIP после нормировки.
Обучите модель логистической регрессии, используя полученную после нормировки выборку в качестве
тренировочного набора данных. Используйте следующие параметры:
В Python, используйте модель с параметрами:
LogisticRegression(random_state=2019, solver='lbfgs').
Выполните классификацию новой звезды с параметрами:
[0.157, 0.311, 0.676, 0.586, 0.307, 0.848, 0.673, 0.64]
4) Введите вероятность отнесения звезды к классу пульсар.
Выполните классификацию новой звезды, с помощью метода k-ближайших соседей, используя
нормализованные данные выборки.
5) Введите расстояние от новой звезды до ближайшего соседа, используя евклидову метрику.
6) Введите класс для новой звезды при k=5 и евклидовой метрике.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, neighbors

# чтение данных
data = pd.read_csv('pulsar_stars.csv', delimiter=',')

# число строк в полученной выборке:
# SELECT COUNT(*)
# FROM PULSAR_STARS
# WHERE (TARGET = 0 AND MIP >= 104.1953125 AND MIP <= 104.5859375)
# OR (TARGET = 1 AND MIP >= 101.9609375 AND MIP <= 115.3515625)
print('COUNT:', len(data))

# выборочное среднее для столбца MIP:
# SELECT ROUND(AVG(MIP), 3)
# FROM PULSAR_STARS
# WHERE (TARGET = 0 AND MIP >= 104.1953125 AND MIP <= 104.5859375)
# OR (TARGET = 1 AND MIP >= 101.9609375 AND MIP <= 115.3515625);
print('AVG MIP:', round(float(np.mean(list(data['MIP']))), 3))

# отбор данных для предикторов, удаление столбца TARGET
X = pd.DataFrame(data.drop(['TARGET'], axis=1))

# столбец отклика
Y = pd.DataFrame(data['TARGET']).values.ravel()

# линейная нормировка всех значений предикторов
linear_norm = preprocessing.MinMaxScaler().fit_transform(X)
X_norm = pd.DataFrame(linear_norm, columns=X.columns)

# выборочное среднее для столбца MIP после нормировки
print('AVG MIP norm:', round(float(np.mean(list(X_norm['MIP']))), 3))

# обучение модели логистической регрессии
log_reg = linear_model.LogisticRegression(random_state=2019, solver='lbfgs').fit(X_norm, Y)

# вероятность отнесения новой звезды к классу пульсар
star_class = 1  # 1 - звезда является пульсаром, 0 - звезда не является пульсаром
new_star = [0.157, 0.311, 0.676, 0.586, 0.307, 0.848, 0.673, 0.64]
new_star_pred = log_reg.predict_proba([new_star])
print('New Star:', round(float(new_star_pred.tolist()[0][star_class]), 3))

# расстояние от новой звезды до ближайшего соседа и класс для новой звезды при k=5 и евклидовой метрике
neighs = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
neighs.fit(X_norm, Y)
neigh_kn = neighs.kneighbors([new_star])
neigh_cl = neighs.predict([new_star])
print('New Star neighbor:', round(neigh_kn[0][0].tolist()[0], 3))
print('New Star class:', neigh_cl[0])
