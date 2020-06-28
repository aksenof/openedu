# -*- coding: utf-8 -*-

"""
Вам доступна таблица некоторых синтетических данных, на основании которых необходимо выполнить
классификацию нового объекта, с помощью метода k-ближайших соседей.
Файл данных в формате CSV (task_data.csv).
1) Введите расстояние от нового объекта с координатами (60, 48) до ближайшего соседа,
используя евклидову метрику.
2) Введите идентификаторы трех ближайших точек к (60, 48) для евклидовой метрики.
3) Введите класс для нового объекта с координатами (60, 48) при k=3
и евклидовой метрике.
4) Введите расстояние от нового объекта с координатами (60, 48) до ближайшего соседа,
используя метрику городских кварталов (Манхеттенское расстояние).
5) Введите идентификаторы трех ближайших точек к (60, 48) для городских кварталов (Манхеттенское расстояние).
6) Введите класс для нового объекта с координатами (60, 48) при k=3
и метрике городских кварталов (Манхеттенское расстояние).
"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# словарь расстояний
distances = {'Manhattan': 1, 'Euclid': 2}

# количество ближайших точек
k = 3

# чтение данных
data = pd.read_csv("task_data.csv", delimiter=',', index_col='id')

# отбор данных для предикторов, удаление столбца 'Class', индекс не включается в данные
X = pd.DataFrame(data.drop(['Class'], axis=1))

# столбец отклика 'Class'
Y = pd.DataFrame(data['Class']).values.ravel()

# координаты нового объекта
new_object = [60, 48]


# функция для получения результатов
def get_result(distance):
    p = distances.get(distance, 1)
    neigh = KNeighborsClassifier(n_neighbors=k, p=p)
    neigh.fit(X, Y)
    neigh_cl = neigh.predict([new_object])
    neigh_kn = neigh.kneighbors([new_object])
    print(u'{d}:'.format(d=distance))
    print('расстояние от {obj} до ближайшего соседа:'.format(obj=new_object),
          round(neigh_kn[0][0].tolist()[0], 3))
    print('идентификаторы трех ближайших точек к {obj}: {ind}'.format(
          obj=new_object, ind=','.join(list(str(i+1) for i in neigh_kn[1][0].tolist()))))
    print('класс для {obj}: {cl}'.format(obj=new_object, cl=neigh_cl[0]))


get_result('Euclid')
print('-----')
get_result('Manhattan')
