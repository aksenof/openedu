# -*- coding: utf-8 -*-

"""
В прилагаемом файле (candy-data.csv) представлены данные, собранные путем голосования за самые лучшие (или, по
крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель линейной многомерной регрессии.
В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer,
hard, bar, pluribus, sugarpercent, pricepercent, отклик — winpercent.
В качестве тренировочного набора данных используйте данные из файла, за иключением следующих
конфет: Dum Dums, Lifesavers big ring gummies. Обучите модель.
1) Введите предсказанное значение winpercent для конфеты Dum Dums.
2) Введите предсказанное значение winpercent для конфеты Lifesavers big ring gummies.
3) Введите предсказанное значение winpercent для конфеты с параметрами: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0.339, 0.564].
Вводите ответы с точностью до трех знаков.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


# чтение данных из CSV файла
data = pd.read_csv('candy-data.csv', delimiter=',', index_col='competitorname')

# тренировочный набор данных
data_train = data.drop(['Dum Dums', 'Lifesavers big ring gummies'])

# отбор данных для предикторов, удаление двух последних столбцов, индекс не включается в данные
X = pd.DataFrame(data_train.drop(['winpercent', 'Y'], axis=1))

# столбец отклика
Y = pd.DataFrame(data_train['winpercent'])

# обучение модели
model = LinearRegression().fit(X, Y)

# предсказание для Dum Dums:
dum_data = data.loc['Dum Dums', :].to_frame().T
dum_pred = model.predict(dum_data.drop(['winpercent', 'Y'], axis=1))
print('[winpercent] Dum Dums:', round(float(dum_pred), 3))

# предсказание для Lifesavers big ring gummies:
life_data = data.loc['Lifesavers big ring gummies', :].to_frame().T
life_pred = model.predict(life_data.drop(['winpercent', 'Y'], axis=1))
print('[winpercent] Lifesavers big ring gummies:', round(float(life_pred), 3))

# предсказание для [0, 0, 1, 1, 0, 0, 0, 0, 0, 0.339, 0.564]:
new_pred = model.predict([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0.339, 0.564]])
print('[winpercent] New data:', round(float(new_pred), 3))
