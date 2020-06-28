# -*- coding: utf-8 -*-

"""
В прилагаемом файле (candy-data.csv) представлены данные, собранные путем голосования за самые лучшие (или, по
крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель логистической регрессии.
В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer,
hard, bar, pluribus, sugarpercent, pricepercent, отклик — Y.
В качестве тренировочного набора данных используйте данные из файла, за иключением
следующих конфет: 3 Musketeers, Candy Corn, Root Beer Barrels. Обучите модель.
Используйте модель с параметрами: LogisticRegression(random_state=2019, solver='lbfgs').
Обучите модель и выполните предсказание для всех конфет из прилагаемого файла (candy-test.csv) тестовых данных.
1) Введите вероятность отнесения конфеты Werthers Original Caramel к классу 1.
2) Введите вероятность отнесения конфеты Sugar Babies к классу 1.
Выполните оценку модели с помощью матрицы ошибок и рассчитайте следующие параметры при
пороге отсечения (Treshhold) 0.5.
3) Введите значение Recall, или TPR для тестового набора данных.
4) Введите значение Precision для тестового набора данных.
5) Введите значение AUC для тестового набора данных.
Вводите ответы с точностью до трех знаков.
"""

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# CSV файлы
csv_data_file = 'candy-data.csv'
csv_test_file = 'candy-test.csv'

# набор данных
data = pd.read_csv(csv_data_file, delimiter=',', index_col='competitorname')

# тренировочный набор данных
data_train = data.drop(['3 Musketeers', 'Candy Corn', 'Root Beer Barrels'])

# отбор данных для предикторов, удаление двух последних столбцов, индекс не включается в данные
X = pd.DataFrame(data_train.drop(['winpercent', 'Y'], axis=1))

# столбец отклика
Y = pd.DataFrame(data_train['Y'])

# обучение модели
model = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, Y.values.ravel())

# тестовый набор данных
data_test = pd.read_csv(csv_test_file, delimiter=',', index_col='competitorname')

# отбор данных для предикторов, удаление последнего столбца, индекс не включается в данные
X_test = pd.DataFrame(data_test.drop(['Y'], axis=1))

# предсказание с помощью обученной модели, порог отсечения по умолчанию составляет 0.5
Y_pred = model.predict(X_test)

# столбец отклика Y из тестовых данных и преобразование в массив
Y_true = data_test['Y'].to_frame().T.values.ravel()

# вероятность отнесения конфеты Werthers Original Caramel к классу 1
wer_class = 1  # 0 или 1
wer_data = data_test.loc['Werthers Original Caramel', :].to_frame().T
wer_pred = model.predict_proba(wer_data.drop(['Y'], axis=1))
print('Werthers Original Caramel:', round(float(wer_pred.tolist()[0][wer_class]), 3))

# вероятность отнесения конфеты Sugar Babies к классу 1
sugar_class = 1  # 0 или 1
sugar_data = data_test.loc['Sugar Babies', :].to_frame().T
sugar_pred = model.predict_proba(sugar_data.drop(['Y'], axis=1))
print('Sugar Babies:', round(float(sugar_pred.tolist()[0][sugar_class]), 3))

# поиск значений Recall, Precision, AUC для тестового набора данных
fpr, tpr, thresholds = metrics.roc_curve(Y_true, Y_pred)
rec = metrics.recall_score(Y_true, Y_pred)
pre = metrics.precision_score(Y_true, Y_pred)
auc = metrics.auc(fpr, tpr)
print('Recall:', round(float(rec), 3))
print('Precision:', round(float(pre), 3))
print('AUC:', round(float(auc), 3))
