# -*- coding: utf-8 -*-

"""
Перед вами результаты наблюдений длительности нахождения человека в очереди в зависимости от
количества людей в этой очереди. (1_regression.csv)
Обучите модель линейной регрессии для прогнозирования и введите указанные параметры.
1) Определите выборочное среднее x
2) Определите выборочное среднее y
3) Найдите коэффициент o1 (Ответ округлите до сотых)
4) Найдите коэффициент o0 (Ответ округлите до сотых)
5) Оцените точность модели, вычислив r2 статистику (Ответ округлите до сотых)
"""

import numpy as np
import pandas as pd
from scipy import stats

csv_file = '1_regression.csv'
data = pd.read_csv(csv_file, delimiter=',')

x, y = list(data[u'X']), list(data[u'Y'])

slope, intercept, rvalue, pvalue, stderr = stats.linregress(np.array(x), np.array(y))

print(u'выборочное среднее x: ', np.mean(x))
print(u'выборочное среднее y: ', np.mean(y))
print(u'коэффициент o1: ', round(slope, 2))
print(u'коэффициент o0: ', round(intercept, 2))
print(u'r2 статистика: ', round(rvalue**2, 2))
