# -*- coding: utf-8 -*-

"""
В прилагаемом файле (salary_and_population.csv) представлены данные о средней заработной плате и населению РФ
по регионам на 1 января 2019 год по данным Росстата. Представим ситуацию, что из-за невнимательности операциониста,
регионы: Алтайский край, Ростовская область, Еврейская АО, Ямало-Ненецкий АО, Амурская область
оказались не представлены в итоговой сводке.
Роль невнимательного операциониста придется исполнить Вам (нужно удалить данные по указанным регионам)
и удалить данные по населению, а далее работать уже с новой выборкой.
Определите:
1) выборочное среднее заработной платы (Ответ округлите до сотых)
2) выборочную медиану заработной платы (Ответ округлите до сотых)
3) оценку дисперсии заработной платы (Ответ округлите до сотых)
4) оценку среднеквадратического отклонения заработной платы (Ответ округлите до сотых).
"""

import numpy as np
import pandas as pd

csv_file = 'salary_and_population.csv'

del_regs = [u'Алтайский край',
            u'Ростовская область',
            u'Еврейская АО',
            u'Ямало-Ненецкий АО',
            u'Амурская область']

heads = {u'reg': u'Region_RU',
         u'eng': u'Region_EN',
         u'sal': u'AVG_Salary',
         u'pop': u'Population'}

data = pd.read_csv(csv_file, delimiter=',')
data_reg = data[heads.get(u'reg')]
data_sal = data[heads.get(u'sal')]
new_data_sal = list(data_sal[num] for num, reg in enumerate(data_reg) if reg not in del_regs)

# выборочное среднее заработной платы:
men_new_data_sal = round(float(np.mean(new_data_sal)), 2)

# выборочная медиана заработной платы:
med_new_data_sal = round(float(np.median(new_data_sal)), 2)

# оценка дисперсии заработной платы:
var_new_data_sal = round(float(np.var(new_data_sal)), 2)

# оценка среднеквадратического отклонения заработной платы:
std_new_data_sal = round(float(np.std(new_data_sal)), 2)

print('  mean:', men_new_data_sal)
print('median:', med_new_data_sal)
print('   var:', var_new_data_sal)
print('   std:', std_new_data_sal)
