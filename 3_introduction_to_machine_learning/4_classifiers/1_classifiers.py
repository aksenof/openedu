# -*- coding: utf-8 -*-

"""
На основе анализа некоторого числа писем электронной почты сформирована таблица, содержащая
информацию о классификации писем на группы «спам» и «не спам», а также суммарное число слов
входящих в эти группы:
        SPAM    HAM
Emails  15      24
Words   97      174
Во второй таблице представлены данные, по уникальным словам, и числу их вхождений в указанные группы:
            SPAM    HAM
Cash		2       3
Coupon      3       4
Free        4       3
Purchase    8       7
Access      14      19
Bill        2       27
Million     5       45
Refund      0       7
Investment  18      21
Gift        41      38
Ваша задача построить модель наивного байесовского классификатора и определить класс, к которому будет
отнесено письмо, содержащее текст: Coupon Membership Free Unlimited Gift Investment Purchase
1) Укажите вероятность того, что письмо является спамом, исходя из тренировочного набора данных.
2) Укажите значение логарифма апостериорной вероятности отнесения письма к классу «спам» - y_спам*
3) Укажите значение логарифма апостериорной вероятности отнесения письма к классу «не спам» - y_не_спам*
4) Укажите вероятность отнесения письма к классу «спам» — P(спам|письмо)
Вводите ответы с точностью до трех знаков.
"""

from math import log, e

Emails = {'spam': 15, 'ham': 24}
Words = {'spam':  97, 'ham': 174}
Data = {
    'Cash':       {'spam': 2,  'ham': 3},
    'Coupon':     {'spam': 3,  'ham': 4},
    'Free':       {'spam': 4,  'ham': 3},
    'Purchase':   {'spam': 8,  'ham': 7},
    'Access':     {'spam': 14, 'ham': 19},
    'Bill':       {'spam': 2,  'ham': 27},
    'Million':    {'spam': 5,  'ham': 45},
    'Refund':     {'spam': 0,  'ham': 7},
    'Investment': {'spam': 18, 'ham': 21},
    'Gift':       {'spam': 41, 'ham': 38}
}


def get_log(word, group):
    return log((Data.get(word, {}).get(group, 0) + 1)/(len(Data) + Words.get(group, 0)))


def sum_logs(p, group):
    result = log(p) + \
             get_log('Coupon',     group) + \
             get_log('Membership', group) + \
             get_log('Free',       group) + \
             get_log('Unlimited',  group) + \
             get_log('Gift',       group) + \
             get_log('Investment', group) + \
             get_log('Purchase',   group)
    return result


p_spam = Emails['spam']/(Emails['spam']+Emails['ham'])
p_ham = Emails['ham']/(Emails['spam']+Emails['ham'])
Y_spam = sum_logs(p_spam, 'spam')
Y_ham = sum_logs(p_ham, 'ham')
Ys = list(sorted([Y_spam, Y_ham]))
P_spam_email = round(1/(1 + e**(Ys[0]-Ys[1])), 3)

print('вероятность спам:', round(p_spam, 3))
print('y_спам*:', round(Y_spam, 3))
print('y_не_спам*:', round(Y_ham, 3))
print('P(спам|письмо):', P_spam_email)
