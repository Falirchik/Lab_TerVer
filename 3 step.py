import random as rand
import math
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def density_xi(x, r):
    if x <= 0:
        return 0
    else:
        return 2 ** (-r / 2) * (1 / math.gamma(r / 2)) * x ** (r / 2 - 1) * math.exp(-x / 2)


def integrate_xi(a, b, r):
    n = 1000
    temp = (b - a) / n
    result = 0
    for i in range(n):
        result += (density_xi(a + temp * i, r) + density_xi(a + temp * (i + 1), r)) * temp / 2
    return result


def F_xi(R0, r):
    return 1 - integrate_xi(0, R0, r)


def n_in_segmet(X, N, l, r):
    n = 0
    for i in range(N):
        if X[i] >= l:
            if X[i] < r:
                n += 1
            else:
                return n
    return n


# эмпирическая функция распределения
# def FN(x, SV, N):
#     if x <= SV[0]:
#         return 0
#     else:
#         for i in range(N - 1):
#             if x <= SV[i + 1]:
#                 return 1. * (i + 1) / N
#             if x > SV[-1]:
#                 return 1
#         if x > SV[-1]:
#             return 1


# функция распределения (теоретическая)
def Fk(y, sigma):
    return math.sqrt(2 * sigma ** 2 * (-1) * math.log(1 - y))

def F(y,sigma):
    return 1 - math.exp(-y**2/(2*sigma**2))


# плотность
def f(y, sigma):
    return y / sigma ** 2 * math.exp(-y ** 2 / (2 * sigma ** 2))


def SampleMean(X, N):  # выборочное среднее
    tmp = 0
    for i in range(N):
        tmp += X[i]
    return tmp / N


def SampleVariance(X, N, xm):  # выборочная дисперсия
    tmp = 0
    for i in range(N):
        tmp += (X[i] - xm) ** 2
    return tmp / N

def Median(sigma):
    return sigma*math.sqrt(math.log(4))


def SampleMedian(X, N):  # выборочная медиана
    if N % 2 == 1:
        return X[N // 2]
    else:
        return (X[N // 2 - 1] + X[N // 2]) / 2


def ExpectedValue(sigma):  # мат ожидание
    return math.sqrt(math.pi / 2) * sigma


def variance(sigma):  # дисперсия
    return sigma ** 2 * (2 - math.pi / 2)


N = int(input("Введите число экспериментов: "))  # размер выборки

SV = [0] * N  # задаем массив
sigma = float(input("sigma = "))
t = [0] * N
for i in range(N):
    rand.seed()
    t[i] = rand.random()
    # print("t= ", t)
    SV[i] = Fk(t[i], sigma)  # добавляем в вектор значение
SV.sort()
print(SV)

# En = ExpectedValue(sigma)
# xm = SampleMean(SV, N)
# Dn = variance(sigma)
# S2 = SampleVariance(SV, N, xm)
# R = SV[N - 1] - SV[0]
# median = SampleMedian(SV, N)
#
# # ТАБЛИЦА ТЕОРЕТИЧЕСКИХ И ВЫБОРОЧНЫХ ЧИСЛОВЫХ ХАРАКТЕРИСТИК
#
# value_list = [[En, xm, math.fabs(En - xm), Dn, S2, math.fabs(Dn - S2), median, R]]
# column_list = ["Математическое ожидание (En)", "Выборочное среднее (x)", "|En - x|", "Дисперсия (Dn)",
#                "Выборочная дисперсия (S2)", "|Dn - S2|", "Выборочная медиана", "Размах выборки"]
# print(tabulate(value_list, column_list, tablefmt="grid"))

def Fn(x):
    if x <= SV[0]: #нет элементов меньше, чем такой x
        return 0
    else:
        for i in range(N-1):
            if x<= SV[i+1]:
                return (1.*(i+1)/N)
            if x>SV[-1]:
                return 1
        if x>SV[-1]:
            return 1
#     # elif x > SV[N-1]: #все элементы выборки меньше, чем такой x
#     #     return 1
#     # else:
#     #     for i in range(1, N): #какая-то часть эелементов больше, какая-то меньше, считаем сколько их
#     #         if x <= SV[i]:
#     #             return i/N
#
# x1 = np.linspace(0, 10, 1000)
# #x1 = [float(i) for i in x1]
#
# Dmer=0
# SVmax=0
# MaxD=0
# for i in range(0,1000):
#     MaxD=abs(Fn(x1[i])-F(x1[i],sigma))
#     if(MaxD>=Dmer):
#         Dmer=MaxD
#         SVmax=x1[i]
# print("Мера расхождения функций распределения =  ",Dmer)

print("Введите число интервалов:")
count_intervals = int(input())
intervals = [0] * (count_intervals - 1)
p = 1 / count_intervals
for i in range(count_intervals - 1):
    intervals[i] = Fk((i+1)*p,sigma)

q = [0] * count_intervals
q[0] = F(intervals[0], sigma)
q[count_intervals - 1] = 1 - F(intervals[count_intervals - 2], sigma)
for i in range(1, count_intervals - 1):
    q[i] = F(intervals[i], sigma) - F(intervals[i - 1], sigma)

value_list3 = [[q[i] for i in range(count_intervals)]]
column_list3 = ["( -infinity; " + str(intervals[0]) + ")"]
for i in range(1, count_intervals - 1):
    column_list3 += ["[ " + str(intervals[i - 1]) + "; " + str(intervals[i]) + ")"]
column_list3 += ["[ " + str(intervals[count_intervals - 2]) + "; +infinity)"]
print(tabulate(value_list3, column_list3, tablefmt="grid"))

new_boards = [0] * (count_intervals + 1)
new_boards[0] = -1
new_boards[count_intervals] = intervals[count_intervals - 2] + 1
for i in range(1,count_intervals):
    new_boards[i] = intervals[i-1]
# for i in range(0,count_intervals + 1):
#     print(new_boards[i])
fn1 = [n_in_segmet(SV,N,new_boards[i-1],new_boards[i])/(N * (new_boards[i] - new_boards[i-1])) for i in range(1, count_intervals + 1)]

fig2 = plt.figure(figsize=(12, 7))
for i in range(1,count_intervals + 1):
    x_temp = [new_boards[i-1], new_boards[i]]
    y_temp = [fn1[i-1], fn1[i-1]]
    plt.plot(x_temp, y_temp, 'black')
plt.show()

print("Введите уровень значимости:")
alpha = float(input())

R0 = (n_in_segmet(SV, N, -10000, intervals[0]) - N * q[0]) ** 2 / (N * q[0])
for i in range(1, count_intervals - 1):
    R0 += (n_in_segmet(SV, N, intervals[i - 1], intervals[i]) - N * q[i]) ** 2 / (N * q[i])
R0 += (n_in_segmet(SV, N, intervals[count_intervals - 2], 10000) - N * q[count_intervals - 1]) ** 2 / (
            N * q[count_intervals - 1])
print(R0)
FR0 = F_xi(R0, count_intervals - 1)
print(FR0)

if FR0 >= alpha:
    print("Гипотеза принята")
else:
    print("Гипотеза отвергнута")

