import random as rand
import math
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


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
def FN(x, SV, N):
    if x <= SV[0]:
        return 0
    else:
        for i in range(N - 1):
            if x <= SV[i + 1]:
                return 1. * (i + 1) / N
            if x > SV[-1]:
                return 1
        if x > SV[-1]:
            return 1


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

SV = [0] * N  # задаем массив случайных величин
sigma = float(input("sigma = "))
t = [0] * N
for i in range(N):
    rand.seed()
    t[i] = rand.random()
    # print("t= ", t)
    SV[i] = Fk(t[i], sigma)  # добавляем в вектор значение
SV.sort()
print(SV)

En = ExpectedValue(sigma)
xm = SampleMean(SV, N)
Dn = variance(sigma)
S2 = SampleVariance(SV, N, xm)
R = SV[N - 1] - SV[0]
median = SampleMedian(SV, N)

# ТАБЛИЦА ТЕОРЕТИЧЕСКИХ И ВЫБОРОЧНЫХ ЧИСЛОВЫХ ХАРАКТЕРИСТИК

value_list = [[En, xm, math.fabs(En - xm), Dn, S2, math.fabs(Dn - S2), median, R]]
column_list = ["Математическое ожидание (En)", "Выборочное среднее (x)", "|En - x|", "Дисперсия (Dn)",
               "Выборочная дисперсия (S2)", "|Dn - S2|", "Выборочная медиана", "Размах выборки"]
print(tabulate(value_list, column_list, tablefmt="grid"))

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
x_end = float(input("Введите правую границу оси Х: "))
x1 = np.linspace(0, x_end, 1000)

Dmer=0
SVmax=0
MaxD=0
for i in range(0,1000):
    MaxD=abs(Fn(x1[i])-F(x1[i],sigma))
    if(MaxD>=Dmer):
        Dmer=MaxD
        SVmax=x1[i]
print("Мера расхождения функций распределения =  ",Dmer)

plt.plot(x1, [Fn(i) for i in x1], color='red')
plt.plot(x1, [F(i,sigma) for i in x1], color='blue')
plt.legend(['F^(x)', 'F(x)'], loc=2)
plt.grid()
plt.show()

print("Введите число отрезков для гистограммы")
A = int(input())
boards = [0] * (A + 1)
for i in range(A + 1):
    if i != A:
        print("Введите левую границу " + str(i + 1) + "отрезок:")
    else:
        print("Введите правую границу последнего отрезка:")
    boards[i] = float(input())
fn = [n_in_segmet(SV,N,boards[i-1],boards[i])/(N * (boards[i] - boards[i-1])) for i in range(1, A + 1)]
z = [(boards[i] + boards[i - 1]) / 2 for i in range(1, A + 1)]
f_x = [f(z[i], sigma) for i in range(A)]
value_list2 = [[z[i] for i in range(A)], [f_x[i] for i in range(A)], [fn[i] for i in range(A)]]
column_list2 = ["[" + str(boards[i-1]) + "," + str(boards[i]) + ")" for i in range(1, A + 1)]
print(tabulate(value_list2, column_list2, tablefmt="grid"))

fig1 = plt.figure(figsize=(12, 7))
for i in range(1,A + 1):
    x_temp = [boards[i-1], boards[i]]
    y_temp = [fn[i-1], fn[i-1]]
    plt.plot(x_temp, y_temp, 'black')
plt.show()


