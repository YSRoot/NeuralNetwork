# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:31:31 2018

@author: Xiaomi
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
# библиотека для работы с csv и dataframe
import pandas as pd

# активационная функция (1/(e^(-x)))
def f(x):
    return sp.expit(x)
# производная активационной функции
def f1(x):
    return x*(1-x)
def init_weight(inputs, hiddens, outputs):
#    матрица весов от входного слоя к 1му скрытому слою
#    принимает рандомные значения и имеет размерность [inputs х hiddens]
    np.random.seed(10)
    w1 = np.random.random((inputs,hiddens))
#    матрица весов от скрытого слоя к выходному слою слою
#    принимает рандомные значения и имеет размерность [hiddens+1 х outputs]
#    эта матрица имеет кол-во строк на единицу больше т.к. нужно учитывать мнимую единицу
    np.random.seed(10)
    w2 =  np.random.random((hiddens+1,outputs))    
    return w1,w2
#функция тренировки сети
# на вход имеет несколько аргументов:
#    inputs_list - обучающее множество (входные сигналы)
#    w1 - матрица весов от входного слоя к 1му скрытому слою
#    w2 - матрица весов от 1го скрытого слоя к 2му скрытому слою
#    targets_list - целевое множество
#    lr - скорость обучения сети
#    error - допустимая погрешность в обучении

def train(inputs_list, targets_list, w1, w2, lr, error):
 #    счетчик эпох
    era = 0
#    глобальная ошибка
    global_error = 1
#    список ошибок
    list_error = []
#   Главный цикл обучения, повторяется пока глобальная ошибка больше погрешности
    while global_error>error:    
#        локальная ошибка 
        local_error = np.array([])
        # побочный цикл, прогоняющий данные с input_list
        # функция enumerate(matrix) возвращает индекс и значение строк
        # которая сохраняется в переменные i, value
        # i - индекс строки input_list
        # value - переменная которая хранит в себе строки матрицы input_list
#        размерность буфера зависит от кол-ва узлов в скрытом слое
        buff_hidden = np.zeros(len(w1[0]))
        
        for i,inputs in enumerate(inputs_list):
            # переводит листа inputs в двумерный вид (для возможности проведения операции транспонирования)
            inputs = np.array(inputs, ndmin=2)
#            targets - содержит локальный таргет для данного инпута
            targets = np.array(targets_list[i], ndmin=2)

#            прямое распространение
#            скалярное произведение строки на матрицу весов
            hidden_in = np.dot(inputs,w1)
#            применение активационной функции к вектору
            hidden_out = f(hidden_in + buff_hidden)
#            добавляем буфер с предедущей итерации
#            этот буфер есть память
            buff_hidden = hidden_out
#            добавление в начало вектора мнимой единицы для обучения сети
            hidden_out = np.array(np.insert(hidden_out,0,[1]), ndmin=2)

#            скалярное произведение строки на матрицу весов            
            final_in = np.dot(hidden_out,w2)
            final_out = f(final_in)
#            final_out = replace(final_out)
#            вычисление ошибки выходного слоя            
            output_error = targets - final_out
#            вычисление ошибки скрытого слоя            
            hidden_error = np.dot(output_error,w2.T)
#            добавление в список локальных ошибок текущую ошибку           
            local_error = np.append(local_error, output_error)
#            метод обратного распространение ошибки
#            изменение матрицы весов 2
            w2 += lr*output_error*f1(final_out)*hidden_out.T
#            в методе обратного распространения ошибки исключается мнимая единичка для совпадения размерностей
#            hidden_error[:,1:] - означает весь вектор за исключением первого элемента            
            w1 += lr*hidden_error[:,1:]*f1(hidden_out[:,1:])*inputs.T
#        глобальная ошибка - это средняя по модулю от всех локальных ошибок
        global_error = abs(np.mean(local_error))
#        global_error = np.sqrt(((local_error) ** 2).mean())
#        эпоха увеличивается на 1
        era+=1
#        вывод в консоль текущую глобальную ошибку
        print(global_error)
#        в список ошибок добавляется глобальная ошибка
        list_error.append(global_error)
#        если при обучении количество эпох превысит порог 10000 то обучение прекратится
        if era >10000: break
#    возвращает измененные веса, количество эпох, и список ошибок, так же возвращаем буфер для теста
    return w1, w2, era, list_error,buff_hidden

# функция для проверки обученной сети и вывода результата
def query(inputs_list, w1,w2,buff):
#    создаем список в котором будем хранить "outs" для тестового множества
    final_out = np.array([])
    for i,inputs in enumerate(inputs_list):
#       прямое распространение так же как и при обучении для получении "out"
        inputs = np.array(inputs, ndmin=2)
        
        hidden_in = np.dot(inputs,w1)
        hidden_out = f(hidden_in+buff)
        buff = hidden_out
        hidden_out = np.array(np.insert(hidden_out,0,[1]), ndmin=2)
        
        final_in = np.dot(hidden_out,w2)
        final_out = np.append(final_out,f(final_in))
#    возвращаем значение вектора "out"       
    return final_out

data = pd.read_csv('air.csv')
# нормолизуем данные
data['#Passengers'] = data['#Passengers']/1000

# сохраняем размерности массива данных
n,m = data.shape
#количество данных, которые будут тестироваться
t = 43
#кол-во данных на котором пройдет обучение
N = n - t

# составляем выборку обучающего множества из первых 600 строк датасета
inputs = data['#Passengers'][0:N].values
#задаем срез выборки
m = 3
lst_inputs = []
# по размерности среза формируем инпуты
for i in range(N-m):
    lst_inputs.append(inputs[i:m+i])
    
inputs = np.array(lst_inputs)

## добавляем столбец мнимых единичек для множества
inputs = np.c_[np.ones(N-m),inputs]
## составляем целевое множество
targets = data['#Passengers'][m:N].values

# из оставшихся 114 строк составляем тестовое множество
test = data['#Passengers'][N:n].values
# копируем данные для будущего теста с полученным результатом
test_fst = test.copy()    

lst_test_inputs = []
# так же формируются тесты с срезами
for i in range(t-m):
    lst_test_inputs.append(test[i:m+i])
test = np.array(lst_test_inputs)  
test = np.c_[np.ones(t-m),test]
targets_test = data['#Passengers'][N:n]

# скорость обучения
lr = 0.7
# допустимая погрешность обучения
eps = 10**(-6)
# количество узлов в входном слое с учетом единичке
# т.е. кол-во столбцов датасета +1 мнимая единичка
input_layer = m+1
# количество узлов в скрытом слое
hidden_layer = 100
# кол-вол узлов в выходном слое равна количеству параметров в выбранном датасете
output_layer = 1

# инициализация весов в зависимости от количества узлов в слоях сети
w1, w2 = init_weight(input_layer, hidden_layer, output_layer)

# train network
w1, w2, era, lst, buff = train(inputs, targets, w1, w2, lr, eps)

# result_test - сохранит значение "outs" NN
result_test = query(test, w1, w2,buff)
# отрисовка графиков
plt.plot(np.arange(len(result_test)),result_test, color='r')
plt.plot(np.arange(40),test_fst[3:43],color='g')
plt.show
