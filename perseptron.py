# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def train(inputs_list, target_list, weight, learning_rate):
    # n = 6, m = 3
    n,m = inputs_list.shape
    #количество эпох
    era=0
    while True:
        error=0
        for i,value in enumerate(inputs_list):
            net = np.dot(value, weight)
            if net < 0:
                out = 0
            else:
                out = 1
           #out = 0. if net < 0 else 1.
            local_error = target_list[i] - out
            if local_error != 0:    
                weight += value*learning_rate*local_error
                error += 1
        #cчетчик эпох
        era+=1
        #Просмотр за статусом обучений персептрона
        #query(inputs_list,target_list,weight)
        
        #если ошибок нет выход из цикла
        if error == 0: break
    print("Количество пройденных эпох - " + str(era))
    return weight

def query(inputs_list, targets_list, weight):
    n,m = inputs_list.shape
    error = 0
    dots = []
    triangle = []
    x = np.arange(np.min(inputs_list), np.max(inputs_list),0.02)
    y = np.vectorize(lambda x: ((-weight[0]-weight[1]*x)/weight[2]))
    y = y(x)
    for i,value in enumerate(inputs_list):
        net = np.dot(value, W)
        if net < 0:
            dots.append(value)
            out = 0
        if net > 0:
            triangle.append(value)
            out = 1
            
        local_error = targets_list[i] - out
        error += local_error
        
    dots = np.asarray(dots)
    triangle = np.asarray(triangle)
# библиотека для работы с матричными операциями
import numpy as np
# библиотека для работы с графиками
import matplotlib.pyplot as plt

# Функция тренировки нейронной сети
# inputs_list – входные данные,
# targets_list – целевые данные,
# weight - веса,
# learning_rate – скорость обучения
def train(inputs_list, targets_list, weight, learning_rate):
    # размерность матрицы входных данных (input_list)
    n, m = inputs_list.shape
    # количество эпох
    era = 0
    # Главный цикл обучения, повторяется пока глобальная ошибка не будет равна 0
    while True:
        # Глобальная ошибка
        error = 0
        # побочный цикл, прогоняющий данные с input_list
        # функция enumerate(matrix) возвращает индекс и значение строк
        # которая сохраняется в переменные i, value
        # i - индекс строки input_list
        # value - переменная которая хранит в себе строки матрицы input_list
        for i,value in enumerate(inputs_list):
            # вычисляется net с помощью скалярного произведения (вектора на вектор)
            net = np.dot(value, weight)
            # если условия выполняются out = 0 иначе out = 1
            if net < 0:
                out = 0
            else:
                out = 1
            # условие выше можно написать и по-другому
            # out = 0. if net < 0 else 1.
            
            # вычисляется отклонение результата от правильного ответа
            # это отклонение сохраняется в переменную local_error
            local_error = targets_list[i] - out
            # если локальная ошибка существует то
            if local_error != 0:    
                # изменяются веса
                weight += value*learning_rate*local_error
                # и глобальная ошибка увеличивается на единицу
                error += 1
        # cчетчик эпох
        era += 1
        # Просмотр за статусом обучения персептрона
#        query(inputs_list,targets_list,weight)
        
        # если глобальная ошибка равна 0, т.е. ошибок нет, выход из цикла
        if error == 0: break
    # вывод количества пройденных эпох
    print("Количество пройденных эпох - " + str(era))
    # возвращается веса в конце выполнения работы функции
    return weight

# функция для проверки обученной сети и вывода результата
def query(inputs_list, targets_list, weight):
    # размерность матрицы input_list
    n,m = inputs_list.shape
    # Глобальная ошибка
    error = 0
    # x - вектор хранящий в себе диапазон значений
    # от минимального значения до максимального, матрицы input_list с шагом 0.02
    x = np.arange(np.min(inputs_list), np.max(inputs_list),0.02)
    # Y функция y = a*x + b для отрисовки прямой разделяющей данные
    Y = np.vectorize(lambda x: ((-weight[0]-weight[1]*x)/weight[2]))
    # y получает значение "Y(x)" для каждого "x" (y — это вектор)
    y = Y(x)
    # цикл, прогоняющий данные с input_list
    # функция enumerate(matrix) возвращает индекс и значение строк
    # которая сохраняется в переменные i, value
    # i - индекс строки input_list
    # value - переменная которая хранит в себе строки матрицы input_list
    for i,value in enumerate(inputs_list):
        # вычисляется net с помощью скалярного произведения вектора на вектор
        net = np.dot(value, weight)
        # если условия выполняются out = 0 иначе out = 1
        if net < 0:
            out = 0
        else:
            out = 1
        
        # отрисовка точек в зависимости от класса (мужчины или женщины)
        if(targets_list[i] == 0):
            # plt.scatter(x,y, marker, color)
            # scatter() принимает на вход несколько аргументов:
            # "х", "у", вид маркера, цвет маркера
            # если условие выполняется - женщины
            plt.scatter(value[1], value[2], marker='o', color = 'm')
        else:
            # иначе - мужчины
            plt.scatter(value[1], value[2], marker='^', color = 'r')
            
        # вычисляется отклонение результата от правильного ответа
        # это отклонение сохраняется в переменную local_error
        local_error = targets_list[i] - out
        # если локальная ошибка существует, то глобальная ошибка увеличивается
        error += local_error
    # plt.plot() рисует прямую по координатам
    # на вход идет вектор "х" и вектор "у"
    # так же цвет и толщина линии
    plt.plot(x, y, color = 'k', linewidth = 3)
    # в переменные x1, x2, y1, y2 сохраняются нижние и верхние границы графика
    x1, x2, y1, y2 = plt.axis()
    # ставятся ограничения отображения графика по оси "у" от 0 до 3
    plt.axis((x1, x2, 0, 3))
    # показать график
    plt.show()
    # Вывод ошибки (погрешности)
    print("Погрешность - " + str(error))
    pass

#скорость обучения
lr = 0.7

#обучающее множество
inputs = np.array([
        [1.,0.3,0.2],
        [1.,0.3,0.2],
        [1.,8.0,1.5],
        [1.,0.7,0.8],
        [1.,2.1,0.5],
        [1.,7.2,1.7],
        [1.,1.7,0.7],
        [1.,4.7,0.9],
        [1.,2.9,0.3],
        [1.,6.5,1.5],
        [1.,5.7,1.6],
        [1.,4.1,0.3],
        [1.,3.2,0.1],
        [1.,7.0,1.2],
        [1.,8.0,1.3],
        [1.,4.1,0.6],
        [1.,6.5,1.4]])
# целевые данные    
targets = np.array([1,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,0])

# np.random.seed ставит «семя» для функции random
# в питоне, как и в других высокоуровневых языках стоит псевдорандом
# возвращающее число, созданное на основе другого числа
# по умолчанию — время на данный момент
# мы меняем стандартное значение «seed» на свое
np.random.seed(2)
#случайные веса
W = np.random.random_sample(3)
# тренировка сети, возвращает веса и сохраняет в переменную W
W = train(inputs, targets, W, lr)

# Проверка данных на линейную разделимость
#query(inputs, targets, W)

# Данные для проверки на необученных значениях
# раскомментируйте и проверьте на правильность обученной сети
inpt=np.array([
        [1,3.2,1.1],
        [1,2.8,0.7],
        [1,1.5,0.3],
        [1,6.0,1.7],
        [1,7.2,1.3]])
tt = np.array([1,1,1,0,0])
query(inpt, tt, W)
