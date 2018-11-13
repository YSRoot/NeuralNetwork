# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:07:32 2018

@author: Xiaomi
"""

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
    
    plt.scatter(dots[:,1], dots[:,2], marker='o', color = 'm', label = "$net < 0$")
    plt.scatter(triangle[:,1], triangle[:,2], marker='^', color = 'r', label = "$net> 0$")
    plt.plot(x,y, color = 'k', label = "$net = 0$", linewidth = 3)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 3))
    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
    plt.show()
    
    print("Погрешность - " + str (error))
    pass

#скорость обучения
lr = 0.7

#обучающее множество
inputs=np.array([
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
    
targets = np.array([1,1,0,1,1,0,1,1,1,0,0,1,1,0,0,1,0])

#случайные веса
np.random.seed(2)
W = np.random.random_sample(3)
x = np.arange(np.min(inputs),np.max(inputs),0.2)
W = train(inputs, targets, W, lr)

# Проверка данных на линейную разделимость
#query(inputs,targets,W)

#Проверочные данные
#inpt=np.array([
#        [1,3.2,1.1],
#        [1,2.8,0.7],
#        [1,1.5,0.3],
#        [1,6.0,1.7],
#        [1,7.2,1.3]])
#tt = np.array([1,1,1,0,0])
#query(inpt,tt,W)
