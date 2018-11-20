# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:18:23 2018

@author: Xiaomi
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    return sp.expit(x)

def f1(x):
    return x*(1-x)

def init_weight(inputs,hiddens,hiddens2, outputs):
    w1 = np.random.random((inputs,hiddens))
    
    w2 = np.random.random((hiddens+1,hiddens2))
    
    w3 =  np.random.random((hiddens2+1,outputs))
    return w1,w2,w3

def train(inputs_list,w1,w2,w3,targets_list,lr,error):
    era = 1
    global_error = 1
    list_error = []
    while global_error>error:    
        local_error = []
        for i,inputs in enumerate(inputs_list):
            inputs = np.array(inputs, ndmin=2)
            targets = np.array(targets_list[i], ndmin=2)
            
            hidden_in = np.dot(inputs,w1)
            hidden_out = f(hidden_in)
            hidden_out = np.array(np.insert(hidden_out,0,[1]), ndmin=2)
            
            hidden_in2 = np.dot(hidden_out,w2)
            hidden_out2 = f(hidden_in2)
            hidden_out2 = np.array(np.insert(hidden_out2,0,[1]), ndmin=2)
            
            final_in = np.dot(hidden_out2,w3)
            final_out = final_in
            
            output_error = targets - final_out
            hidden_error2 = np.dot(output_error,w3.T)
            
            hidden_error = np.dot(hidden_error2[:,1:],w2.T)
            
            local_error.append(output_error)
            w3 += lr*output_error*hidden_out2.T
            w2 += lr*hidden_error2[:,1:]*f1(hidden_out2[:,1:])*hidden_out.T
            w1 += lr*hidden_error[:,1:]*f1(hidden_out[:,1:])*inputs.T
        
        local_error = np.asarray(local_error)
        global_error = abs(np.mean(local_error))
#        global_error = np.sqrt(((local_error) ** 2).mean())
        era+=1
        print(global_error)
        list_error.append(global_error)
        if era >100000: break
    return w1,w2,w3,era,list_error

def query(inputs_list, w1,w2,w3):
    final_out = np.array([])
    for i,inputs in enumerate(inputs_list):
            inputs = np.array(inputs, ndmin=2)
            
            hidden_in = np.dot(inputs,w1)
            hidden_out = f(hidden_in)
            hidden_out = np.array(np.insert(hidden_out,0,[1]), ndmin=2)
            
            hidden_in2 = np.dot(hidden_out,w2)
            hidden_out2 = f(hidden_in2)
            hidden_out2 = np.array(np.insert(hidden_out2,0,[1]), ndmin=2)
            
            final_in = np.dot(hidden_out2,w3)

            final_out = np.append(final_out,final_in)
            
    return np.around(final_out)



data = pd.read_csv('titanic_data.csv', index_col='PassengerId')
target_data = data['Survived'].values
data = data.drop(columns=['Survived']).values


inputs = data[0:600]
inputs = np.c_[np.ones(600),inputs]
targets = target_data[0:600]

test = data[600:714]
test = np.c_[np.ones(114),test]
targets_test = target_data[600:714]

lr = 0.3
eps = 10**(-8)

input_layer = 7
hidden_layer = 9
hidden_layer2 = 4
output_layer = 1

w1, w2, w3 = init_weight(input_layer, hidden_layer, hidden_layer2, output_layer)

# train network
w1, w2, w3, era, lst = train(inputs, w1, w2, w3, targets, lr, eps)
print("Количество пройденных эпох - " + str(era))

result_test = query(test,w1,w2,w3)
eq = sum(result_test == targets_test)/len(test)
print("Точность - " + str(eq))

#plt.plot(np.arange(114),result_test,color='r')
#plt.plot(np.arange(114),targets_test,color='b')
plt.plot(np.arange(era-1),lst)
plt.show()