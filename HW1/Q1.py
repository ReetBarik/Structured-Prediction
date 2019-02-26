# -*- coding: utf-8 -*-
"""
Created on Sun Feb 05 23:26:11 2019

@author: Reet Barik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def phi(x, y, order):
    
    labels = len(alphabet)
    phi_unary = np.zeros(token * labels)
    phi_binary = np.zeros(labels ** 2)
    phi_ternary = np.zeros(labels ** 3)
    phi_quarternary = np.zeros(labels ** 4)
    
    for i in range(len(y)):
        
        phi_unary[token * y[i] : token * (y[i] + 1)] += x[i]
        if (i < len(y) - 1):
            phi_binary[y[i] * labels + y[i + 1]] += 1
        if (i < len(y) - 2):
            phi_ternary[y[i] * (labels ** 2) + y[i + 1] * labels + y[i + 2]] += 1
        if (i < len(y) - 3):
            phi_quarternary[y[i] * (labels ** 3) + y[i + 1] * (labels ** 2) + y[i + 2] * labels + y[i + 3]] += 1
    
    if (order == 1):
        return np.concatenate((phi_unary, phi_binary))
    if (order == 2):
        return np.concatenate((phi_unary, phi_binary, phi_ternary))
    if (order == 3):
        return np.concatenate((phi_unary, phi_binary, phi_ternary, phi_quarternary))
        
    
    
def rgs(x, phi, w, R, order):
    
    y_hat = []
    y_hat_scores = []
    for i in range(R):
        y_pred = np.random.randint(0, len(alphabet), size=(len(x)))
        s_best = np.dot(w, phi(x, y_pred, order))
        
        flag = False
        while True:
            if (flag):
                break
            for j in range(len(x)):
                if (flag):
                    break
                for label in range(len(alphabet)):
                    if (flag):
                        break
                    temp = y_pred[j]
                    y_pred[j] = label
                    s = np.dot(w, phi(x, y_pred, order))
                        
                    if (s > s_best):
                        s_best = s;
                    else:
                        y_pred[j] = temp
                        flag = True
                        
        y_hat.append(y_pred)
        y_hat_scores.append(s_best)
                        
    return y_hat[np.argmax(y_hat_scores)]
                        


def structured_perceptron(phi, order, R, eta, MAX, w = None):
    
    training = w is None
    accuracy = []
    labels = len(alphabet)
    
    if (training):
        print('\nTraining')
        data = train_data
        if (order == 1):
            w = np.zeros((token * labels) + (labels ** 2))
        if (order == 2):
            w = np.zeros((token * labels) + (labels ** 2) + (labels ** 3))
        if (order == 3):
            w = np.zeros((token * labels) + (labels ** 2) + (labels ** 3) + (labels ** 4))
    else:
        print('\nTesting')
        data = test_data
    
    print('Feature Order = ' + str(order) +'\n')
    
    for i in range(MAX):
        correct = 0
        count = 0
        for j in range(len(data['X'])):
            x = data['X'][j]
            y = data['Y'][j]
            
            count += len(y)
            y_hat = rgs(x, phi, w, R, order)
            
            correct += np.sum(y == y_hat)
            
            if (training):
                if (np.sum(y_hat != y) > 0):
                    w = (w + (eta * (phi(x, y, order) - phi(x, y_hat, order))))
                    
        accuracy.append((correct * 100) / count)
        print('Iteration = ' + str(i))
        print('Number of correct predictions = ' + str(correct) + ' ,out of ' + str(count))
            
    
    plt.plot(range(MAX), accuracy, color='orange', marker='o')
    plt.xlabel('No. of iterations')
    plt.ylabel('Hamming Accuracy')
    if (training):        
        plt.title('Hamming Accuracy on training set, Order =' + str(order))
    else:
        plt.title('Hamming Accuracy on testing set, Order =' + str(order))
    
    if (dataset == 0):
        name = 'Nettalk'
        if (training):
            name += 'Training' + str(order)
        else:
            name += 'Testing' + str(order) 
    else:
        name = 'OCR'
        if (training):
            name += 'Training' + str(order)
        else:
            name += 'Testing' + str(order) 
            
    name += '.png'
    
    plt.savefig(name)
    plt.close()

    return w
                   
phi_order = [1,2,3]   

print()
print('Text to speech Dataset\n') 
dataset = 0

train_df_1 = pd.read_csv("data/nettalk_stress_train.txt", sep='\t', header=None, names=['index', 'x', 'y', 'delim'])
train_df_1["x"] = train_df_1["x"].str.replace("im", "")

train_data = {}
x = []
y = []
alphabet = set()
for i in range(len(train_df_1)):

    if ((train_df_1.iloc[i - 1]['index'] > train_df_1.iloc[i]['index']) and i > 0):
    
        if 'X' in train_data:
            train_data['X'].append(x)
            train_data['Y'].append(y)
        else:
            train_data['X'] = [x]
            train_data['Y'] = [y]
    
        x = []
        y = []
    
    temp = []
    for j in train_df_1.iloc[i]['x']:
        temp.append(int(j))
    x.append(temp)
    y.append(int(train_df_1.iloc[i]['y']))
    for l in y:
        alphabet.add(l)
        
        
test_df_1 = pd.read_csv("data/nettalk_stress_test.txt", sep='\t', header=None, names=['index', 'x', 'y', 'delim'])
test_df_1["x"] = test_df_1["x"].str.replace("im", "")

test_data = {}
x = []
y = []
alphabet = set()
for i in range(len(test_df_1)):

    if ((test_df_1.iloc[i - 1]['index'] > test_df_1.iloc[i]['index']) and i > 0):
    
        if 'X' in test_data:
            test_data['X'].append(x)
            test_data['Y'].append(y)
        else:
            test_data['X'] = [x]
            test_data['Y'] = [y]
    
        x = []
        y = []
    
    temp = []
    for j in test_df_1.iloc[i]['x']:
        temp.append(int(j))
    x.append(temp)
    y.append(int(test_df_1.iloc[i]['y']))
    for l in y:
        alphabet.add(l)
        
token = len(train_data['X'][0][0])
        
for i in phi_order:
    W = structured_perceptron(phi, i, 10, 0.01, 20)
    W = structured_perceptron(phi, i, 10, 0.01, 20, W)

print()
print('Handwriting recognition Dataset\n') 

dataset = 1
                
train_df = pd.read_csv("data/ocr_fold0_sm_train.txt", sep='\t', header=None, names=['index', 'x', 'y', 'delim'])
train_df["x"] = train_df["x"].str.replace("im", "")
train_f = train_df['x'].isnull()
train_data = {}
x = []
y = []
alphabet = set()
for i in range(len(train_f)):
    if (train_f[i] == True):
        if 'X' in train_data:
            train_data['X'].append(x)
            train_data['Y'].append(y)
        else:
            train_data['X'] = [x]
            train_data['Y'] = [y]
        x = []
        y = []
    else:
        temp = []
        for j in train_df.iloc[i]['x']:
            temp.append(int(j))
        x.append(temp)
        y.append(ord(train_df.iloc[i]['y']) - 97)
        for l in y:
            alphabet.add(l)
            
            
test_df = pd.read_csv("data/ocr_fold0_sm_test.txt", sep='\t', header=None, names=['index', 'x', 'y', 'delim'])
test_df["x"] = test_df["x"].str.replace("im", "")
test_f = test_df['x'].isnull()
test_data = {}
x_test = []
y_test = []

for i in range(len(test_f)):
    if (test_f[i] == True):
        if 'X' in test_data:
            test_data['X'].append(x_test)
            test_data['Y'].append(y_test)
        else:
            test_data['X'] = [x_test]
            test_data['Y'] = [y_test]
        x_test = []
        y_test = []
    else:
        temp = []
        for j in test_df.iloc[i]['x']:
            temp.append(int(j))
        x_test.append(temp)
        y_test.append(ord(test_df.iloc[i]['y']) - 97)


token = len(train_data['X'][0][0])

for i in phi_order:
    W = structured_perceptron(phi, i, 10, 0.01, 20)
    W = structured_perceptron(phi, i, 10, 0.01, 20, W)


