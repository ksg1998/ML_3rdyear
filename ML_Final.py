# -*- coding: utf-8 -*-
"""
@authors: GOUTHAM SEKAR
          Krithika Suwarna
          Sai Tarun
"""

from random import seed
from random import randrange
from random import random
import numpy as np
from math import *
import operator
from csv import reader
from math import exp
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split as tts
from ann_visualizer.visualize import ann_viz;
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score 


results1=[]

print("-------------------------------------Artificial Neural Networks-------------------------------------------------")


    
def string_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def data_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def scale(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            #if((minmax[i][1] - minmax[i][0])==0):
                #print(i)
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
def cleaning(dataset):
    
    clean_l=[]
    for cname in data.columns[0:]:
        clean=[]
        clean=list(data[cname])
        #print(clean)
        rm=clean
        #rm=set(clean)
        #print(rm)
        rm=list(rm)
        l = [x for x in rm if ~np.isnan(x)]
        l=set(l)
        #print(set(l))
        l=list(l)
        #print(len(l))
        if(len(l)<=1):
            clean_l.append(cname)
    return clean_l


def transfer_derivative(output):
    return output * (1.0 - output)

def transfer_derivative2(output):
    return 1.0 - (tanh(output))**2

def transfer_derivative3(output):
    if(output<0):
        return 1.0
    else:
        return 0.5

def kfold_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    
    for i in range(n_folds):
        #ctr=randrange(40)
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            #fold.append(dataset_copy[ctr])
            #ctr=ctr+1
        dataset_split.append(fold)
        
    return dataset_split

def train_test_split(dataset, split=0.70):
    train = list()
    train_size = split * len(dataset)
    #t1=len(dataset)-int(train_size)
    #print(train_size)
    #print(len(list(dataset)))
    dataset_copy = list(dataset)
    #print(dataset_copy)
    train1=dataset[0:int(train_size)]
    copy=dataset[int(train_size):]
     
    
    i=0
    #print(len(dataset_copy))
    while len(train) < train_size:
        #print(i)
        i=i+1
        index = randrange(len(dataset_copy))
        #print(index)
        train.append(dataset_copy.pop(index))
    return train1, copy


    
def ann_algo1(dataset, algorithm, n_folds, *args):
    folds = kfold_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train, [])
        test = list()
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train, test, *args)
        actual = [row[-1] for row in fold]
            
        cm = confusion_matrix(actual, predicted)
        #print(cm)
        print((actual,predicted))
        #print((actual,predicted))
        accuracy = accuracyMetric(actual, predicted)
        scores.append(accuracy)
    return scores

def ann_algo2(dataset, algorithm, *args):
    train,test = train_test_split(dataset)
    #print(len(test))
    scores = list()


    predicted = algorithm(train, test, *args)
    actual = [row[-1] for row in test]

    cm = confusion_matrix(actual, predicted)
    print(cm)
    #print((actual,predicted))
    print((actual,predicted))
    accuracy = accuracyMetric(actual, predicted)
    scores.append(accuracy)
    return scores

def activation(weights, inputs):
    activat1 = weights[-1]
    for i in range(len(weights)-1):
        activat1 += weights[i] * inputs[i]
    return activat1

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def tanh(activation):
    return np.tanh(activation)

def ReLU(activation):
    return max(activation,0)



def feedforward(network, row):
    inputs = row
    #print(network)
    for layer in network:
        new_inputs = []
        for neurons in layer:
            activat1 = activation(neurons['weights'], inputs)
            neurons['output'] = ReLU(activat1)
            new_inputs.append(neurons['output'])
        inputs = new_inputs
    return inputs


def delta_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        #print(layer)
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neurons in network[i + 1]:
                    error += (neurons['weights'][j] * neurons['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neurons = layer[j]
                errors.append(expected[j] - neurons['output'])
        for j in range(len(layer)):
            neurons = layer[j]
            neurons['delta'] = errors[j] * transfer_derivative3(neurons['output'])
            
def new_weights(network, row, l_rate):
    for i in range(len(network)):
        input1 = row[:-1]
        if i != 0:
            input1 = [neurons['output'] for neurons in network[i - 1]]
        for neurons in network[i]:
            for j in range(len(input1)):
                neurons['weights'][j] += l_rate * neurons['delta'] * input1[j]
            neurons['weights'][-1] += l_rate * neurons['delta']
            
def network_training(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            output = feedforward(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            delta_error(network, expected)
            new_weights(network, row, l_rate)
        
            
def network_initialization(n_inputs, n_hidden, n_outputs):
    network = list()
    hlayer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hlayer)
    olayer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(olayer)
    return network


def predict(network, row):
    #print(network)
    outputs = feedforward(network, row)
    return outputs.index(max(outputs))

def feedbackward(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    #print(n_inputs)
    #n_inputs = 1
    n_outputs = len(set([row[-1] for row in train]))
    #results1.append(n_outputs)
    #print(n_outputs)
    #n_outputs=1
    network = network_initialization(n_inputs, n_hidden, n_outputs)
    network_training(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)


def accuracyMetric(actual, predicted):
    correct = 0
    print('Precision Score : ',precision_score(actual,predicted))
    print('Recall Score : ',recall_score(actual,predicted))
    print('F Score : ',f1_score(actual,predicted))
    print('Classification Report : \n',classification_report(actual,predicted))
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


filename="LBW_1.csv"
data=pd.read_csv(filename)
#print(data)
ctr=0

#
clean_l=cleaning(data)
#print(i)
#print(clean_l)
    #if(len(rm)<=2)
    

for cname in data.columns[0:]:
    
    if is_numeric_dtype(data[cname]):
        
        if(data[cname].dtype=='float64'):
            
            #elif(cname=='weight1' or cname=='HB'):
                #data[cname]=data[cname].fillna(data[cname].mode())
            
            data[cname]=data[cname].fillna(method='ffill')
                #print(cname)
        else:
            data[cname]=data[cname].fillna(method='ffill')
            #print(cname)
    
#print(ctr)      


for i in clean_l:
    data.drop([i], axis = 1, inplace = True)
    
ctr=0
    
for cname in data.columns[0:]:
    ctr=ctr+1
#print(ctr)

#data.drop(["education"], axis = 1, inplace = True)
#data.drop(["community"], axis = 1, inplace = True)
#data.drop(["agelast"], axis = 1, inplace = True)
#data.drop(["age"], axis = 1, inplace = True)
#data.drop(["weight1"], axis = 1, inplace = True)
#data.drop(["education"], axis = 1, inplace = True)

#print(data)

d=data.values.tolist()

for i in range(len(d)):
    d[i][ctr-1]=str(int((d[i][ctr-1])))


#for i in range(len(data[0])-1):
    #str_column_to_float(data, i)
    
#print(d)

#print(d)


string_to_int(d, len(d[0])-1)

#print(d)


minmax = data_minmax(d)
scale(d, minmax)

n_folds = 10
l_rate = 0.1
n_epoch = 100
n_hidden =10
'''scores = ann_algo1(d, feedbackward,n_folds, l_rate, n_epoch, n_hidden)
#print(actualarr)
#print(predictedarr)
print('Scores: %s' % scores)
results.append((sum(scores)/float(len(scores))))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))'''

n_folds = 10
l_rate = 0.1
n_epoch = 500
n_hidden =10

avg = []

for i in range(0,5):
    scores = ann_algo2(d, feedbackward, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    results1.append((sum(scores)/float(len(scores))))
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    avg.append(sum(scores)/float(len(scores)))


average_acc = sum(avg)/len(avg)
print("Average Accuracy Score : ",average_acc)

print("------------------------------------------K-Nearest Neigbours---------------------------------------------------")

def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

filename="LBW_1.csv"
data=pd.read_csv(filename)
#print(data)
colname=[]
ctr=0
for cname in data.columns[0:]:
    colname.append(cname)
    if is_numeric_dtype(data[cname]):
        if(data[cname].dtype=='float64'):
            if(cname=='education'):
                data[cname]=data[cname].fillna(-1)
            data[cname]=data[cname].fillna(method='ffill')
        
        else:
            data[cname]=data[cname].fillna(method='ffill')
data.drop(data.columns[[7]], axis = 1, inplace = True)
colname.pop(-1)
colname.pop(7)
#print(data)
d=data.values.tolist()
#print(d)
#print(colname)

for i in range(len(d)):
    d[i][8]=int(str(int((d[i][8]))))
#print(d)

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            #if((minmax[i][1] - minmax[i][0])==0):
                #print(i)
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
minmax=dataset_minmax(d)
normalize_dataset(d, minmax)
targ=[]
#print(d)
for i in d:
    targ.append(i[-1])
#print(len(d))


feat1=[]

for i in d:
    feat1.append(i[0])
x=pd.DataFrame({colname[0]:feat1})
#print(x)
feat=[]
j=1
while(j<8):
    feat=[]
    for i in d:
        feat.append(i[j])
    x[colname[j]]=feat    
    j=j+1
    
#print(x)
#for i in d:
    #for j,k in zip(i,feature):
        #df=pd.DataFrame({k:j})
#print(df)
y=pd.DataFrame({'reslt':targ})
#print(dfy)
#print(feature)
target=['reslt']
X = data[colname] # Features
#print(X)

y_pred=[]

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
x_train1=x_train.values.tolist()
x_test1=x_test.values.tolist()

#print(x_test1)

'''def convert(data1):
    Row_list =[]  
    for index, rows in data1.iterrows(): 
        # Create list for the current row 
        my_list =[rows.age, rows.weight1, rows.history , rows.HB ,rows.IFA , rows.BP1 , rows.res] 
        Row_list.append(my_list) 
    # Print the list 
    return(Row_list)

x_test1=convert(x_test)
print(x_test1)
x_train1=convert(x_train)
print(x_train1)'''

def euclideanDist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return sqrt(distance)

def manhattanDist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance

def all_neigh1(test_row):
    result=[]
    for j in range(len(x_train1)):
            euclidean_dist=euclideanDist(test_row,x_train1[j],8)
            result.append([euclidean_dist,j])
    return(result)



def prediction(dists, k):
    predict=[]
    for l in range(k):
        #ind=result.index(dup[l])
        predict.append(y_train.iloc[dists[l][1],0])
    return(max(set(predict),key=predict.count))

def k_nearest1(k):
    for i in range(len(x_test1)):  
        new=all_neigh1(x_test1[i])
        new.sort(key= operator.itemgetter(0))
        x=prediction(new,k)
        y_pred.append(x)
        

        
actual=[]
def accuracy():
    ctr=0
    #print(y_pred)
    for i in range(len(y_pred)):
        actual.append(y_test.iloc[i,0])
        if(y_pred[i]==y_test.iloc[i,0]):
            ctr+=1
    #print(actual)
    cm=confusion_matrix(actual,y_pred)
    print(cm)
    print('Precision Score : ',precision_score(actual,y_pred))
    print('Recall Score : ',recall_score(actual,y_pred))
    print('F Score : ',f1_score(actual,y_pred))
    print('Classification Report : \n',classification_report(actual,y_pred))
    print("Accuracy :",(ctr/len(y_test))*100)


k_nearest1(7)
accuracy()












