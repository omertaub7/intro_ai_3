import matplotlib.pyplot as plt
from sklearn import tree, metrics
import numpy as np

import csv

#Read Train file


train_file = open('train.csv')
data = csv.reader(train_file, delimiter=',')
train_data, train_results = [], []
for row in data:
    train_data.append(row[0:8])
    train_results.append(row[8])
train_data, train_results=train_data[1:len(train_data)],train_results[1:len(train_results)]

#build And Train the tree
class_weight_dict={'0':0.2 , '1':0.8}
DT2 = tree.DecisionTreeClassifier(criterion="entropy",min_samples_split=9 , class_weight=class_weight_dict)
DT2.fit(train_data,train_results)



#Load test file
test_file = open('test.csv')
data = csv.reader(test_file, delimiter=',')
test_data,test_results = [], []
for row in data:
    test_data.append(row[0:8])
    test_results.append(row[8])
test_data, test_results=test_data[1:len(test_data)], test_results[1:len(test_results)]


#Test the tree
DT2_results = DT2.predict(test_data)
#Generate prediction tree
conf_mat1=metrics.confusion_matrix(test_results,DT2_results)
conf_mat1[0][0],conf_mat1[1][1]=conf_mat1[1][1],conf_mat1[0][0]
errw=conf_mat1[0][1]+4*conf_mat1[1][0];
print(conf_mat1)
print(errw)


