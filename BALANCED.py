import matplotlib.pyplot as plt
from sklearn import tree, metrics

import csv

#Read Train file
clf = tree.DecisionTreeClassifier(criterion="entropy")
train_file = open('train.csv')
data =list(csv.reader(train_file, delimiter=','))
train_data, train_results = [], []

positives_count=sum( 1 for row in data if row[8]=='1' )

negatives_count=0

for row in data:
    if(row[8]=='0' and negatives_count >= positives_count): continue
    else:
        train_data.append(row[0:8])
        train_results.append(row[8])
        if(row[8]=='0'):
            negatives_count+=1


train_data, train_results=train_data[1:len(train_data)],train_results[1:len(train_results)]

#Train the tree
clf.fit(train_data,train_results)

#Load test file
test_file = open('test.csv')
data = csv.reader(test_file, delimiter=',')
test_data,test_results = [], []
for row in data:
    test_data.append(row[0:8])
    test_results.append(row[8])
test_data, test_results=test_data[1:len(test_data)], test_results[1:len(test_results)]
#Test the tree
clf_results = clf.predict(test_data)
#Generate prediction tree
conf_mat1=metrics.confusion_matrix(test_results,clf_results)
conf_mat1[0][0],conf_mat1[1][1]=conf_mat1[1][1],conf_mat1[0][0]
print(conf_mat1)
