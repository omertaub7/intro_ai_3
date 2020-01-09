import matplotlib.pyplot as plt
from sklearn import tree, metrics
import graphviz
import csv
#Read Train file
clf = tree.DecisionTreeClassifier()
train_file = open('train.csv')
data = csv.reader(train_file, delimiter=',')
train_data, train_results = [], []
for row in data:
    train_data.append(row[0:8])
    train_results.append(row[8])
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
print(metrics.confusion_matrix(test_results,clf_results))

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

#Now we are going to prune the tree
for val in [3, 9, 27]:
    clf = tree.DecisionTreeClassifier(min_samples_split=val)
    clf.fit(train_data,train_results)
    print("~~~~~~~~~ X =", val, "~~~~~~~~~")
    clf_results = clf.predict(test_data)
    print(metrics.confusion_matrix(test_results,clf_results))
    tree.plot_tree(clf)
    plt.show()
